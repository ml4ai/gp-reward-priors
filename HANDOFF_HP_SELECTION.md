# Hand-off: hyperparameter selection procedure (antmaze, all model families)

> Status 2026-08-16. **Stage 1 is complete for all three families; stage 3 is
> in progress** — medium_play's 4-chain rung is measured (§4.5, §10.2). The BNN's
> round-2 merged sweeps have all fired and their winners are transcribed into
> `scripts_bnn/antmaze_<v>_bnn_antmaze_eval.yaml`, verified field-by-field
> against wandb. Round 1's two-tier BNN design was discarded (§3.7); its results
> remain in §6 as the record. **Stage 3 (BNN draw budget) is under way — the
> exact next command is in §10.2; stage 4 not started.** MR and PT are
> unaffected throughout.
> Start at §10 if you are picking this up cold.
> Companion documents: `HANDOFF.md` (project + map-informed prior),
> `HANDOFF_CVAR_SAMPLER_2026-08.md` (sampler fixes and CVaR diagnostics),
> `fsghmc_sampler_fixes_handoff.md` (the external review those fixes came from).

---

## 0. Why this document exists

Every hyperparameter reported for MR, PT, and BNN on D4RL antmaze was chosen by
one pre-registered procedure, applied identically to all three model families.
The procedure is designed to be **auditable**: the budget, the stopping rule,
the selection metric, and the seed discipline were all fixed before the sweeps
ran, and nothing was adjusted in response to results. This document is the
specification; deviations from it are bugs, not judgment calls.

An earlier round of these sweeps **was** adjusted mid-flight (per-variant gates
and burn-in step sizes were changed after observing failures). That round was
discarded and everything re-run from scratch precisely so that no post-hoc
adjustment appears anywhere in the reported procedure. Do not re-introduce
reactive tuning.

---

## 1. Seed discipline — the load-bearing invariant

**Selection uses seed 0. Evaluation uses seeds 1–10. They never mix.**

This holds for *every* stage, including the stage-4 IQL grid search: there is a
single **seed-0 selection lineage** and ten separate **evaluation lineages**.

This matters more than it looks, because in this codebase `seed` does *not* only
control run-time pseudo-randomness (weight init, minibatch order, sampler
noise). It **also selects the data files**. The antmaze_eval scripts derive
their splits as:

```
{data_root}/{variant}/eval/seed_{seed}/{variant}_pref_{train,val,test}_{seed}.hdf5
```

So a sweep run at `seed: 0` trains on the seed-0 training split and is scored on
the seed-0 **validation** split — a partition of the preference data that no
evaluation run ever sees. The ten evaluation seeds each have their own disjoint
train/val/test partition. Consequently:

- No hyperparameter was ever selected on data used to produce a reported number.
- The seed-0 *test* split exists but is **not** used for selection either; it is
  incidental to the eval script's data loader.
- Reported evaluation results at seeds 1–10 are therefore out-of-sample with
  respect to the entire selection procedure, including the stage-4 normalization
  search (§5), which also runs at seed 0.

**If you add a stage, give it a selection seed outside 1–10.** The whole
argument collapses if any tuning touches an evaluation seed.

---

## 2. The stages

| stage | what is chosen | selected on | seed |
|---|---|---|---|
| 1 | model architecture + prior strength + (BNN) sampler schedule | validation loss | 0 |
| ~~2~~ | *merged into stage 1 in round 2 — see §3.7* | — | — |
| 3 | (BNN only) chain count / draws per chain | MCMC tail diagnostics | 0 |
| 4 | output normalization function | max mean IQL score over eval points | 0 |

Stage 1 is an automated wandb sweep — **one per (family × variant), for every
family alike**. Stage 3 is a deliberate manual step. Stage 4 is a small grid
search over the downstream RL objective.

**Stage 2 no longer exists.** In round 1 the BNN alone was split into two tiers,
a warm-up tier and a sampling tier; that split is what failed (§3.7). The number
2 is left vacant rather than renumbering stages 3 and 4, so that every existing
cross-reference and the round-1 record below stay valid.

---

## 3. Stage 1: sweep-based selection

### 3.1 Shared design (identical across all families)

**Search:** wandb `method: bayes`, **one sweep per (family × antmaze variant)**
— 12 sweeps in total, and the same shape for every family. The BNN's
architecture, prior strength and sampler schedule are searched *together* and
scored on one metric; there is no second tier and nothing is inherited between
sweeps.

**Trial budget: `run_cap: 130` for every family.**

| family | swept params (d) | run_cap |
|---|---|---|
| MR | 3 | 130 |
| PT | 4 | 130 |
| BNN | 9 | 130 |

This is the single most important fairness property: a reviewer cannot argue the
proposed method simply got tuned harder than the baselines. 130 is generous for
MR's 3-dimensional space — that is intentional, since over-tuning a *baseline*
is the safe direction.

**The cap is a safety limit, not the answer.** The stopping rule below is what
ends a sweep, and in round 1 it fired at trials 20–56, far short of the cap.
Note also that the invariant is one-sided: the argument only requires the BNN to
receive **no more** tuning than the baselines, so a BNN sweep that stops earlier
than they do strengthens the claim rather than weakening it.

**The BNN searches 9 dimensions against MR's 3 and PT's 4 at the same cap**, so
its coverage per dimension is thinner. That is a real cost of merging, accepted
deliberately: round 1 bought better coverage by factoring the search into two
tiers, and the factorisation rested on an independence assumption that turned
out to be false (§3.7). Thin honest coverage beats efficient coverage of the
wrong space.

**Selection metric: minimise validation loss.** Never accuracy. Accuracy is
insensitive to confidence, and the downstream use of these models is a
posterior-predictive quantity, so a calibrated loss is the right target.

| family | metric key | notes |
|---|---|---|
| MR, PT | `eval_loss_best` | requires `criteria_key: loss` in the config |
| BNN | `val_predictive_cross_entropy` | Wu et al. Eq. (10) predictive, `E[σ(f)]` — see §3.6.2 |

Round 1's BNN warm-up tier selected on `warmup_final_nll`, a quantity measured
before any sampling occurred. That metric is retired: §3.7 explains why it could
not see the failure that mattered.

**Stopping rule: stop when the best-so-far has not improved for K = 15
consecutive trials.** Applied uniformly to every sweep.

wandb has **no built-in convergence stop for bayes sweeps**, so this is
evaluated out-of-band with `check_sweep_convergence.py` (repo root). Trials
after the trigger point are discarded. Because the rule is deterministic given
the trial sequence, applying it retrospectively is equivalent to having stopped
there — but see §7 for the one case where that costs something.

**Always report both** the stopping-rule winner (primary, pre-registered) and
the best-of-all-trials value. They usually coincide; when they do not, disclose
the gap (§7).

### 3.2 Search spaces

`width`, `embd_dim`, `head_dim` are **log2 exponents** — the config stores the
exponent and `__post_init__` raises 2 to it. A sweep value of 8 means 256 units.

**MR** (`scripts_mr/sweep_antmaze_<variant>_mr_antmaze_eval.yaml`)

| param | distribution | range |
|---|---|---|
| `width` | int_uniform | 6–9 (64–512) |
| `depth` | int_uniform | 1–6 |
| `lr` | log_uniform_values | 1e-5 – 1e-2 |

Fixed: `epochs: 5000`, `criteria_key: loss`, `seed: 0`.

**PT** (`scripts_pt/sweep_antmaze_<variant>_pt_antmaze_eval.yaml`)

| param | distribution | range |
|---|---|---|
| `embd_dim` | int_uniform | 6–8 (64–256) |
| `head_dim` | int_uniform | 5–7 (32–128, clamped to embd_dim) |
| `num_layers` | int_uniform | 1–4 |
| `lr` | log_uniform_values | 1e-5 – 1e-2 |

Fixed: `epochs: 5000`, `criteria_key: loss`, `seed: 0`.

**BNN — merged sweep**
(`scripts_bnn/sweep_antmaze_<variant>_bnn_antmaze_eval.yaml`)

| param | distribution | range | vs round 1 |
|---|---|---|---|
| `width` | int_uniform | 6–10 (64–1024) | unchanged |
| `depth` | int_uniform | 2–6 | unchanged |
| `n_meas` | int_uniform | 0–64 (0 = functional prior off) | unchanged |
| `map_amp2` | log_uniform_values | 1 – **1e4** | **expanded** |
| `sghmc_lr` (lr_min *and* burn-in) | log_uniform_values | 5e-5 – 5e-4 | unchanged |
| `sghmc_lr_max` | log_uniform_values | 5e-4 – 5e-3 | unchanged |
| `cycle_length` | q_uniform | 500 – 3000, q = 250 | unchanged |
| `mdecay` | log_uniform_values | 1e-3 – **1.0** | **expanded** |
| `fraction_cool` | uniform | 0.1 – 0.5 | unchanged |

Fixed per trial: `num_chains: 4`, `chains_per_gpu: 4`, `num_samples: 75`,
`n_discarded: 5`, `num_burn_in_steps: 20000`, `samples_per_cycle: 1`,
`chain_init_jitter: 0.0`, `use_cyclical_lr: true`, `warmup_log_every: 250`,
`early_stop_acc_threshold: 0.0`, `seed: 0`.

`n_meas` and `map_amp2` are the two **prior-strength** knobs, and both are
searched, including `n_meas = 0` so the optimiser can reject the prior outright
if it does not help. `map_amp2` scales the whole map kernel (`K → amp2·K`),
changing the prior's reward amplitude while leaving its correlation structure —
the map-informed part — untouched. It exists because the `bt_pool="mean"` fix
tied per-point reward amplitude to Bradley–Terry logit magnitude; without it the
legacy O(1) prior amplitude is incompatible with the likelihood and the sweep
can only escape by driving `n_meas → 0`. See `HANDOFF_CVAR_SAMPLER_2026-08.md`.

The `sghmc_lr` ceiling meets the `sghmc_lr_max` floor at 5e-4, so
`lr_max ≥ lr_min` holds by construction. The 5e-3 cap on `lr_max` reflects a
measured divergence cliff under mean pooling (stable at 0.0048, divergent at
0.0064 for medium_play).

**Range changes and their justification.** Round 1's ranges are pre-registered
and §9 forbids *narrowing* them in response to results. Two were **expanded**,
on the narrow ground that a round-1 winner sat against a cap, which is evidence
the optimum lay outside the searched region:

- **`mdecay` 1e-1 → 1.0.** Round-1 winners reached 0.0953 (warm-up tier,
  medium_diverse) and 0.0892 (sampling tier, large_diverse) — 89–95% of the old
  cap.
- **`map_amp2` 1e3 → 1e4.** All four round-1 winners landed at 313–773, the top
  ~20% of a log-uniform 1–1e3 range. §7 had already flagged this as possibly
  range-limited.

Three caps were left alone despite round-1 winners approaching them:

- **`width` (10) and `depth` (6).** medium_diverse's warm-up-tier winner sat at
  *both* ceilings, but it reached them under `warmup_final_nll` — the metric
  round 2 retires precisely because it cannot see samplability (§3.7). A ceiling
  hit under a discarded metric is not evidence about the optimum under the new
  one. They are also not narrowed, even though the round-1 failure involved the
  largest network in range: narrowing on that basis is exactly the reactive
  tuning §9 prohibits.
- **`sghmc_lr_max` (5e-3).** The cap encodes a measured divergence cliff, a
  stability fact rather than a performance result.

**`sghmc_lr` now sets the burn-in step size as well as the cool-phase minimum**,
because there is no separate `burn_in_lr` (see below). Its range is unchanged,
since widening it upward would break the `lr_max ≥ lr_min` construction. One
consequence to be aware of: burn-in now runs at ≤ 5e-4 rather than round 1's
fixed 0.002. That follows from the pre-registered construction and was not
chosen for its effect.

**Burn-in length: 20,000 steps, not round 1's 5,000.** This follows from the
same construction. Round 1 burned in at a fixed 2e-3; round 2 burns in at the
swept `sghmc_lr`, capped at 5e-4 — 4× smaller at the ceiling and up to 40×
smaller at the floor. Holding the length fixed while shrinking the step would
under-burn every trial systematically, so the length is scaled by the same 4×
the ceiling shrank. Three of four round-1 warm-up winners used a step *above*
the new ceiling (1.64e-3, 6.96e-4, 1.96e-3), which is how far short 5,000 steps
would now fall.

It is deliberately **not** made adaptive to the drawn `sghmc_lr`. Burn-in
quality is part of the candidate under test in a merged design, so a
configuration needing more burn-in than it gets simply scores worse and the
optimiser avoids it. The fixed increase removes the systematic shortfall; it
does not try to equalise across the swept range.

**Per-trial horizon: 75 draws per chain, not round 1's 35.** Round 1 ranked
schedules at roughly 1/9 of the production draw count, and the failure it missed
only enters the metric with length — in the round-1 pilot, val CE ran 0.2953 →
0.3036 → 0.3411 at 33 / 75 / 150 draws. 35 draws cannot separate a drifting
configuration from a stable one; 75 can, at ~1/4 the cost of 150. Per-trial cost
is ~1.7 h at `cycle_length` 500 and ~10 h at 3000, ~6 h at mid-range.

**Set the stage-3 production horizon to this same 75 draws per chain, and buy
total draws with `num_chains`** (§4). Chains do not extend the horizon, so
selection and production run at the same horizon by construction and round 1's
mismatch cannot recur.

**`burn_in_lr` is deliberately absent, and must stay absent.** Burn-in inherits
the swept `sghmc_lr`. The base config a sweep points at must not set it — if it
does, burn-in silently uses that value instead and the point of merging is lost.
A `null` in the sweep cannot fix this: a wandb agent serialises null onto the
command line as the string `"None"`, which pyrallis rejects (§8). Remove the
line. Round 1's fixed `burn_in_lr: 0.002` is discussed in §3.7; it is the
proximate cause of the failure that ended that round.

One value deserves its own note because it looks like tuning and is not:

- **`early_stop_acc_threshold: 0.0` — the warm-up gate is DISABLED.** Every
  trial runs to completion and is ranked on `val_predictive_cross_entropy`;
  no proxy criterion is applied anywhere in the sweep. Accuracy is always ≥ 0,
  so the check `warmup_final_acc < 0.0` never fires; `0.0` rather than `null`
  because of a wandb/pyrallis interaction documented in §8. See §3.5 for why an
  earlier static 0.75 gate was removed, and §3.6 for what a divergent trial
  actually looks like — **`max_param_step: 0.5` is a crash guard, not a
  divergence guard**: it keeps a blown-up run from dying, but the chains can
  still reach Inf gradients and return a degenerate posterior.

### 3.3 What is deliberately NOT swept

- **Map-prior geometry: `map_eta`, `map_sig_c2`, `map_sig_g2`, `map_sig_n2`.**
  These are prior-*design* choices fixed from the maze layout and prior-sample
  diagnostics. Tuning them on any downstream signal would smuggle the
  inferential target back into the prior. Only prior *strength* (`n_meas`,
  `map_amp2`) is searched.
- **Shared likelihood: `bt_pool: "mean"`.** All three families must share one
  Bradley–Terry likelihood for comparability. Not a free parameter.
- **Sampler safety: `max_param_step: 0.5`, `clip_grad_norm_value: 100.0` with
  `clip_during_sampling: false`.** The clip is scoped to burn-in because
  clipping during sampling is non-measure-preserving and biases the CVaR tail.
- **`batch_size: 64`, `epochs: 5000`, `num_burn_in_steps: 20000`** (round 1
  used 5,000; see §3.2 for why the length changed with the step size).

---

### 3.4 Where the operative configuration actually lives

**Read values from the sweep yaml or the config yaml, never from a dataclass
default.** The `TrainConfig` defaults in the `run_*.py` scripts are fallbacks;
they are the operative value only for settings that are genuinely uniform across
the whole project. Anything variant-specific, family-specific, or
selection-specific is set explicitly in:

- `scripts_<family>/sweep_antmaze_<variant>_*.yaml` — what the sweep searched
  and what it pinned;
- `scripts_<family>/antmaze_<variant>_*_antmaze_eval.yaml` — the production
  configuration, including the transcribed winners;
- the IQL run config for stage 4.

This matters when reconstructing what a run actually did: a default that looks
authoritative in the script may have been overridden by pyrallis from the yaml
or by the sweep agent's CLI arguments, and the resolved values are what wandb
records in each run's `config`.

### 3.5 Why there is no warm-up gate

The reasoning below was established in round 1 and carries over unchanged: the
merged sweep also runs every trial to completion.

Round 1's sampling tier originally early-stopped any trial whose warm-up
accuracy fell below a static 0.75, to avoid spending ~6 h sampling from a broken
starting point. That gate was **removed and all four sweeps restarted**, for two
reasons.

**It rejected on the wrong quantity.** The gate reads warm-up *accuracy*, which
is not the selection metric. Anything it rejects is excluded from the search
without ever being scored on `val_mean_cross_entropy`.

**It became a hard wall in `mdecay`.** Because `burn_in_lr`, `seed` and the
architecture are fixed, and `cycle_length`/`fraction_cool`/`lr_min`/`lr_max`
only act after burn-in, warm-up outcome is a deterministic function of `mdecay`
alone — friction acts during burn-in as well as sampling. Measured on the
discarded run, pass/fail separated perfectly by `mdecay` in every variant:

| variant | failed at mdecay ≤ | passed at mdecay ≥ | incumbent's mdecay |
|---|---|---|---|
| medium_play | (none) | 8.58e-3 | 1.79e-2 |
| large_diverse | 1.653e-3 | 1.957e-3 | 3.78e-2 |
| large_play | 3.567e-3 | 9.520e-3 | 5.32e-2 |
| **medium_diverse** | **2.249e-2** | **2.362e-2** | **2.362e-2** |

For medium_diverse the wall removed roughly the bottom 60% of the swept
`mdecay` range, and its best configuration sat *on* the wall — at the lowest
passing value observed — while its val CE improved monotonically as `mdecay`
fell toward it. The gate was blocking the exact direction the objective wanted.
Schedules were being rejected for a **burn-in** reason even where they might
sample well.

The gate was also not paying for itself: across 47 completed trials it stopped
10 (21%) and saved ~60 of ~294 GPU-hours, and 6 of those 10 were
medium_diverse's — i.e. most of the "saving" was the harmful kind.

**What removing it does and does not fix.** It converts a hard rejection into a
soft penalty: a low-friction schedule still burns in badly and will still likely
score a poor val CE. The optimizer will still learn to avoid low `mdecay`. The
gain is that the *data* now decides whether the cyclical hot phases can recover
from a poor warm-up, rather than a threshold assuming they cannot. The complete
fix would be a `burn_in_mdecay` decoupling friction the way `burn_in_lr`
decouples step size; that was **not** done, since it is a mid-procedure code
change, and the residual confound is disclosed instead (§7).

### 3.6 What a divergent trial looks like

With no gate, low-friction schedules run to completion and are scored. Some of
them **diverge numerically**. Observed on round 1's restarted sampling-tier
sweeps; the signatures apply unchanged to the merged sweep:

| warm-up NLL | val CE | gradnorm max | %>clip | outcome |
|---|---|---|---|---|
| 0.24 – 0.51 (8 trials) | 0.236 – 0.299 | 3.6 – 1.8e3 | 0 – 0.04% | healthy |
| 0.95 | 0.692 | 6.7e14 | 1.95% | near-degenerate |
| 2.78 | 0.306 | 1.5e5 | 0.52% | poor |
| 23.8 | 0.858 | 3.0e11 | 62% | worse than chance |
| **350** | 0.693 | **Inf** | 12.7% | **diverged: all diagnostics NaN** |

Three things to know about these.

**`max_param_step: 0.5` is a crash guard, not a divergence guard.** The
diverged trial completed and logged; it did not crash. But its gradients reached
`Inf`, its per-chain and predictive variances, R-hat, ESS and every CVaR
diagnostic came back `NaN`, and its reward function collapsed to a constant.

**Cross-entropy is not monotone in brokenness** (true of both the plug-in
`val_mean_cross_entropy` and the selection metric
`val_predictive_cross_entropy`). A run that collapses
to constant output scores exactly `ln 2 = 0.6931` — every pair gets p = 0.5 —
which is *better* than a confidently-wrong run (0.858 above). Among failures,
total collapse looks less bad than partial failure. Harmless for ranking here,
since both are far from the ~0.25 healthy range, but do not read the metric as a
severity scale.

**Warm-up *accuracy* is blind to this; warm-up *NLL* is not.** The diverged trial
had `warmup_final_acc = 0.710`, which reads as mediocre-but-usable, while its
`warmup_final_nll` was **350** — confidently wrong, the signature of a weight
blow-up. This is a second, independent reason the old accuracy gate was the
wrong instrument (§3.5): it could not see the failure mode it was nominally
there to catch. `warmup_final_nll` — stage 1's own selection metric — separates
these cases cleanly, and `gradnorm_sampling_pct_over_clip` separates them after
the fact (0–0.04% healthy vs 0.5–62% troubled).

**Detection.** `check_sweep_convergence.py` flags these automatically: a trial
with NaN/Inf in the convergence diagnostics, or `gradnorm_sampling_pct_over_clip`
above 1%, is reported under `!! DIVERGED`. Such trials still count toward the
search — the optimiser scored them — but are **unusable for stage 3**, which
needs ESS and R-hat. If a stage-2 *winner* is ever flagged this way, do not
carry it forward.

### 3.6.1 The `!! DIVERGED` flag is two different things

Reviewed 2026-08-08. The flag fires on either of two conditions, and in practice
they identify **different populations**. Read the underlying numbers before
acting on the label; do not treat "6 diverged" as six blow-ups.

*Genuine failure* — NaN/Inf diagnostics, or a nonsense metric. In medium_diverse
(`o9g70yby`) three of six flagged trials are this kind: `5yj857hf` (12.7% over
clip, every diagnostic NaN), `00spgb6j` (9.4%, val CE 46.1), `cie2dj2r` (2.0%,
collapsed to `ln 2`). These are the §3.6 failure mode and are correctly excluded.

*Threshold call on a continuum* — the other three (`9dyx7pi2`, `oi3s2q3x`,
`ff384ml8`) sit at 1.1–1.8% over clip with healthy warm-up NLL (0.46–0.59),
finite diagnostics (CVaR ESS 13–19, CVaR R-hat 1.17–1.26) and val CE 0.300–0.305.
The clean leader `nfqz8f11` is at 0.21% and clean `lzyn872v` at 0.90% — the 1%
cutoff is splitting a smooth range, not separating two populations.

This matters because those three score better than **every** clean trial in that
sweep except the leader. If one of them ever takes the lead, the rule in §3.6
("do not carry a flagged winner forward") would discard a plausibly usable
configuration on a 1% cutoff.

**Decide this before it happens, not after.** Both defensible options are:
(a) keep the pre-registered rule as written and accept the cost, or (b) report
`gradnorm_sampling_pct_over_clip` as a graded diagnostic and gate on the
NaN/Inf condition alone. Either is fine; choosing in response to a specific
trial's result is exactly the reactive tuning §0 exists to prevent.

### 3.6.2 Theory alignment: what the metrics must measure

Audited 2026-08-11 against Wu et al. (2025), *Functional Stochastic Gradient
MCMC for Bayesian Neural Networks* (AISTATS), the paper this sampler
implements. **Read this before adding or interpreting any BNN diagnostic.**

**The object of inference is f, not w.** Proposition 3.2 establishes that the
stationary measure of the functional Hamiltonian dynamics is the function-space
posterior `P_{f|D}`, with potential `U(f) = Φ(f) + I₀(f)` — a functional of f
alone (`Φ` the likelihood, `I₀` the Onsager–Machlup functional of the functional
prior). The parameter-space update (Eq. 9) is a reparameterisation whose induced
measure on f is the target. The code matches: `f_pref_net.py` takes no
weight-space prior, and the prior enters only as `∇_w log p_GP = −Jᵀ K⁻¹(f−m)`.

Two consequences that are easy to get wrong, and were:

- **Weight-space diagnostics measure nothing here.** Since `U` depends on w only
  through f, every direction that leaves f unchanged is a flat direction, along
  which the chain performs an *unconfined random walk*. A growing weight norm is
  what this sampler is supposed to do. Measured on a round-1 production run,
  ‖w‖ followed the free-diffusion law ‖w‖² = ‖w₀‖² + c·t to within 5% across
  310 draws. `param_rhat`, `param_ess`, `param_within_chain_var` and the
  sampling weight-norm statistics have been **removed** for this reason.

  **Re-confirmed on round 2** (medium_play `c8`, 8 chains × 75 draws,
  2026-08-17), which matters because the round-1 measurement above was taken on
  the design §3.7 discarded. Per-draw `Δ‖w‖²` over the three printed intervals
  was 0.3310 / 0.3263 / 0.3118 — linear in draw index to ~3%, with RMS ‖w‖
  growing 1.9838 → 5.2756 (2.66×). Per-chain growth spanned 2.44×–2.78×, i.e.
  the rate `c` varies ~1.30× across chains. Textbook free diffusion, on a
  healthy run that passes every function-space check. **This is the number to
  cite when explaining why a growing weight norm is not evidence of anything**,
  and it is why the `--weight-trace` flag that produced it was subsequently
  deleted rather than kept (§4.5): having confirmed the law, there is no
  standing question left for a weight-space statistic to answer.
- **Stationarity is a claim about f.** `util.function_space_drift` compares the
  first half of each chain's draws against the second, **relative to that
  comparison's own Monte Carlo error**:

  | metric | read as |
  |---|---|
  | `fn_drift_loc_z_median` | location shift ÷ MCSE. Stationary ⇒ **~0.67**, 95th **~2** |
  | `fn_drift_scale_z_median` | \|log(sd₂/sd₁)\| ÷ its ESS-based SE. Same scale |
  | `fn_drift_loc_sd_median` | raw shift in posterior-sd units — magnitude only |
  | `fn_drift_scale_ratio_median` | raw sd₂/sd₁ — magnitude only |

  **Do not read the raw sd-unit shift as evidence.** It has no fixed null: on
  *stationary* AR(1) chains at the production shape (4 chains × 75 draws) it
  rises from 0.08 to 0.31 as autocorrelation goes ρ = 0 → 0.95, while the
  z-score stays flat at ~0.6–0.7. An early round-2 reading of ~0.27 raw looked
  like drift and was almost certainly just autocorrelation. The z-scores divide
  by an MCSE computed from ESS, so they are on the same scale whatever the draw
  budget or the mixing rate, and they still fire on real problems (a location
  trend gives z ≈ 2.7; a 4× spread ramp gives z_scale ≈ 2.6).

**The predictive is `E[σ(f)]`, not `σ(E[f])`.** Equation (10) defines
`p(y*|x*,D) ≈ (1/S) Σⱼ p(y*|f(x*; wⱼ))` — the *likelihood* averaged over draws.
Round 2 originally selected on `val_mean_cross_entropy`, which averages the
*reward* over draws and squashes once. That plug-in is blind to posterior width:
holding the mean reward fixed and inflating the posterior 6× moves it by 0.010
while the predictive moves by 0.174. Selecting on a width-blind statistic while
deploying CVaR₀.₀₅ — a functional of exactly that width — was a misalignment
between the procedure and the method it evaluates. **The selection metric is now
`val_predictive_cross_entropy`;** the plug-in is still logged for comparison.

**Verified to match Algorithm 2**: momentum resampled from `N(0, M)` at each
cycle onset (not zeroed); a fresh measurement set `X_M` drawn every inner step;
one sample collected per outer iteration.

**A consequence of mean pooling worth understanding, not just disclosing.** With
T = 100, `bt_pool: "mean"` divides the Bradley–Terry logit by exactly 100
relative to the return convention. Matching that logit scale requires rewards
100× larger, and `map_amp2` scales the *kernel*, so reward sd goes as
`√map_amp2` — putting the natural prior amplitude at roughly **100² = 10⁴ times**
the sum-pooling value. This is not a tuning observation; it follows from the
likelihood.

It also explains a pattern that had been read as an empirical finding. Round 1
capped `map_amp2` at 1e3 and all four winners landed at 313–773; round 2 capped
at 1e4 and two leaders landed at 6647 and 8699. In both rounds the cap sat at or
below the scale the likelihood implies, so the search was boundary-limited, and
"every variant prefers a large `map_amp2`" was substantially a statement about
the cap. The range is now 1–1e6, giving two decades above the predicted scale.
**Re-read the prior-strength result in that light**: the informative question is
whether `map_amp2` lands near 10⁴ (the likelihood's own scale, i.e. the prior is
doing nothing beyond matching units) or materially away from it (the prior is
carrying real information).

**Known deviations, to disclose:**

| deviation | status |
|---|---|
| **Cyclical step size** (Zhang et al. 2020) in place of the paper's decaying ε | Deliberate — standard SGHMC gets trapped in a single basin. Correctly implemented (samples taken only at cool-phase end, momentum resampled at cycle start, structurally close to Alg. 2's outer loop). But the *composition* of cSGMCMC with fSGHMC is analysed by neither paper. `function_space_drift` is the empirical check that early and late cycles are one measure. **2026-08-18: that check fired** — medium_play carries a location drift common to 14 of 16 chains (alignment 0.7564) that does not shrink with chain count. **2026-08-19: the schedule was tested directly and CLEARED** (§4.3.6). At matched compute, `use_cyclical_lr False` made the drift *worse* on every axis (`loc_sd` 0.4222 → 0.5443, `scale_ratio` 1.4361 → 2.1326, `scale_z` 1.3564 → 2.5834, failing a gate it had passed), so the drift is intrinsic to the sampler or model, not the schedule. **That run also supplies the empirical support this row previously lacked:** without cycling, predictive CE degrades 0.2029 → 0.3580 and accuracy drops 6.3 points. Report the composition as tested and retained on evidence, not merely assumed. **One new caveat to disclose:** the cyclical run's average LR is ~2.25× the constant-LR run's, yet its spread grows far *less* (1.44× vs 2.13×), so the per-cycle anneal appears to re-concentrate the chain. Sampling only at the cold point may therefore under-disperse relative to the true posterior — untested, and not measured by `scale_ratio`, which is a growth ratio rather than an absolute width. |
| **`max_param_step: 0.5`** clamps momentum every step, sampling included | Not measure-preserving when it binds, and unlike the gradient clip it was never scoped to burn-in. Now instrumented: `param_clamp_sampling_pct` must be ~0 on any selected run, else that run sampled the wrong measure. |
| **Scale** | The paper's networks are 141–10,401 parameters, converging in 500–2,000 iterations. `width: 10, depth: 6` is ~6.3M. Nothing in the paper supports that regime. |
| **`bt_pool: "mean"`** — the likelihood `Φ(f)` pools rewards by masked *mean* over timesteps, where the preference-learning literature uses the *sum* (return) | **Resolved 2026-08-11.** Applied identically in MR, PT and BNN, so cross-family comparability holds. Every segment is exactly T=100 valid timesteps (verified, all four variants, train and val), so mean = sum/100 *exactly*: the two are the same model up to a global temperature, and no length confound exists. One sentence in the paper, no further action. |

### 3.6.3 Winner acceptance criteria

Pre-registered 2026-08-12, **before any round-2 sweep fired**. A configuration
that samples something other than `P_{f|D}` is not a valid winner however good
its score, so selection is a *constrained* minimisation: the winner is the
**lowest `val_predictive_cross_entropy` among ELIGIBLE trials up to the stopping
trigger**.

**Eligibility.** All three must hold:

| criterion | threshold | why this number |
|---|---|---|
| `val_fn_drift_loc_z_median` | ≤ 2.0 | The per-point stationary null is \|N(0,1)\|: median ~0.67, 95th ~2. A median at 2.0 means the *typical* point has shifted 3× the null median, so this is deliberately lenient — set to avoid rejecting on noise, not to be strict. |
| `val_fn_drift_scale_z_median` | ≤ 2.0 | Same null, same reasoning. |
| `param_clamp_sampling_pct` | ≤ 0.01% | `max_param_step` is not measure-preserving when it binds. 0 is the exact null; observed values are 0.0000–0.0030%, so this is "inert" with margin. |

**Divergence.** A trial with NaN/Inf convergence diagnostics is ineligible — it
is unusable, not merely suspect. `gradnorm_sampling_pct_over_clip` is
**reported but does NOT gate**, resolving the question §3.6.1 left open. The
reason is that the gradient clip is disabled during sampling
(`clip_during_sampling: false`), so that percentage measures a *symptom* of
large gradients rather than a distortion of the measure; the two criteria that
do measure distortion — the momentum clamp and function-space drift — now carry
the decision. The 1% cutoff was always a threshold on a continuum (§3.6.1).

**Procedure when a sweep fires:**

1. Take trials up to the trigger in ascending `val_predictive_cross_entropy`.
2. For each in turn: if it lacks `fn_drift_*` (it predates the metric, §3.6.2),
   **re-run that exact config once** to populate diagnostics. Re-runs are for
   diagnostics ONLY — **rank on the original trial's CE**, never the re-run's.
   Nominally same config and seed, so a re-run should reproduce closely (the
   `limlikvn` re-run matched to six significant figures), but it is formally a
   second draw and taking whichever score came out better would be selection
   bias.

   **Launch a diagnostic re-run with `OUT_DIR` containing `diag_rerun`** — that
   substring is how `check_winner_eligibility.py` finds it. The re-run is a
   separate wandb run, so the sweep trial's own summary still shows nothing;
   without the marker the diagnostics exist but are invisible at winner
   selection, which is exactly how `limlikvn` first appeared ineligible when it
   was not. A distinct `OUT_DIR` is required anyway, or the re-run clobbers
   whatever that sweep's in-flight trial is writing (§8).
3. The winner is the first trial that satisfies all three criteria.
   `check_winner_eligibility.py` (repo root) does steps 1–3 mechanically: it
   ranks by the metric, applies the thresholds, resolves paired diagnostic
   re-runs by matching swept parameters, and reports the winner, the gap to the
   lowest-metric trial, and how many trials were rejected.
4. **Do not extend the search because the best trial was ineligible.** The
   stopping rule governs how many trials the search gets; these criteria govern
   which are eligible. Resuming to find a better-behaved configuration is
   exactly the result-driven extension §9 prohibits.

   The stopping rule is deliberately **not** eligibility-aware, and it was
   considered. Making it so is defensible statistically — right now the counter
   can be reset by a trial that cannot win — and the obvious fairness objection
   fails, since MR and PT have no eligibility constraint and the rule would be
   formally identical across families. It fails on the **budget** instead:
   eligibility-aware stopping can only ever *extend* a sweep, so the BNN would
   systematically receive more trials than the baselines, and §3.1's invariant
   is one-sided — the proposed method must get no **more** tuning. Changing a
   pre-registered rule after observing that ineligible runs reset counters would
   also be reactive under §9.

   **The cost is real and is disclosed, not fixed:** because the counter tracks
   the raw metric, a sweep can fire while the *eligible* frontier is still
   improving — the search stops on progress it cannot use. This is the known
   limitation of applying acceptance as a filter over an unconstrained search.
   `check_winner_eligibility.py` reports when the best eligible trial last
   improved, and warns if that was within K trials of the trigger. If it warns,
   say so in the write-up.
5. If **no** trial in the search is eligible, that is a finding about the
   sampler at these settings, not a licence to keep drawing. Resume only then,
   and disclose that the search was extended and why.

**Disclose:** the eligibility rule, the thresholds and their derivation from the
stationary null, how many trials each sweep rejected, and each winner's three
numbers. If a winner was not the lowest-CE trial, report the gap — the same
convention §7 uses for rule-vs-best.

### 3.7 Why round 1 was discarded: the two-tier design

Round 1 split the BNN search into a **warm-up tier** (architecture + prior
strength, 70 trials, selected on `warmup_final_nll` with sampling disabled) and
a **sampling tier** (schedule, 60 trials, selected on `val_mean_cross_entropy`,
inheriting the warm-up tier's winner). Both tiers completed and all eight sweeps
fired; their results are kept in §6 as the record. The design was then
discarded, before any stage-3 or stage-4 result was produced, when the first
production-budget run exposed a failure the procedure could not have caught.

**The premise.** Factoring the search into tiers is only valid if the best
architecture is roughly independent of the sampler schedule. That assumption
was never stated as an assumption, and it is false.

**What the pilot showed.** The medium_diverse stage-2 winner, run at the
production budget (8 chains × 310 draws, seed 0), produced:

| | sampling tier (4 × 35) | production pilot (8 × 310) |
|---|---|---|
| `val_mean_cross_entropy` | 0.2843 | **0.4252** |
| `gradnorm_sampling_pct_over_clip` | 0.21% | **6.25%** |
| `gradnorm_sampling_max` | 3.6e3 | **8.3e10** |
| `gradnorm_burnin_max` | 142.76 | 142.76 *(identical)* |

Identical burn-in maxima confirm both runs started from the same point, so the
divergence is entirely in the sampling phase. An offline draw ladder over the
saved chains showed CE degrading **monotonically** with draws — 0.2953, 0.3036,
0.3411, 0.3814, 0.4252 at 33/75/150/225/310 — with accuracy nearly flat
(0.8668 → 0.8551). That is the confidently-wrong signature: under
`bt_pool: "mean"` a growing reward magnitude inflates the Bradley–Terry logit
and hence CE while leaving the ranking intact. A weight trace confirmed the
mechanism directly: `avg |w|` grew 5.74 → 23.50, a factor of 4.09, monotonically,
in **all eight chains** (3.17×–4.30×).

**Every convergence diagnostic improved as this happened.** CVaR ESS median rose
64 → 1802, CVaR R-hat median fell 1.1454 → 1.0382, bulk R-hat median 1.560 →
1.108. This is not coincidence and not a quirk of these estimators: a drifting
chain has a large and growing *within*-chain variance, which pushes R-hat toward
1 and inflates ESS. **The tail diagnostics are not merely blind to this failure;
they are fooled by it, in the direction that makes a diverging run look
converged.** Anything selected or certified on them alone is unsafe when drift
is possible, which is why stationarity is now measured directly and gated on:
`fn_drift_loc_z_median` / `fn_drift_scale_z_median` (§3.6.2, §3.6.3).

*Historical note:* the first response to this was a weight-norm statistic,
`sampling_weight_growth`. It was later removed — the weight norm of this sampler
diffuses freely by construction and measures nothing about convergence (§3.6.2).
The function-space drift metrics replaced it. Do not go looking for the weight
statistic; it is gone.

**Three separate seams contributed, all products of the split:**

1. **The warm-up tier's metric could not see samplability.**
   `early_stop_acc_threshold: 1.01` made every tier-1 trial stop before
   sampling, so `warmup_final_nll` scored architectures on a quantity measured
   before a single posterior draw existed. medium_diverse's winner was width 10,
   depth 6 — both *range ceilings*, the largest network available — and by its
   own metric it was healthy (`avg |w|` 2.10, NLL 0.317).
2. **The tiers disagreed about burn-in.** Tier 2 replaced tier 1's own swept
   `sghmc_lr` with a fixed `burn_in_lr: 0.002`, described in §3.2 as a design
   fix. It is a good approximation for the two variants that behaved (their
   tier-1 winners used 1.64e-3 and 1.96e-3) and a poor one for the two depth-6
   variants (6.96e-4 and 2.56e-4 — 2.9× and 7.8× smaller). Measured cost, in
   burn-in NLL: medium_play +42%, large_diverse +22%, **medium_diverse +54%,
   large_play +111%**. medium_diverse began sampling from weights 1.9× larger
   than its tier-1 winner produced. This is the proximate cause.
3. **The horizon was 1/9 of production**, and the sampling tier's objective was
   *biased toward* instability rather than merely blind to it: at 35 draws a
   larger step size buys better CE and its cost has not yet appeared.
   medium_diverse's winning `sghmc_lr` was 4.959e-4 against a range ceiling of
   5e-4 — 99.2% of the boundary.

**Why merging fixes this structurally rather than by patching.** Nothing is
inherited, so seam 2 cannot exist and the §10.3 transcription trap disappears
with it. One metric, applied after sampling, so seam 1 cannot exist. A per-trial
horizon matched to production (§3.2) closes seam 3. And §3.5's "residual
burn-in/sampling confound" stops being a confound at all: friction acting during
burn-in as well as sampling is only a confound when a *separately chosen*
architecture is being held fixed — when the whole package is scored end to end,
it is simply a property of the candidate under test. The `burn_in_mdecay`
decoupling §3.5 identified but declined as a mid-procedure change is not needed.

**What round 1 still costs, and what to disclose.** Roughly 400 GPU-hours across
both tiers plus the ~50-hour pilot. The paper should state that the BNN
hyperparameters come from a single merged sweep matched in trial budget to the
baselines, and — because it is the honest account of how the procedure was
arrived at — that an earlier two-tier design was discarded when its selected
configuration proved non-stationary at production length. §0's standing rule
applies unchanged: the round was restarted from scratch rather than patched, so
no post-hoc adjustment appears anywhere in the reported procedure.

---

## 4. Stage 3 (BNN only): the draw budget

**Read §3.6.2 before this section.** It establishes what these dynamics target
and therefore which diagnostics mean anything; several natural-looking
statistics measure nothing here.

### 4.1 What stage 3 chooses

**`num_chains` only.** `chains_per_gpu` follows from it as placement. Everything
else — the nine selected fields, `num_samples`, `num_burn_in_steps` — stays
exactly as transcribed.

**Correction (2026-08-18): stage 3 also owns `chain_init_jitter` and
`samples_per_cycle`.** Every sweep config holds both fixed with the note that
"their real effects (wall-clock vs tail-ESS; R-hat honesty) are invisible to a
val-loss metric, so they are set with the tail diagnostics after this sweep, not
searched by it" — and *this* is that stage. This section said `num_chains` only,
so the ladder climbed the one axis the sweeps never deferred, while the two they
did defer stayed at their placeholders. `chain_init_jitter` is still **0.0**,
which gives every chain an identical start; `f_pref_net.py:136` skips the jitter
entirely at that value, and its own comment warns that shared starts
"under-estimate R-hat (chains are artificially similar)". That is a live threat
to every R-hat in §4.3 and §4.5 and it is where §4.3.2's drift most likely comes
from — see §4.3.3 and §4.3.4.

When jitter *is* non-zero it is applied **once per chain, in the worker
process** ([f_pref_net.py:126–143](optbnn/sgmcmc_bayes_net/f_pref_net.py:126)):
after the shared warm-up weights are copied in, and **before** that chain's own
`num_burn_in_steps` burn-in and any sampling. Per parameter tensor it adds
`randn_like(param) * chain_init_jitter * std(param)` — a *relative* perturbation
scaled by each tensor's own spread, drawn from the chain's RNG stream
(`set_seed(seed + chain_idx)`), so it is deterministic in (seed, chain index)
and tensors with zero spread are left alone. Note the ordering consequence: the
per-chain burn-in runs *after* the jitter, so a small jitter can be substantially
washed out before the first draw is kept, and the dispersion that reaches the
samples is not the dispersion that was injected.

**`num_samples` is pinned at 75 and is NOT a free parameter.** Stage 1 scored
candidates at 75 draws per chain, so 75 is the horizon at which the winners were
certified. Total draws are bought with chains, which do not extend any chain's
horizon, so selection and production run at the same horizon by construction.

This is the round-1 mistake, and it is worth understanding rather than just
obeying: round 1 selected at 35 draws and deployed at 310, and the selected
configuration was already non-stationary by draw 33 (§3.7). **If the tail
diagnostics cannot be satisfied by adding chains at a fixed horizon, that is a
finding about the schedule — not a licence to lengthen the chains.**

**Why it is not a sweep.** The quantity being improved is Monte-Carlo error on
the downstream CVaR₀.₀₅, not model quality, and it is not visible in
`val_predictive_cross_entropy` — a chain can have excellent predictive CE and
still resolve the lower 5% tail poorly.

Note that "more draws always help" is **false here** and was falsified in round
1: past a point, additional draws came from a drifting chain and made the model
worse. That is why §4.2 gates on stationarity first.

### 4.2 Prerequisite: re-check stationarity at the new chain count

Before reading any tail number, confirm the run is still sampling the target:

    val_fn_drift_loc_z_median    <= 2.0
    val_fn_drift_scale_z_median  <= 2.0     (stationary null: median ~0.67, 95th ~2)
    param_clamp_sampling_pct     <= 0.01%

These are the §3.6.3 criteria the winners already satisfy at 4 chains × 75
draws. **Changing the chain count does not automatically preserve them** — more
chains means more chances that one wanders — so re-check rather than assume. A
run failing these is not sampling `P_{f|D}`, and every tail statistic computed
from it is meaningless regardless of how good it looks.

### 4.2.1 The z-gate is not comparable across rungs — read the effect size too

**A rung can fail the gate above purely by being measured better.** From
`optbnn/utils/util.py:431`,

    z_loc = |m2 - m1| / sqrt(mcse1^2 + mcse2^2)

The numerator is a first-half-vs-second-half shift in function space. The
denominator is an MCSE, which falls as ~1/√C. So `loc_z` is a *significance
test whose power grows with chain count*, and holding it to a fixed 2.0
tightens the criterion mechanically as you climb the ladder — doubling the
chains inflates `z` by ~√2 on power alone, with the sampler behaving
identically. `scale_z` has the same 1/√ESS denominator and the same property.

The companion raw metrics carry no chain-count dependence and are what say
whether the drift is actually *larger*:

    val_fn_drift_loc_sd_median        |m2 - m1| in sd units
    val_fn_drift_scale_ratio_median   sd2 / sd1

Always read the pair. `stage3_ladder.py` now prints both and divides each
rung-to-rung `z` change into its power component (√ of the draw ratio) and the
real component that survives it, with the raw ratio as an independent check.

**The sharper test is on `loc_sd` alone.** It is a difference of means over
*all* draws, so under a stationary sampler it must fall as 1/√(total draws). A
`loc_sd` that stays flat while the draws multiply is a real non-stationarity —
and, critically, one that was already present in the rungs that *passed*, which
merely lacked the power to resolve it. A PASS at a low chain count is therefore
**not** evidence of stationarity; it is evidence of not having looked hard
enough. §4.3.2 is the worked case.

This does not make the gate useless — a run that fails it is still not to be
trusted. It means a failure is a prompt to decompose, not a verdict on its own,
and that the direction of the raw drift across rungs is the real diagnostic.

### 4.3 Starting point (measured, the four stage-1 winners at 4 chains × 75)

| metric | medium_play | large_play | medium_diverse | large_diverse |
|---|---|---|---|---|
| `cvar_ess_median` | 52.9 | 210 | 272 | 57.4 |
| `cvar_mcse_rel_median` | 0.220 | 0.195 | 0.143 | 0.319 |
| `cvar_rhat_median` | 1.113 | 1.013 | 1.013 | 1.077 |
| `cvar_rhat_pct_over_1.01` | 99.9% | 59.0% | 61.4% | 99.4% |
| `q05_ess_median` | 33.5 | 184 | 245 | 42.1 |
| `folded_rhat_95th_pct` | 1.541 | 1.082 | 1.086 | 1.312 |

**Provenance of this row.** These are the winners' own stage-1 sweep trials,
which already ran at 4 chains × 75 draws — the stage-3 `c4` rung is not a
separate experiment for three of the four variants, it is the configuration the
sweep selected, read back. Only medium_play also has a dedicated `c4` run
(`exp/stage3_medium_play_c4_0`, 2026-08-16), made because its sweep trial's saved
chains had been overwritten and `--worst-k` needs them (§10.3). That run
reproduced its trial to six significant figures, so the two sources are
interchangeable; if a later `c4` number appears with a different provenance,
that is why, and it does not indicate a discrepancy.

**This row is aggregate statistics only — no winner's chains survive, for any
variant.** The overwrite is structural, not bad luck: the sweep yamls do not
override `OUT_DIR`, so every trial in a sweep writes to the same deterministic
`{OUT_DIR}_{seed}` path and each one clobbers its predecessor's
`sampling_f/chain_*`. The winners were trials 27 / 21 / 27 / 25 of 40 / 29 / 42 /
40, so in all four sweeps later trials overwrote them. Everything logged to
wandb survives — which is the whole of the table above — but anything needing
saved chains does not: `--worst-k`, `--draw-ladder`, `--ce-ladder`. Those need a
fresh run at some rung, and the cheapest way to get them is to read them off the
`c8` run rather than re-run `c4` a second time for the sake of the diagnostic.

**medium_play and large_diverse are the weak pair** — roughly 4–5× less CVaR ESS
and near-universal R-hat exceedance. large_play and medium_diverse are already in
good shape. Expect the work to be concentrated on the first two, and do not
assume a single chain count suits all four.

Relative MCSE is already comfortably below 1 everywhere (0.14–0.32), so the
binding constraint is R-hat, not Monte-Carlo noise — which matters, because of
§4.5.

**Architecture, not schedule, is what tracks mixing — but do not over-read
which part of the architecture.** Across all 140 round-2 trials carrying tail
diagnostics, the swept *schedule* parameters are uncorrelated with every mixing
metric: `sghmc_lr_max` gives ρ = −0.03 / −0.01 / −0.02 and `fraction_cool`
0.08 / 0.12 / −0.13 against CVaR R-hat, folded R-hat and CVaR ESS. The two
levers cyclical SGMCMC is built around do nothing measurable for mixing here.
`depth` is the only variable clearing a Bonferroni bar (ρ = **+0.456** with CVaR
ESS, **−0.354** with CVaR R-hat); `width` is weaker (−0.256 with folded R-hat).

Among the four winners, however, `depth` does **not** explain the spread —
medium_diverse is `depth: 2` and mixes best. What lines up there is `width`:
both `width: 64` winners (medium_play, large_diverse) are the two poor mixers,
and both wide winners (512, 1024) are the two good ones. With n = 4 that is
anecdote, and it conflicts with the population correlation, so **treat the
per-variant cause as unidentified**. What is solid: mixing quality varies ~5×
across winners (CVaR ESS 53–272), it is an architecture effect rather than a
schedule one, and nothing in stage 3 can change it.

Note also §4.5: with `chain_init_jitter: 0.0` these R-hats are optimistic, and
the schedule-parameter null above may itself be an artifact — chains sharing a
start cannot express hot-phase exploration as between-chain diversity, which is
exactly what R-hat measures.

### 4.3.1 medium_play ladder — measured 2026-08-17, and what it shows

| metric | `c4` (300 draws) | `c8` (600 draws) | ratio | §4.6 ideal |
|---|---|---|---|---|
| `ess_bulk` median | 6.73 | 14.57 | **2.16×** | 2.00× |
| `rhat_bulk` median | 1.7208 | 1.5972 | fell | — |
| `cvar_ess_median` | 52.9 | 59.2 | **1.12×** | 2.00× |
| `q05_ess_median` | 33.5 | 43.2 | 1.29× | 2.00× |
| `cvar_mcse_rel_median` | 0.2204 | 0.2706 | **1.23×** | 0.71× |
| `cvar_rhat_median` | 1.1134 | 1.1497 | rise expected (§4.5) | — |
| `cvar_rhat_max` | 1.2686 | 2.1479 | moved — not a ceiling (§4.5) | — |
| `folded_rhat_95th` | 1.5405 | 1.3895 | fell | — |
| unresolved points | 56 (0.88%) | 144 (2.25%) | **2.57×** | — |

The §4.2 gate passes at both rungs (`c8`: loc_z 1.1385, scale_z 1.3564, clamp
0.0000), so the tail numbers are readable. But the margin to 2.0 narrowed by
roughly a third from `c4` (0.8256 / 1.0298), so re-read the gate at `c16` rather
than assuming it holds.

> **Superseded by §4.3.2.** Both PASSes here are low-power false negatives, and
> the narrowing margin is mostly the √C denominator effect of §4.2.1, not the
> sampler degrading. The `c16` rung showed the raw drift never shrinks with
> draws: it is real at `c4` and `c8` too. The tail numbers in this table are
> **not** readable in the sense claimed above. The nesting result and the
> bulk-vs-tail finding below are unaffected — both rest on measured quantities,
> not on the gate.

**The `c8` run's first four chains reproduce `c4` exactly.** `--num-chains 4` on
`exp/stage3_medium_play_c8_0` gives output byte-identical to
`exp/stage3_medium_play_c4_0` — every digit, bulk through unresolved count.
Chain initialisation and the per-chain RNG stream are deterministic in (seed,
chain index), so **the rungs are nested**: `c8` is `c4` plus four new chains.
Two consequences. §4.3's "reproduced to six significant figures" is exact. And
the `c4` → `c8` difference carries no run-to-run confound at all — it is
attributable entirely to chains 5–8. Use `--num-chains` for this check on every
later rung; it costs no sampling.

**Bulk and tail move in opposite directions, and that is the finding.** Adding
chains improves the bulk slightly *better* than the ideal (ESS 2.16×, R-hat
falling), while the tail barely moves and its relative MCSE goes the wrong way.
Since `mcse = sd(u)/√ESS` on the Rockafellar–Uryasev integrand
`u = min(f − VaR, 0)/α`, and ESS rose, `sd(u)/pred_sd` must have grown ~30%:
chains 5–8 agree with chains 1–4 about the bulk and **disagree about how deep
the lower tail goes**. The worst points corroborate it — `pred_sd` up to 20.0
and CVaR down to −61.6 at `c8`, against §4.5's `c4` record of `pred_sd` 5–15 and
O(10) magnitudes — and they sit in the *same* maze regions `c4` flagged
(x ≈ 19–21, y ≈ 4–5; y ≈ 20.7). More of them, not elsewhere.

**The draw ladder shows the same thing with the chains held fixed**, so it is
not an artifact of adding chains. Within the `c8` run, truncated to the first N
draws per chain:

| draws/chain | total | `cvar_ess_median` | `cvar_mcse_rel_median` | unresolved |
|---|---|---|---|---|
| 25 | 200 | 44.6 | 0.251 | 2.59% |
| 50 | 400 | 54.4 | 0.252 | 1.30% |
| 75 | 600 | 59.2 | 0.271 | 2.25% |

ESS gains 1.22× then 1.09× per step, and relMCSE *rises* over the last step with
the chains identical. **More sampling of either kind keeps finding deeper
excursions.** Read §4.6's stop rule with that in mind — it does not apply here.

### 4.3.2 The `c16` rung — measured 2026-08-17. Stage 3's axis is the wrong one.

| metric | `c4` (300) | `c8` (600) | `c16` (1200) | §4.6 ideal per step |
|---|---|---|---|---|
| `ess_bulk` median | 6.73 | 14.57 | 26.83 | 2.00× |
| `rhat_bulk` median | 1.7208 | 1.5972 | 1.7037 | → 1 |
| `cvar_ess_median` | 52.9 | 59.2 | 89.4 | 2.00× |
| `cvar_mcse_rel_median` | 0.2204 | 0.2706 | 0.2234 | 0.71× |
| `cvar_rhat_median` | 1.1134 | 1.1497 | 1.1802 | rise expected (§4.5) |
| `folded_rhat_95th` | 1.5405 | 1.3895 | 1.3676 | — |
| unresolved points | 56 (0.88%) | 144 (2.25%) | 42 (0.66%) | — |
| `loc_z` / `scale_z` | 0.826 / 1.030 | 1.139 / 1.356 | **2.515** / 1.996 | ≤ 2.0 |
| `loc_sd` / `scaleRatio` | 0.4319 / 1.4731 | 0.4222 / 1.4361 | 0.6460 / 1.4664 | — |

All three rungs differ in `num_chains` **only** — 75 draws, 20 000 burn-in
steps, seed 0, `chains_per_gpu: 4`, same host — so the movement is attributable
to chain count alone.

**The rungs are nested exactly, again.** `--num-chains 8` on the `c16` run
reproduces the standalone `c8` output to within one ULP on a single value
(`cvar_ess_max` 434.5072 vs 434.5073, float summation order); every other digit,
including the full worst-20 listing, is identical. So `c16` is `c8` plus eight
new chains, and everything below is attributable to chains 9–16.

**The gate failure at `c16` is mostly, but not entirely, the §4.2.1 artifact.**
Decomposing each step (√2 expected from power alone):

| step | `loc_z` | = power × real | raw `loc_sd` |
|---|---|---|---|
| `c4`→`c8` | 1.379× | 1.414 × **0.975** | **0.978×** |
| `c8`→`c16` | 2.209× | 1.414 × **1.562** | **1.530×** |

The whole `c4`→`c8` rise was denominator shrinkage — nothing degraded. Scale is
the same story at both steps (`scaleRatio` 0.975×, 1.021×). Location at
`c8`→`c16` is different: a real 1.53× rise that the raw metric confirms
independently.

**And that exposed the actual problem.** `loc_sd` must fall as 1/√(total draws)
under stationarity (§4.2.1). It does not:

| rung | draws | `loc_sd` | stationarity requires | obs/req |
|---|---|---|---|---|
| `c4` | 300 | 0.4319 | 0.4319 | 1.00 |
| `c8` | 600 | 0.4222 | 0.3054 | 1.38 |
| `c16` | 1200 | 0.6460 | 0.2160 | **2.99** |

Quadrupling the draws should have halved that shift. It grew by half instead.
**There is a real location drift of ~0.4–0.65 sd units that does not shrink with
sampling, and it is present at every rung** — `c4` and `c8` passed the gate only
because 300 and 600 draws lack the power to resolve it.

**Why more chains cannot fix it.** `function_space_drift` splits each chain's
own draws in half and pools; the drift is *within-chain*. Adding chains does not
lengthen any chain, so it cannot reduce a per-chain drift — it only measures it
with more precision. §4.1 raises `num_chains` with `num_samples` pinned at 75,
so **stage 3 as specified ladders an axis orthogonal to the binding
constraint.** The flat `cvar_mcse_rel_median` across 4× the compute (0.220 →
0.271 → 0.223, against an ideal 0.110) is the same fact seen from the tail: you
cannot drive down the Monte-Carlo error of an estimand that is still moving.

Note also `rhat_bulk` median never converges (1.72 → 1.60 → 1.70) and
`cvar_rhat_median` rises monotonically. §4.5 attributes rising R-hat to
detection power, which is correct as far as it goes — but it is the same power
argument that made the gate look fine at `c4`, and here it coexists with a
target that genuinely is not fixed.

**This is not specific to medium_play.** All four stage-1 winners carry
substantial raw drift at `c4` while all four pass the z-gate:

| variant | `loc_z` (≤2.0) | raw `loc_sd` |
|---|---|---|
| medium_play | 0.8256 | 0.4319 |
| large_play | 0.9168 | 0.2108 |
| medium_diverse | 1.3408 | 0.3902 |
| large_diverse | 0.7951 | 0.3778 |

Only medium_play has been laddered far enough to expose it. Do not read the
other three's §3.6.3 acceptance as established stationarity — it is the same
low-power PASS.

**What follows.** Do not launch `c32`: it would fail the gate harder on power
while saying nothing new about a within-chain problem, and it costs a full run.
The next experiment has to move `chain_init_jitter` (§4.1 — the sweeps always
deferred it to this stage, and it is still at the 0.0 that gives every chain an
identical start), `num_burn_in_steps` and/or `num_samples`, or the cyclical
step-size schedule that `function_space_drift` also tests (§3.6.2) — a drift
surviving 20 000 burn-in steps implicates the schedule as much as the budget.

> **§4.3.3 corrects two things below.** The `c16` unresolved-count improvement
> in the table above is a drift artifact, not progress; and the initialisation
> hypothesis at the end of this section is refuted — with `chain_init_jitter =
> 0` there is no per-chain initialisation to equilibrate.

The cheapest discriminating measurement needs no new sampling: compare the two
halves of the existing `c16` run against each other. `--chain-range` selects
a non-prefix slice (0-based, END exclusive, matching the on-disk `chain_N`
names, so the chains this section calls 9–16 in prose are `8:16`):

```
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_play_c16_0 --chain-range 0:8 \
    2>&1 | tee exp/stage3_medium_play_c16_0_lower8_diag_tail.txt
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_play_c16_0 --chain-range 8:16 \
    2>&1 | tee exp/stage3_medium_play_c16_0_upper8_diag_tail.txt
```

`0:8` must reproduce the standalone `c8` capture exactly — it is the same
selection `--num-chains 8` makes, so it doubles as a check that the range
plumbing is sound before the `8:16` number is trusted. Note that neither half
answers this on the z-scores: both are 8-chain selections, so they are equal in
power and directly comparable, but §4.2.1 still applies to reading them against
the 16-chain numbers. Compare the halves on raw `loc_sd`.

### 4.3.3 The half-split — measured 2026-08-18. Two corrections to §4.3.2.

Both halves are 8 chains × 75 draws from the same run, so they are equal in
power and directly comparable (§4.2.1).

| | lower `0:8` | upper `8:16` | full `c16` |
|---|---|---|---|
| `loc_z` median | 1.1385 | **2.5848** | 2.5152 |
| `loc_sd` median | **0.4222** | **0.8877** (2.10×) | 0.6460 |
| `scale_z` / ratio | 1.3564 / 1.4361 | 1.4599 / 1.4897 | 1.9962 / 1.4664 |
| §4.2 verdict | PASS | **FAIL** | FAIL |
| `rhat_bulk` median | 1.5972 | 1.8072 | 1.7037 |
| `cvar_ess_median` | 59.2 | **66.2** | 89.4 |
| `cvar_mcse_rel_median` | 0.2706 | **0.1810** | 0.2234 |
| unresolved | 144 (2.25%) | **22 (0.34%)** | 42 (0.66%) |

`0:8` reproduced the standalone `c8` capture exactly, so the range plumbing is
sound. The upper half drifts **2.10×** more on raw effect size. Two things
follow, and both correct §4.3.2.

**Correction 1 — the initialisation hypothesis is refuted, not confirmed.**
§4.3.2 proposed that chains 9–16 drift more because initialisation had not
equilibrated inside the burn-in. There is no per-chain initialisation to fail.
**`chain_init_jitter = 0`** for every run in this project (the sweeps set it
explicitly, the production configs omit it and take the same code default), and
at `f_pref_net.py:136` the jitter block is skipped entirely at zero — the
comment there reads `0.0 -> identical shared start`. All 16 chains begin at the
*same* warm-up point and differ only in `set_seed(seed + chain_idx)`, which
drives the SGHMC noise stream and the minibatch order. The chains are therefore
**exchangeable by construction**, and no mechanism makes "later" ones worse.
What is left for the 2.10× is chance across 8-vs-8 exchangeable chains, or GPU
placement — `device_idx = rank // chains_per_gpu` puts 0–7 on GPUs 0–1 and 8–15
on GPUs 2–3. The half-split cannot separate those; §4.3.4 can.

**Correction 2 — `c16`'s tail improvement is a drift artifact.** §4.3.2 records
unresolved falling 2.25% → 0.66% at `c16` without comment. The half that drifts
*more* has the **better** tail statistics: higher CVaR ESS, lower relative
MCSE, and 6.5× fewer unresolved points. Drift inflates within-chain variance and
widens the predictive spread (`sd2/sd1 ≈ 1.49`), which populates the lower 5%
and makes the tail look better resolved. So `c16`'s headline improvement is
contributed by its worst-behaved chains. This is §7.1's round-1 pathology
recurring inside the tail diagnostics that were built to replace R-hat — the
same lesson, one level further in. Note `rhat_bulk` moves the other way
(1.5972 → 1.8072), which is why the two must be read together.

### 4.3.4 Per-chain drift — what to measure instead

The pooled gate cannot distinguish one drift shared by every chain from two
chains drifting badly among fourteen that are fine, and those call for opposite
fixes. `--per-chain-drift` runs the §4.2 gate on each chain separately and adds
the statistic that separates them. With `delta[c,p] = E2[f] - E1[f]` for chain
`c` at point `p` in pooled-sd units, compare per point

    |mean_c delta[c,p]|   do the chains move TOGETHER?
    mean_c |delta[c,p]|   how far does a typical chain move?

Their ratio is ~1 when every chain follows one common trajectory and ~1/√C when
the chains wander independently. **A common shift is the signature of a shared
start that has not equilibrated**: the transient is identical in every chain, so
it does not average out over chains, and adding chains cannot reduce it — which
is exactly the non-shrinking `loc_sd` of §4.3.2. Independent wandering says the
opposite, that pooling more chains does help. A third case — a few chains
dominating — is flagged separately when the per-chain `loc_sd` max is ≥3× the
median, because the pooled gate is then reporting those chains rather than the
sampler.

Verified on synthetic chains: a common ramp gives alignment 0.9987 with 16/16
chains shifting one way, independent random-sign ramps give 0.0904, and two
planted outliers among fourteen stationary chains are named individually at
6.20× the median.

This is the measurement that decides what the next *sampling* run should change,
so run it before choosing between burn-in, `chain_init_jitter`, and the
schedule:

```
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_play_c16_0 --per-chain-drift \
    2>&1 | tee exp/stage3_medium_play_c16_0_perchain_diag_tail.txt
```

If it reports COMMON drift, the shared start is implicated and
`chain_init_jitter` is the first thing to move — see §4.1, which the sweep
configs always intended stage 3 to set. If it names a few outlier chains,
check what they share (GPU index above all) before touching any schedule
setting.

### 4.3.5 Per-chain result — measured 2026-08-18. The drift is common and ongoing.

`--per-chain-drift` on all 16 `c16` chains:

| | value | reference |
|---|---|---|
| ALIGNMENT | **0.7564** | 1.00 = one common drift; 0.25 = 16 independent |
| signed shift | **14/16 positive** | sign test p = 0.0042 |
| per-chain `loc_sd` | median 1.2098, max/median **1.33×** | ≥3× would mean outlier chains |
| mean signed shift | **+0.647** | pooled `loc_sd` = 0.6460 |

**The drift is common to essentially every chain.** The mean of the per-chain
signed shifts lands on the pooled `loc_sd`, so the pooled number *is* the common
component rather than an aggregation artifact. A shift shared by all chains
cannot average out over chains, which is §4.3.2's non-shrinking `loc_sd`
demonstrated directly rather than inferred. `max/median` 1.33× rules out the
outlier-chain case.

**This also explains §4.3.3's 2.10×, and it was not what it looked like.**
Per-chain `loc_sd` by half is lower 1.084 vs upper 1.232 — only **1.14×**. The
*signed* shifts split lower +0.412 vs upper +0.756 — **1.83×**, essentially the
whole pooled gap. The cause is chains **3 (−0.4618)** and **4 (−0.6390)**, the
only two counter-drifting chains in the run, both of which happen to sit in the
lower half where they partially cancel. The lower half looked better because it
contained the cancellation. This also closes the GPU question left open in
§4.3.3: with `chains_per_gpu: 4`, chain 3 is on GPU0 and chain 4 on GPU1, so the
counter-drifting pair is not a placement group. Four chains per GPU is thin
evidence, but nothing in the table suggests placement matters.

**Correction to §10.2: `chain_init_jitter` is the wrong fix for *this*.** A
common directional drift is not a dispersion problem. Jitter spreads chains
*around* the shared start; if the start is off in a common direction, every
chain still relaxes the same way and the common component survives. Jitter
remains worth setting for **R-hat honesty** — shared starts understate
between-chain variance, per `f_pref_net.py:130` — but that is a separate defect
and it will not move `loc_sd`. §4.1 still holds that stage 3 owns the parameter.

> **The schedule hypothesis below was REFUTED by direct test on 2026-08-19 —
> see §4.3.6.** Turning the schedule off made the drift *worse* on every axis.
> The per-chain result above stands; only the attribution paragraphs that
> follow are wrong. They are kept because the reasoning was sound and the
> refutation is the useful part of the record.

**The schedule is the leading suspect, and the arithmetic favours it.**
From the `c16` config: `sghmc_lr` 2.4897e-4, `sghmc_lr_max` 8.7152e-4
(**3.50×**), `cycle_length` 2750, `samples_per_cycle` 1, `fraction_cool` 0.337,
`n_discarded` 5. So 80 cycles × 2750 = **220 000 sampling steps against 20 000
of burn-in — the sampling phase is 11× the burn-in.** Three consequences:

- **A start transient is a poor explanation.** It would have to survive burn-in
  *and* still produce a 0.65 sd shift between step ~100 000 and ~206 000. More
  burn-in is unlikely to be the fix.
- **Burn-in runs at `lr_min` and cycling begins only after it**
  (`f_pref_net.py:724–745`; `burn_in_lr` is None in all these runs). The chain
  equilibrates at one step size and is then driven by a schedule it never saw
  during burn-in. Any drift the schedule causes therefore starts *at the first
  sampling step*, which no amount of burn-in reaches.
- **The spread grows as well as the location** (`scale_ratio` 1.4664). A
  widening chain is being heated, which fits a per-cycle energy injection better
  than a decaying transient. Momentum is resampled at every cycle start at the
  `lr_max` scale (`f_pref_net.py:746–772`), though with `mdecay` 0.1946 the
  momentum relaxation time is ~5 steps, so momentum equilibrates long before the
  sample is taken at the coldest step — the plausible mechanism is the
  *position* distribution not returning fully during the ~927-step cool phase,
  netting a displacement per cycle.

This is a deviation from Wu et al. (2025), which uses a fixed step size; the
cyclical schedule is a bolt-on carrying no stationarity guarantee for these
dynamics, and §3.6.2 records that `function_space_drift` was written partly to
test it. **Treat "the cyclical schedule causes the drift" as the leading
hypothesis, not as established.**

**`--drift-blocks K` is the shape test, and at 75 draws it is underpowered.**
It splits the draws into K blocks and reports the block-to-block shift against
a noise floor of `0.6745*sqrt(2/ess_block)` — a stationary chain sits *at* the
floor, so "flat" only means an ongoing drive when the shifts clear it. With
`ess_bulk` ≈ 26.8 over 1200 draws (~2% efficiency), the per-block floor at K=5
is ≈ 0.41 while a linear drift totalling 0.646 gives only ≈ 0.16 per step. It
will almost certainly report SHAPE NOT RESOLVED. Run it anyway — the `sd/first`
column is far better determined than a difference of means and will show
whether the widening is progressive — but do not expect it to settle the
question:

```
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_play_c16_0 --drift-blocks 5 \
    2>&1 | tee exp/stage3_medium_play_c16_0_blocks_diag_tail.txt
```

**The decisive experiment is one run: turn the schedule off at matched
compute.** Constant step size at the value samples are currently taken at, same
total gradient steps, same burn-in, everything else identical:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 8 --chains_per_gpu 4 \
    --use_cyclical_lr False --keep_every 2750 \
    --OUT_DIR ./exp/stage3_medium_play_nocyc > ../exp/stage3_medium_play_nocyc.log 2>&1 &
```

`keep_every 2750` matches `cycle_length`, so 75 draws cost the same 206 250
steps and the comparison is compute-matched rather than confounded by budget.
In non-cyclical mode the LR stays at `sghmc_lr` throughout (`burn_in_lr` is
None), which is exactly the `lr_min` the cyclical run samples at. Eight chains
is enough: `loc_sd` is a raw effect size and the comparison is against the `c8`
/ lower-half numbers (`loc_sd` 0.4222, alignment to be read from
`--per-chain-drift`).

Read the result on **raw `loc_sd` and ALIGNMENT**, not on the tail statistics —
§4.3.3 showed those improve as drift worsens. If `loc_sd` collapses, the
schedule is the cause and stage 3's remit changes from a budget to a schedule
decision. If it survives, the schedule is exonerated and the next suspect is
weight-space diffusion along directions that are only approximately
f-preserving (§3.6.2) — which would show up as `loc_sd` tracking ‖w‖ growth.

### 4.3.6 The schedule is exonerated — measured 2026-08-19

The compute-matched `use_cyclical_lr False` run (`stage3_medium_play_nocyc_0`,
wandb `60a6b467`) against the `c8` cyclical run. Both 8 chains, so the z-scores
are comparable as well as the raw effect sizes; identical `sghmc_lr`, `mdecay`,
`num_samples` 75, `num_burn_in_steps` 20 000, `n_discarded` 5, seed 0,
`batch_size` 64, `n_meas` 35, `chain_init_jitter` 0. Runtimes 8 879 s vs 8 304 s
confirm the compute match.

| metric | `c8` cyclical | `nocyc` | |
|---|---|---|---|
| raw `loc_sd` | 0.4222 | **0.5443** | 1.29× **worse** |
| `loc_z` | 1.1385 | 1.3217 | worse |
| raw `scale_ratio` | 1.4361 | **2.1326** | 1.49× **worse** |
| `scale_z` | 1.3564 | **2.5834** | now **FAILS** the 2.0 gate |
| predictive CE | 0.2029 | **0.3580** | much worse |
| predictive accuracy | 0.9172 | 0.8539 | −6.3 points |

**The cyclical schedule does not cause the drift.** Removing it made every
stationarity metric worse and pushed `scale_z` past a gate it had passed. The
drift is present with a constant step size, so it is intrinsic to the sampler or
the model. §4.3.5's hypothesis is refuted.

**The schedule is also doing real work, which closes a gap in §3.6.2.** That
deviation was recorded as deliberate but *unsupported* — justified on the
argument that standard SGHMC gets trapped in a single basin, with no empirical
evidence. This run is that evidence: without cycling, CE degrades 0.2029 →
0.3580 and accuracy drops 6.3 points at matched compute. The most-suspected
deviation turns out to be the one carrying measured support.

**The momentum-resampling confound is quantitatively void.** Momentum
resampling fires only inside the cyclical branch (`f_pref_net.py:772`), so the
`nocyc` run dropped both cycling *and* resampling, and this was initially
flagged as a confound. The arithmetic dismisses it. The update is
`m <- m*(1 - mdecay) + force + noise` (`adaptive_sghmc.py:156–159`), so with
`mdecay` 0.19462 the momentum autocorrelation time is **4.62 steps** and the
fraction retained after one 2750-step cycle is **3e-259** — the memory is gone
within ~100 steps. A resample at cycle start cannot influence a sample collected
2750 steps later, in either the location or the spread. Resampling from the
*stationary* law is measure-preserving anyway; it only decorrelates. **So the
run does isolate the LR schedule after all**, and the confound needs no
follow-up. It would only matter at cycle lengths under ~50 steps.

**Therefore the spread difference is the schedule's doing, not momentum's**, and
its direction is worth noting. The cyclical run's *average* LR over a cycle is
`(lr_min + lr_max)/2` = 5.6e-4, about 2.25× the constant 2.49e-4, so naively it
should diffuse more — yet its spread grows far less (1.44× vs 2.13×). The
per-cycle anneal appears to re-concentrate the chain, with each sample taken at
the coldest step after a quench. Two consequences, both to carry forward:

- It supports **weight-space diffusion** (§3.6.2) as the underlying driver: free
  diffusion accumulates monotonically at constant LR, and the quench partially
  counteracts it.
- It raises a **new question about the cyclical run itself** — sampling only at
  the cold point may *under-disperse* relative to the true posterior, which
  would flatter every diagnostic. Note `scale_ratio` measures spread *growth*
  between halves, not absolute width, so nothing here measures that bias. It is
  untested and belongs on the disclosure list.

**What remains, and what discriminates.** Both surviving hypotheses must explain
a drift common to 14/16 chains (alignment 0.7564) — random slow mixing would
give ~0.25, so the cause is systematic:

1. **Common relaxation from the shared start.** `chain_init_jitter` is 0, so
   every chain begins at the *identical* warm-up point. §4.3.5 demoted this too
   quickly; with the schedule exonerated it is the remaining mechanism that
   naturally produces a common *direction*.
2. **Weight-space diffusion only approximately f-preserving** (§3.6.2). §7.1
   records round 1's weight norm growing 4.09× in *all eight* chains, so
   systematic norm growth is common across chains too and survives the alignment
   test.

`chain_init_jitter` now discriminates rather than merely fixing R-hat. It is a
per-tensor relative perturbation, `chain_init_jitter * std(w)` seeded per chain
(`f_pref_net.py:136–143`). **If the common drift comes from the shared start,
jittering the starts must drop ALIGNMENT.** If alignment stays near 0.75 with
jittered starts, hypothesis 1 is dead and the cause is diffusion.

> **Sizing correction (§4.3.7).** This was first run at **8 chains** to save
> compute, which made it uninformative: ALIGNMENT varies by 0.41 and pooled
> `loc_sd` by 2.10× between the two halves of a single jitter-0 run, against a
> jitter effect of 0.046. Run it at **16 chains**, the only size with a matched
> baseline (`c16`: ALIGNMENT 0.7564, `loc_sd` 0.6460). Command in §4.3.7.

Run it cyclical — `nocyc` is worse on every axis and is not the configuration
to build on.

### 4.3.7 The jitter test — measured 2026-08-19. Inconclusive, and why.

`chain_init_jitter 0.1`, 8 chains, cyclical, against `c8` (identical but
jitter 0):

| metric | `c8` (jitter 0) | `jit 0.1` | |
|---|---|---|---|
| raw `loc_sd` | 0.4222 | 0.3493 | −17% |
| `loc_z` | 1.1385 | 0.9599 | better |
| raw `scale_ratio` | 1.4361 | **1.5750** | +10% worse |
| `scale_z` | 1.3564 | **1.7807** | +31% worse |
| ALIGNMENT | 0.5414 | 0.4955 | −0.046 |
| unresolved points | 2.25% | 3.11% | worse |
| `rhat_bulk` median | 1.5972 | 1.4737 | better |
| `ess_bulk` median | 14.57 | 16.58 | better |
| per-chain `loc_sd` max/median | 1.33× | 2.07× | outlier chain 2 |

**The experiment cannot answer the question, because 8 chains cannot resolve
the statistic.** Measuring ALIGNMENT on both halves of the *single* `c16` run —
same schedule, same jitter 0, same 8 chains each, same draws — gives:

| chains | GPUs | ALIGNMENT | signed | pooled `loc_sd` | verdict |
|---|---|---|---|---|---|
| 0–7 | 0,1 | 0.5414 | 6/8, p = 0.2891 | 0.4222 | INDEPENDENT |
| 8–15 | 2,3 | **0.9516** | **8/8, p = 0.0078** | **0.8877** | COMMON, gate FAILS |

Within one run the C=8 ALIGNMENT spans **0.41** and pooled `loc_sd` spans
**2.10×**. The jitter effect being tested was 0.046 — **swamped roughly 9×**.
Both statistics are too variable at 8 chains for this comparison to mean
anything. The 17% `loc_sd` gain is likewise inside the run-internal range
(`jit` 0.3493 sits below both halves, which is weakly encouraging and nothing
more). **Treat jitter as untested, not as refuted.**

**§4.3.5's "common drift" claim stands.** The halves differ for the reason
§4.3.5 already identified: chains 3 and 4 are the run's only counter-drifting
chains and both sit in the lower half, where they cancel. Under chance
placement two counter-drifters land in the same half 47% of the time and in the
lower half specifically 23% of the time — unremarkable. The 16-chain
measurement (ALIGNMENT 0.7564, 14/16 positive, p = 0.0042) pools both halves
and is the reliable one; the two 8-chain readings straddle it. **Prefer the
16-chain figure and do not read half-splits as independent evidence** — that
error produced §4.3.3's since-corrected hypothesis and nearly produced a second
one here.

**GPU placement remains unimplicated but is not fully excluded.** All 16 chains
ran in one wave (`max_concurrent = num_gpus * chains_per_gpu` = 16), so the
lower half is GPUs 0,1 and the upper half GPUs 2,3 — the split is exactly along
a GPU boundary. Chance placement of two chains explains it without invoking
hardware, and there is no mechanism by which a GPU would bias drift
*direction*; but with 4 chains per GPU this cannot be settled from the data. If
the 16-chain jitter run below reproduces a clean lower/upper split, revisit it.

**Redo the test at 16 chains.** That is the only configuration with a matched
baseline (`c16`: ALIGNMENT 0.7564, `loc_sd` 0.6460), and 8 chains has now been
shown to be below the resolution of both statistics:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --chain_init_jitter 0.1 \
    --OUT_DIR ./exp/stage3_medium_play_jit16 > ../exp/stage3_medium_play_jit16.log 2>&1 &
```

**Methodological note for the write-up.** Choosing 8 chains to save compute
made the run worthless for its purpose; the 16-chain run it was meant to avoid
is now needed anyway, at a total cost higher than running it first. Where a
diagnostic's sampling variability is unknown, measure that variability before
sizing the experiment — here it was available for free from an existing run's
two halves.

### 4.4 Procedure

Run at **seed 0** (the selection lineage — §1; never touch seeds 1–10), from
`scripts_bnn/`, with a distinct `OUT_DIR` per candidate so runs do not clobber
one another (the deterministic `{OUT_DIR}_{seed}` path has destroyed evidence
twice — §10.3):

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=<gpus> nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_<variant>_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains <N> --chains_per_gpu <N/gpus> \
    --OUT_DIR ./exp/stage3_<variant>_c<N> > ../exp/stage3_<variant>_c<N>.log 2>&1 &
```

`--config_path` and `--OUT_DIR` are repo-root-relative and resolve *after* the
script's `os.chdir("..")`; the shell redirect is not, hence `../exp/`. Thread
caps are set inside the script (§10.7) — do not override them.

A sensible ladder is **4 → 8 → 16 chains**, stopping when §4.6 is satisfied.
`num_chains: 8` at `chains_per_gpu: 2` uses 4 GPUs; 16 at 4 uses 4 GPUs.

### 4.5 What to judge it on — and how R-hat will behave

Judge on the **distributional** statistics: `cvar_ess_median`,
`cvar_mcse_rel_median`, `cvar_rhat_median`, `cvar_rhat_pct_over_1.01`,
`q05_ess_median`, `folded_rhat_95th_pct`.

**The repeated `cvar_rhat_max` values are sparsity, not a ceiling** — an earlier
version of this section said otherwise and was wrong. Measured 2026-08-16: with
4 chains × 75 draws the *attainable* folded R-hat maximum (chains forced to
maximal separation) is **1.778**, well above the 1.2686 that medium_play and
large_diverse both report, so nothing is saturating.

The real cause is that the CVaR integrand is mostly zeros. Only `alpha · C · D`
draws fall below VaR — **15 of 300** at 4 × 75 — so R-hat is determined by how
that handful distributes across the chains, and can take only a few distinct
values. In medium_play's worst-20 points, just four distinct values appear and
1.269 occurs 15 times; the most plausible reading is that at those points
essentially all the tail mass comes from a *single chain*. That also explains
round 1's identical-to-16-digits values without invoking a ceiling: fewer draws
below VaR, fewer reachable configurations. Record the extremes; still do not
steer on them, but read a repeated value as tail mass concentrated in one chain
rather than as a censored statistic.

Do **not** use `param_*` diagnostics (they no longer exist — §3.6.2 explains
why weight-space statistics measure nothing for this sampler) or bulk
`pred_rhat`/`pred_ess` (they certify the median, not the tail).

**`--weight-trace` printed a false verdict, and has been removed entirely
(2026-08-18).** It declared `DRIFTING -- chains are not stationary; R-hat/ESS
improvements are an artifact` whenever RMS `‖w‖` grew more than 1.5× — a
`param_*`-era check that survived the §3.6.2 audit by being in a different file.
Under fSGHMC that threshold fires on *healthy* runs: `‖w‖² = ‖w₀‖² + c·t` is the
expected free diffusion along f-preserving flat directions, which at 75 draws
alone gives ~2.7× growth in `‖w‖`. medium_play `c8` duly tripped it (§3.6.2 now
records that measurement, which is the one useful thing the flag ever produced).

It was first re-scoped rather than deleted — the verdict dropped, the function
kept as a check that `‖w‖²` is linear in draw index and that the rate `c` is
common across chains, on the theory that this catches a broken step size or a
binding `max_param_step` clamp. **That re-scoping does not survive scrutiny
either, for a reason that applies to any weight-space check here:**

- The clamp is already measured *directly*. `param_clamp_sampling_pct` exists
  precisely because `max_param_step` is not measure-preserving when it binds,
  and §4.2 gates on it at ≤ 0.01%. Inferring the same condition from a
  norm-growth rate is strictly worse than reading the instrument built for it.
- There is no per-chain step size to be "broken". It is one config value shared
  by every chain, so the failure mode the rate-spread check was aimed at has no
  mechanism.
- Both thresholds were invented, not calibrated. `R² > 0.98` and
  `spread < 1.5×` had no null distribution behind them — and the cyclical step
  size (§3.6.2's first known deviation) makes the diffusion coefficient vary
  *within* a cycle, so exact linearity is not even the right prediction; it
  holds only cycle-averaged.

So the residual justification was two uncalibrated thresholds detecting a
condition that a gating metric already measures directly. The whole function and
its flag are gone. **The general rule stands: no statistic computed from `w`
belongs in this pipeline** (§3.6.2), and "but this one is only a mechanical
check" is exactly how the first one got in.

The same commit fixed the `--draw-ladder` banner, which still asserted the
`_max`/`_min` extremes "are censored at estimator ceilings" — the claim this
section corrected in `fd74b28`. medium_play's own ladder refutes it:
`cvar_rhat_max` moves 2.2188 → 2.0871 → 2.1479.

**Chains buy precision, not mixing.** ESS and MCSE improve roughly with total
draws; R-hat does not, because it measures between-chain *disagreement*.

**Expect R-hat to RISE as chains are added, and do not read that as a
regression.** More chains give more power to detect disagreement that is already
there, so a real difference shows up more strongly rather than averaging away.
Measured on simulated chains at 75 draws (2026-08-16):

| case | 4 chains | 16 chains |
|---|---|---|
| iid — true R-hat = 1 | 0.9985 | 0.9995 |
| chains offset 0.3 sd | 1.0106 | 1.0921 |
| chains offset 0.8 sd | 1.0466 | 1.2540 |

Under the null the statistic is essentially unbiased at 4 chains (median 0.9985,
9% over 1.01), so **medium_play's observed 1.1134 with 99.9% over 1.01 is
genuine disagreement, not small-sample noise** — it exceeds even the 0.8-sd
offset case. Adding chains there should push it *up*, toward the truth.

The correct reading of the ladder is therefore: **ESS and MCSE improve, R-hat
gets worse, and both are working as intended.** Judge the budget on
ESS/MCSE/unresolved-point count; treat R-hat as characterising the sampler, not
as something the chain count is supposed to fix.

- Mode separation is addressed by hot-phase mixing (`lr_max`, `fraction_cool`)
  and `chain_init_jitter` — none of which stage 3 may change. §4.3 records that
  neither swept schedule parameter correlates with any mixing metric anyway.
- **`chain_init_jitter` is 0.0, so every chain starts from the same burn-in
  point.** R-hat therefore *understates* disagreement, and adding chains from a
  shared start adds little independent information about multimodality. Treat
  the R-hat numbers as optimistic, and say so in the write-up.

**What the medium_play c4 diagnosis showed** (`--worst-k 20`, 2026-08-16): only
**56 of 6400 points (0.88%)** are unresolved, and they cluster spatially — three
at (2.6, 12.87), a group at x ∈ [15.7, 20.7] with y ≈ 20.5–21.0, another at
(20.5–20.7, 4.95). So the problem is **localised multimodality in a few maze
regions, not diffuse failure**, and reward magnitudes are O(10) with `pred_sd`
5–15 — nothing like round 1's O(10³). Run `--worst-k` on each variant before
concluding anything from the aggregate R-hat.

### 4.6 When to stop, and what to record

Stop when, for each variant, the median statistics have flattened between
successive chain counts and §4.2 still passes. Concretely: `cvar_ess_median` and
`q05_ess_median` scale roughly linearly with total draws while
`cvar_mcse_rel_median` falls as 1/√(draws) — once an extra doubling buys little,
the budget is enough.

**Flattening only means "enough" if `cvar_mcse_rel_median` is also falling.**
The rule above presumes a *fixed estimand* whose Monte-Carlo error shrinks with
draws; then ESS flattening says the error is as small as it needs to be. If ESS
flattens while relMCSE **rises**, that premise has failed: `sd(u)` is growing,
the sampler is still discovering tail mass, and the CVaR estimate itself is
still moving. Flattening then measures the sampler's inability to reach the tail
faster, not the sufficiency of the budget — and stopping there would freeze the
budget at the rung whose tail statistics look best *because it found the least
tail*. medium_play `c4` → `c8` is exactly this case (§4.3.1): read the two
columns together, never ESS alone.

Where that happens, §4.1 governs: the tail is not satisfiable by adding chains
at a fixed horizon, and that is a finding about the schedule to be reported, not
a licence to lengthen the chains. Record the rung anyway — the ladder is the
evidence for the finding.

**Before applying any of the above, check §4.2.1's stationarity test.** The
whole stop rule presumes a fixed estimand that more draws will resolve. If raw
`loc_sd` is not falling as 1/√draws, there is no fixed estimand to resolve and
the rule does not apply at any rung — not just the one that failed the z-gate.
That is the state medium_play is in (§4.3.2), and it is why the ladder was
halted at `c16` rather than continued to `c32`.

Record per variant: the chosen `num_chains` / `chains_per_gpu`, the full table
of §4.3 metrics at each candidate, the §4.2 numbers at the chosen budget, and
what the `_max` extremes did (they are sparse, not censored — §4.5). Then transcribe `num_chains` and
`chains_per_gpu` into the production configs (§10.3) and re-run the field-by-field
verification.

`stage3_ladder.py <variant>` (repo root, runs locally against the wandb API)
assembles the ladder: it finds the rungs by their `stage3_<variant>_c<N>`
`OUT_DIR` marker, prints the §4.2 gate first and marks any rung that fails it,
then the §4.3 table and the rung-to-rung ratios against the §4.6 ideals
(ESS ~linear in draws, relMCSE ~1/√draws), labelling a rising R-hat as expected
rather than flagging it. The gate block also prints the raw `loc_sd` /
`scaleRatio` beside the z-scores, splits each rung-to-rung `z` change into its
power and real components, and runs §4.2.1's 1/√draws test on `loc_sd`. When
that test fails it says so before the tail table, marks every row compromised
rather than only the failing rung, and replaces §4.6's stop rule with the
reason it does not apply. It also checks that stage 3 moved `num_chains` *only* —
`num_samples` still 75, seed still 0, `burn_in_lr` still absent. With no
dedicated `c4` run it falls back to the recorded stage-1 winner for that rung
(§4.3). It cannot report the unresolved-point count or `--worst-k`, which are
not logged to wandb; take those from the diagnostic below.

`scripts_bnn/diagnose_sampling_tail.py --run-dir <OUT_DIR>` recomputes every tail
statistic from saved chains without re-sampling; `--worst-k` lists the
least-converged points with their torso (x, y) so you can judge whether the
unresolved ones are genuinely multimodal states. `--ce-ladder` and
`--draw-ladder` answer "would fewer draws have done?" from a completed run.
`--num-chains N` reads only the first N chains, which — because chains are
deterministic in (seed, index) — reproduces the lower rung exactly and isolates
what the added chains changed (§4.3.1). `--chain-range START:END` selects any
slice, so the chains a rung *added* can be read on their own (§4.3.3).
`--per-chain-drift` runs the gate on each chain separately and reports how
aligned the chains' shifts are, separating one common drift from a few outlier
chains (§4.3.4). Every capture now leads with the §4.2 gate recomputed from the
saved chains, which is the only way to get it for a chain subset — wandb has
`fn_drift_*` for the run as a whole and nothing finer. `--chain-range START:END` generalises
that to any slice, so the chains a rung *added* can be measured on their own
rather than only in aggregate (§4.3.2). It is 0-based with END exclusive, like
a Python slice and like the `chain_N` directory names; `--num-chains N` is
exactly `0:N`, and the two are mutually exclusive. Every run prints the
selection as directory names and flags a subset explicitly, because the
1-indexed prose of §4.3.2 ("chains 9–16") and the 0-indexed directories
(`chain_8`..`chain_15`) differ by one. There is deliberately **no weight-space
option**: `--weight-trace` was removed on 2026-08-18 and should not be
reintroduced in any form (§4.5).
Pipe it through `tee exp/<run>_diag_tail.txt` — its unresolved-point count and
`--worst-k` listing are the parts of §4.6's record that never reach wandb.

---

## 5. Stage 4 (all families): output normalization

The final hyperparameter is the **normalization function applied to the reward
model's output** before it is consumed by offline RL.

**Procedure:** for each (model family × antmaze variant) winner from the earlier
stages, grid-search over **8 normalization functions, indexed 0–7**. For each
candidate, train an IQL policy on the corresponding antmaze variant using that
normalized reward model, and select the index that **maximises the mean policy
score**.

**Seed: 0** — the same selection lineage as every other stage. All selection,
end to end, happens at seed 0; all evaluation happens at seeds 1–10.

**Selection statistic, precisely.** One IQL run is **1,000,000 training steps**
with an evaluation every **5,000** steps, i.e. **200 evaluation points per run**.
One evaluation point is the **mean score over 100 episodes**. The selected
normalization index is the one maximising the **maximum, over those 200
evaluation points, of the mean-over-100-episodes score**.

**Selection and reporting use the identical statistic.** The reported
evaluation numbers are also the max-over-evaluation-points of the
mean-over-100-episodes score; the only difference between selection and
evaluation is the seed lineage (0 vs 1–10). There is no selection/reporting
mismatch to disclose.

> One thing to state plainly in the paper regardless: this statistic is a
> **max over 200 checkpoints**, which is optimistic relative to reporting the
> final checkpoint. It is a common offline-RL convention and it is applied
> identically to every method and every baseline here, so it does not favour
> any one of them — but it should be named rather than left implicit.

**Implementation.** Defined in `iqlpref/algorithms/offline/iql.py` (one level
above this repo) in `modify_reward(dataset, env_name, normalize_reward, ...)`,
selected by the `normalize_reward: int` config field. Index 0 is the identity —
the call site is guarded by `if config.normalize_reward:`, so 0 is falsy and no
transformation is applied.

`min_ret` / `max_ret` are the **minimum and maximum episode returns in the
dataset** as labelled by the reward model under test, and `trj_lens` is the
per-transition trajectory length (all from `return_reward_range`).
`max_episode_steps = 1000`. With `r` the per-step reward:

| idx | transformation | note |
|---|---|---|
| 0 | `r` (identity) | no normalization |
| 1 | `r − 1` | the −1 shift used on the task (oracle) reward |
| 2 | `r / (max_ret − min_ret) · 1000` | scale only |
| 3 | idx 2, then `− 1` | |
| 4 | `(r − min_ret) / (max_ret − min_ret) · 1000` | **as described in the PT paper** |
| 5 | idx 4, then `− 1` | |
| 6 | `(r − min_ret/trj_lens) / (max_ret − min_ret) · 1000` | per-step share of `min_ret` |
| 7 | idx 6, then `− 1` | **as actually implemented in the PT codebase** |

Two properties worth being aware of:

- **The grid is data- and model-dependent.** Indices 2–7 derive their constants
  from `min_ret`/`max_ret` of the reward model's own labels over the dataset, so
  the same index means a different transformation for each model. This is
  exactly why the index must be re-selected per (family × variant) rather than
  fixed once.
- **Indices 4 and 7 encode a discrepancy in the literature.** The PT paper
  describes index 4, but its released code implements index 7. Including both in
  the grid means the comparison does not depend on which of the two you consider
  canonical — worth one sentence in the paper.

**Why this stage is selected on return rather than on validation loss:** unlike
stages 1–3, this is not a property of the preference model in isolation. A
monotone rescaling of the reward leaves the preference likelihood unchanged
(Bradley–Terry is invariant to it) but materially changes offline-RL behaviour,
because IQL's value targets, advantage weighting, and expectile regression all
depend on the reward's *scale and spread*. Validation CE therefore cannot
distinguish the candidates, and the downstream objective is the only informative
signal.

The operative values (`n_episodes: 100`, `eval_freq: 5000`,
`max_timesteps: 1000000`, `normalize_reward: <idx>`) come from the IQL run
config, not from the dataclass defaults in `iql.py` — see the note on reading
configuration in §3.4.

---

## 6. Results

Entity `champlin-university-of-arizona`. Verified 2026-08-08.

**Read the headings carefully.** MR and PT are unaffected by the round-2
redesign and stand as reported. Everything labelled ROUND 1 below is the record
of a **discarded** design (§3.7) — it is kept because the discard itself has to
be reportable, not because those hyperparameters are in use. No BNN result in
this document is currently valid.

### Stage 1 — BNN merged sweep (metric `val_predictive_cross_entropy`) — complete, 4 of 4 fired

Round 2. Every sweep fired well inside the 130 cap, and every winner is the
lowest-CE **eligible** trial under §3.6.3 — which is not always the lowest-CE
trial.

| variant | sweep | winner | trial / trigger | predictive CE | accuracy | rejected |
|---|---|---|---|---|---|---|
| medium_play | `9gifb8sa` | `0t5bqw02` | 27 / 40 | 0.193983 | 0.9156 | 17 / 40 |
| large_play | `ojk7k4vb` | `oi7cqb9o` | 21 / 29 | 0.204685 | 0.8981 | 12 / 29 |
| medium_diverse | `bq7ygeqe` | `n1ztawsx` | 27 / 42 | 0.237685 | 0.8925 | 14 / 42 |
| large_diverse | `m5sp9bw9` | `yjezedlk` | 25 / 40 | 0.246561 | 0.8818 | 22 / 40 |

Architecture and prior strength:

| variant | `width` | `depth` | `n_meas` | `map_amp2` |
|---|---|---|---|---|
| medium_play | 6 (64) | 2 | 35 | 168939.82 |
| large_play | 9 (512) | 6 | 29 | 925894.92 |
| medium_diverse | 10 (1024) | 2 | 17 | 119680.64 |
| large_diverse | 6 (64) | 4 | 7 | 94945.17 |

Sampler schedule:

| variant | `sghmc_lr` | `sghmc_lr_max` | `cycle_length` | `mdecay` | `fraction_cool` |
|---|---|---|---|---|---|
| medium_play | 2.489686e-04 | 8.715231e-04 | 2750 | 0.194623 | 0.336929 |
| large_play | 1.420262e-04 | 9.123611e-04 | 500 | 0.031181 | 0.120402 |
| medium_diverse | 1.250191e-04 | 3.993862e-03 | 750 | 0.007162 | 0.126945 |
| large_diverse | 7.569419e-05 | 8.436263e-04 | 2750 | 0.376143 | 0.401683 |

Acceptance numbers (§3.6.3 thresholds: both z ≤ 2.0, clamp ≤ 0.01%):

| variant | `loc_z` | `scale_z` | `clamp %` | rule-vs-eligible gap | frontier |
|---|---|---|---|---|---|
| medium_play | 0.83 | 1.03 | 0 | **+0.0035 (1.8%)** | ⚠️ stale 13 |
| large_play | 0.92 | 1.63 | 0 | **+0.0041 (2.0%)** | ⚠️ stale 8 |
| medium_diverse | 1.34 | 1.51 | 1.89e-07 | — | clean (15) |
| large_diverse | 0.80 | 1.13 | 0 | — | clean (15) |

**The acceptance criteria bought stationarity for about 2% of predictive CE.**
In medium_play and large_play the lowest-CE trial failed on drift, so the winner
is the next eligible one. In the other two the lowest-CE trial was already
eligible and there was nothing to trade.

**Rejections are overwhelmingly *scale* drift and concentrate at the top of the
ranking.** 65 of 151 trials up to the triggers were rejected — **43%**, not a
long tail — almost all for `scale_z` above 2 (values to 12.72) rather than
location drift. A further 16 trials predate the drift metric; one of them —
large_play's `limlikvn` — was classified from its paired diagnostic re-run
(§3.6.3), leaving **15 unclassifiable**. None was a winner candidate:
`limlikvn` is eligible on the borrowed diagnostics (loc_z 0.84, scale_z 1.07,
clamp 0) but its CE of 0.2148 ranks below large_play's winner at 0.2047, so the
re-run changed its label without changing the outcome. In large_diverse
8 of the top 11 trials rejected; in large_play 5 of the top 7. The best-scoring
configurations are disproportionately the ones not sampling a stationary
function-space measure — precisely the trade §3.6.3 exists to refuse.

**No architecture pattern survives across variants**: the winners span width 64
to 1024 and depth 2 to 6. `map_amp2` spans 9.49e+04 to
9.26e+05, with large_play's at 93% of the 1e6 cap — see §7 for
why that is weak evidence of range limitation rather than a finding about prior
strength.

### Stage 1 — MR (`MR-training`, metric `eval_loss_best`) — complete, unaffected

| variant | sweep | winner | trial / trigger | metric | width | depth | lr |
|---|---|---|---|---|---|---|---|
| medium_play | `70742ym5` | `za1bgyme` | 18 / 33 | 0.125673 | 8 | 5 | 6.240e-3 |
| medium_diverse | `vilrah4f` | `68b2hjeh` | 18 / 33 | 0.237777 | 6 | 1 | 4.818e-3 |
| large_play | `qkjet6r3` | `r8vpz8s5` | 15 / 30 | 0.154711 | 9 | 4 | 9.378e-3 |
| large_diverse | `59czpdwf` | `88wg34ln` | 5 / 20 | 0.210400 | 7 | 3 | 5.840e-3 ⚠️ |

Transcribed into `scripts_mr/antmaze_<variant>_mr_antmaze_eval.yaml` with
provenance headers and `criteria_key: loss`.

### Stage 1 — PT (`PT-training`, metric `eval_loss_best`) — complete, unaffected

| variant | sweep | winner | trial / trigger | metric | embd | head | layers | lr |
|---|---|---|---|---|---|---|---|---|
| medium_play | `z6nrw1vy` | `hpxovx0h` | 24 / 39 | 0.115663 | 7 | 7 | 1 | 8.561e-3 |
| medium_diverse | `sridqxoj` | `d74wbyb5` | 41 / 56 | 0.268226 | 6 | 7 | 2 | 1.365e-5 ⚠️ |
| large_play | `1z6xo2u0` | `74db66mr` | 2 / 17 | 0.056689 | 6 | 6 | 2 | 2.304e-3 |
| large_diverse | `gjphiwvs` | `b1ep2pcc` | 21 / 36 | 0.184459 | 8 | 6 | 1 | 7.608e-3 |

Transcribed into `scripts_pt/antmaze_<variant>_pt_antmaze_eval.yaml` with
provenance headers and `criteria_key: loss`. Rule winner equals best-of-all for
all four.

⚠️ **medium_diverse is a boundary winner:** `lr = 1.365e-5` against a swept floor
of 1e-5, so the optimum may lie below the searched range. The range is
pre-registered and was **not** widened; record it as a limitation. The note also
lives in the config file itself.

### ROUND 1 (superseded) — BNN warm-up tier, metric `warmup_final_nll`

| variant | sweep | winner | trial / trigger | nll | width | depth | n_meas | map_amp2 |
|---|---|---|---|---|---|---|---|---|
| medium_play | `kk79h8xf` | `05byzfhm` | 41 / 56 | 0.204278 | 6 | 2 | 14 | 313.204 |
| medium_diverse | `pyrz4qou` | `bk27aibh` | 23 / 38 | 0.316808 | 10 | 6 | 35 | 772.779 |
| large_play | `jhpdsl60` | `3orxv3kl` | 10 / 25 | 0.227336 | 7 | 6 | 10 | 623.485 |
| large_diverse | `in2p7l17` | `st3a5fgh` | 17 / 32 | 0.235876 | 6 | 3 | 11 | 459.295 ⚠️ |

**A round-1 finding worth re-testing, not citing.** Every variant selected a
large `map_amp2` (313–773) with a non-trivial `n_meas` (10–35) — the functional
prior retained everywhere at roughly 18–28× the legacy amplitude. It replicated
an earlier, less clean run, so it is probably real. But it was selected on
`warmup_final_nll`, the metric round 2 retires, so **cite the round-2 numbers
instead once they exist.** If round 2 reproduces it across all four variants
under an end-to-end metric and a range extended to 1e4, that is a considerably
stronger result than round 1 could have supported.

### ROUND 1 (superseded) — BNN sampling tier, metric `val_mean_cross_entropy`

Sweeps: `ld9oi90s` (medium_play), `o9g70yby` (medium_diverse), `u5snid84`
(large_play), `gnlrcb7y` (large_diverse). These are the **second** stage-2
attempt; the first was discarded when the warm-up gate was removed (§3.5).

Verified 2026-08-08, all four fired.

| variant | sweep | winner | trial / trigger | val CE | diverged |
|---|---|---|---|---|---|
| large_play | `u5snid84` | `d1ddc4yg` | 10 / 25 | **0.210992** | 7 / 26 |
| large_diverse | `gnlrcb7y` | `k83frxm7` | 17 / 32 | 0.231372 | 0 / 35 |
| medium_play | `ld9oi90s` | `ge5h8lfd` | 5 / 20 | 0.244184 | 0 / 21 |
| medium_diverse | `o9g70yby` | `nfqz8f11` | 10 / 25 | 0.284295 | 8 / 25 |

All four winners have **rule winner = best-of-all** and are themselves clean
(not divergence-flagged). The winning schedules:

| variant | `sghmc_lr` | `sghmc_lr_max` | `cycle_length` | `mdecay` | `fraction_cool` |
|---|---|---|---|---|---|
| medium_play | 1.759684326975111e-4 | 2.7127155004169033e-3 | 2750 | 3.886566686095279e-2 | 0.26490734354273915 |
| large_play | 4.4675299100389534e-4 | 4.123441940823021e-3 | 1500 | 2.2440542355891248e-2 | 0.4464505988695707 |
| large_diverse | 2.2274405063627215e-4 | 4.820665555950937e-3 | 2500 | 8.916002139869199e-2 | 0.41191474620821766 |
| medium_diverse | 4.958689700913806e-4 | 2.1452292827590347e-3 | 2250 | 4.286099171785899e-2 | 0.14154720839003554 |

**medium_diverse resolved the §3.6.1 question without needing a decision.** Its
winner's margin over the clean field is comfortable (0.2843 vs 0.3182 for the
next clean trial, `lzyn872v`, ~12%), and although five divergence-flagged trials
sit at 0.300–0.327 — between the winner and the rest of the clean field — none
ever led. No flagged trial was ever a candidate winner in any of the four sweeps,
so the "carry forward a flagged winner?" question never had to be answered. It
would still have to be answered if stage 2 were ever re-run.

**Divergence is confined to the depth-6 networks.** medium_play (depth 2) and
large_diverse (depth 3) produced zero divergent trials in 56 combined; large_play
(depth 6) hit 27% and medium_diverse (depth 6, width 1024) 32%. Note large_play
had the most hostile search yet produced the best and cleanest winner — the
blow-ups were confined to bad regions of schedule space and never touched the
selected configuration. Per §3.6.1 the medium_diverse count overstates the
instability: 3 of its 8 are outright blow-ups, the other 5 are 1.1–2.1%
over-clip with otherwise healthy diagnostics.

**Predictive-tail health of the four winners at the stage-2 budget** (132 draws;
the distributional statistics only — see §4.5, which also corrects the earlier
claim that the `_max`/`_min` extremes were censored). medium_diverse is the weakest and
should be expected to need the largest stage-3 draw budget, consistent with its
being the largest network:

| variant | `cvar_ess_median` | `cvar_rhat_median` | `cvar_rhat_pct_over_1.01` | `folded_rhat_95th_pct` |
|---|---|---|---|---|
| large_diverse | 81.8 | 1.024 | 76.6% | 1.359 |
| medium_diverse | 49.7 | 1.071 | 97.8% | 1.149 |

These rank *schedules at a fixed small budget* and are not acceptance criteria;
stage 3 re-reads them at the production draw count.

**Discarded first attempt**, for the record (sweeps `zkkg4kdu`, `jpu2vqce`,
`7kfieu41`, `c05yyh72`): ~234 GPU-hours and 47 trials written off when the gate
was removed. None had fired, so no winner was ever read from them; they inform
§3.5 only.

## 7. Disclosures required in the write-up

Split into what is settled and what is still pending round 2.

### 7.1 Settled — report these regardless of how round 2 turns out

**The BNN search was redesigned mid-project, and the first design was
discarded.** This is the most important disclosure in the document and it should
be stated plainly rather than buried. The original two-tier BNN search
(architecture selected on a warm-up-only metric, then a schedule selected on top
of it) produced a configuration that was **non-stationary at production length**:
run at the full draw budget its weight norm grew 4.09× across 310 draws in all
eight chains, validation CE degraded monotonically 0.2843 → 0.4252, and *every*
convergence diagnostic improved while it happened, because a drifting chain's
growing within-chain variance pushes R-hat toward 1 and inflates ESS. Per §0's
standing rule the round was restarted from scratch rather than patched, so no
post-hoc adjustment appears in the reported procedure. §3.7 has the full
account.

Two methodological points are worth making in their own right, since they
generalise beyond this project:

- **Convergence diagnostics can be actively misleading, not merely silent,**
  when a sampler is drifting. Anything certified on R-hat/ESS alone is unsafe
  unless drift is separately excluded.
- **Selecting a sampler at a much shorter horizon than it will be deployed at
  biases toward instability**, rather than merely failing to detect it: at a
  short horizon a larger step size buys a better score and its cost has not yet
  appeared. Round 1 selected at 35 draws and deployed at 310.
- **A drift check that is a significance test will clear a drifting sampler at
  small sample sizes.** Round 1's lesson was to exclude drift separately; stage 3
  found that the drift metric itself has the same failure mode one level down.
  `fn_drift_loc_z` divides a first-half-vs-second-half shift by an MCSE, so its
  power grows as ~√(chains) and a fixed threshold clears low-chain-count runs
  that are drifting exactly as much as the ones it rejects (§4.2.1, §4.3.2).
  Report the effect size (`loc_sd`) next to the test statistic, and check that
  it falls as 1/√draws rather than that the z-score sits under a threshold.
  This is the disclosure that the four stage-1 winners' §3.6.3 stationarity
  acceptance rests on: it was a z-gate PASS at 4 chains, which §4.3.2 shows is
  not by itself evidence of stationarity.

**Budget.** Every family received `run_cap: 130` under the same stopping rule,
and sweeps ran until the rule fired rather than to the cap. Note the invariant
is one-sided — the fairness claim requires the BNN to get *no more* tuning than
the baselines — so a BNN sweep that stops earlier strengthens it. Report also
that the BNN searches 9 dimensions against MR's 3 and PT's 4 at the same cap,
i.e. thinner coverage per dimension for the proposed method.

**Rule-vs-best disagreement (MR/large_diverse).** The stopping-rule winner is
not the best trial observed: t5, 0.210400 against t27's 0.203892 — the rule
winner is 3.2% worse. State the gap convention explicitly (rule ÷ best − 1; the
reverse convention gives 3.1%) and use it consistently. The discarded config
sits in a very different regime (lr 2.579e-4 vs 5.840e-3), so the sweep found a
distinct basin late rather than a near-tie. This affects a **baseline**, i.e. it
costs the comparison nothing in the proposed method's favour.

**Boundary winner (PT/medium_diverse).** `lr = 1.365e-5` against a swept floor
of 1e-5, so the optimum may lie below the searched range. Pre-registered and not
widened; record as a limitation.

**Search-range changes between rounds.** Two BNN ranges were expanded for round
2 — `map_amp2` 1e3 → 1e4 and `mdecay` 1e-1 → 1.0 — on the ground that round-1
winners sat against those caps (89–95% for `mdecay`; the top ~20% of log-space
for `map_amp2`). Nothing was narrowed. `width`/`depth` were left alone despite
round-1 ceiling hits, because those hits occurred under the retired metric.
Disclose the expansions: they were decided from round-1 results, which is
legitimate only because round 1 was discarded wholesale and round 2 was
pre-registered before it ran.

**Stage 4's statistic is a max over 200 checkpoints** (§5) — optimistic relative
to final-checkpoint reporting, applied identically to every method, and worth
naming rather than leaving implicit.

### 7.2 Round-2 BNN results — settled

**Winner selection was constrained, and it cost about 2%.** The winner is the
lowest-CE trial that satisfies §3.6.3, which in two of four variants is not the
lowest-CE trial:

| variant | winner CE | lowest-CE trial | gap |
|---|---|---|---|
| medium_play | 0.193983 | `ecfd7ko7` 0.190494 (rejected, scale_z 2.85) | +1.8% |
| large_play | 0.204685 | `0h1oko58` 0.200631 (rejected, z 2.03/4.56) | +2.0% |

Quote these as *"the eligible winner is X% worse than the lowest-CE trial"*
(winner ÷ best − 1), matching the convention in §7.1. State plainly that the
rejected trials were excluded for failing a **pre-registered** stationarity
criterion, not for scoring badly.

**The eligible frontier was still improving when two sweeps stopped.** The K=15
rule tracks the raw metric, so a sweep can fire while eligible progress
continues. It did, in medium_play (best eligible trial 13 before the trigger) and
large_play (8 before). Disclose this as a limitation of applying acceptance as a
filter over an unconstrained search; §3.6.3 records why the rule was
deliberately not made eligibility-aware.

**Rejection counts** (report per variant): medium_play 17/40, large_play 12/29,
medium_diverse 14/42, large_diverse 22/40 — **65 of 151 trials, 43%**, plus 16
that predate the drift metric, of which one was recovered by a paired
diagnostic re-run, leaving **15 unclassifiable** (3 / 4 / 5 / 3 by variant).
Quote 15 as the unclassifiable count and 16 as the number lacking their own
diagnostics; they are different quantities and the distinction is the re-run
mechanism §3.6.3 specifies. The criteria are **not**
lenient in practice: they reject roughly half the search. §3.6.3 describes the
z ≤ 2 threshold as deliberately permissive relative to the |N(0,1)| null, and
that is true of the threshold, but the observed drift is large enough that it
still excludes most trials.

**The four searches were very uneven in how many valid candidates they
contained, and this should be reported rather than averaged away.** Rejection
rates run from 33% (medium_diverse) to **55% (large_diverse)**, which leaves
large_diverse with only **15 eligible trials out of 40** — roughly a third of
what medium_diverse had. Its winner `yjezedlk` satisfies every criterion, so the
result stands, but the effective search behind it was much thinner than the
headline "40 trials" suggests. State the eligible counts alongside the trial
counts; quoting the latter alone overstates how much of each space was
searched with usable configurations.

**Mixing quality varies ~5× across the four winners, and the search never
optimised for it.** CVaR ESS runs 53 (medium_play) to 272 (medium_diverse) and
folded R-hat 1.027 to 1.270, on winners selected by a predictive-CE metric that
is blind to mixing. Across 140 trials the swept schedule parameters —
`sghmc_lr_max`, `fraction_cool` — show no correlation with any mixing metric
(|ρ| ≤ 0.13), while `depth` is the strongest single predictor (ρ = +0.46 with
CVaR ESS). Report the per-winner tail diagnostics rather than an average, and do
not describe the sampler schedule as having been tuned for mixing: it was tuned
for predictive cross-entropy, and mixing is whatever the selected architecture
happened to deliver. The per-variant cause is not identified — among the four
winners `width` lines up better than `depth`, which contradicts the
population-level correlation, and n = 4 cannot separate them.

Almost all rejections are
`scale_z` above 2 rather than location drift, and they concentrate near the top
of the ranking (8 of large_diverse's top 11; 5 of large_play's top 7). The
honest summary is that **the best-scoring configurations are disproportionately
non-stationary**, and the criteria refuse that trade.

**Budget.** All four fired at trials 29–42 against a cap of 130, so the BNN used
far fewer trials than the MR/PT baselines' sweeps. That is the conservative
direction for §3.1's one-sided invariant and strengthens the fairness claim.

**The prior-strength result did not replicate, and should not be reported as
one.** Round 1 had all four variants selecting `map_amp2` in a tight 313–773
band, which read as a finding. Round 2's winners span 9.5e4 to 9.3e5, and within
a sweep the metric is close to flat across decades of amplitude — in large_play, eligible
trials spanning `map_amp2` 157 to 925,895 — nearly four decades — all scored
between 0.2047 and 0.2400. Combined with §3.6.2's
observation that mean pooling alone implies an amplitude ~1e4 times the
sum-pooling value, the defensible statement is that **predictive CE does not
identify prior strength above a floor**, not that a large amplitude is preferred.
`n_meas` behaves the same way: large_diverse's winner uses 7 measurement points
and large_play's 29, with no consistent ordering.

**Boundary winner.** large_play's `map_amp2` = 925,895 sits at 93% of the 1e6
cap. Flag as possibly range-limited, but note it is weak evidence: the metric is
flat over decades there, so the optimiser drifts upward without a sharp optimum
rather than pressing against a real boundary.

**Still pending: stage 3 and stage 4.** No draw-budget or normalization result
exists yet.

---

## 8. Tooling

**`launch_hp_sweeps.sh <bnn|baselines>`** (repo root) — creates and runs the
sweeps on the GPU box. The old `phase1`/`phase2` arguments are gone: those
numbers encoded the retired two-tier BNN structure. `bnn` launches the four
round-2 merged sweeps on GPUs 0–3; `baselines` launches the eight MR/PT sweeps,
which are already complete and reuse their cached ids. There is no combined
mode — the two GPU maps overlap.

Preflights the seed-0 data splits, tuning sets, env, and GPU count; rejects any
`FILL_ME` left in a sweep yaml; and, for `bnn`, **refuses to launch if a base
config sets `burn_in_lr`** — burn-in must inherit the swept `sghmc_lr`, and a
base config that overrides it would have the sweep scoring configurations it is
not actually running, invisibly (§3.7). Caches sweep ids per set
(`exp/sweep_ids_bnn_round2.txt`, `exp/sweep_ids_phase1.txt`) so re-runs resume
rather than duplicate, with the BNN set on a fresh file so it cannot resurrect a
retired tier sweep. Exports the §10.7 thread caps, matching `train_rewards.sh`,
so selection and evaluation runs share numerics. Launches exactly **one agent
per sweep** (serial trials give the Bayes optimiser full history, and the eval
scripts write to deterministic per-seed output paths that two concurrent runs
would clobber).

Note it does **not** refuse on the `SUPERSEDED-ROUND1` marker that
`train_rewards.sh` blocks on. That is deliberate: the merged sweep overrides
every swept field of its base config, whereas `train_rewards.sh` trains from
those values directly.

**`check_winner_eligibility.py`** (repo root) — applies the §3.6.3 acceptance
criteria and names the winner, which `check_sweep_convergence.py` does not: that
script reports the best-*metric* trial, which is not the same thing. Ranks the
trials up to the stopping trigger, applies the pre-registered thresholds, and
reports the winner, the gap to the lowest-metric trial (disclose it) and the
rejection count. Resolves paired diagnostic re-runs automatically by matching
swept parameters — normalising `width`, since a sweep trial logs the log2
exponent while a hand-launched run logs the expanded value, and an unnormalised
comparison silently matches nothing.

**`check_sweep_convergence.py`** (repo root) — evaluates the stopping rule
out-of-band:

```bash
python check_sweep_convergence.py --entity champlin-university-of-arizona \
    --patience 15 BNN-training/zkkg4kdu MR-training/70742ym5
```

Reports progress vs cap, whether the rule fired and at which trial, the rule
winner *and* best-of-all with their gap, and an **unsynced-trial scan**. It
sorts trials chronologically — this is essential and easy to get wrong, since
the wandb API returns runs in name order and the patience rule depends on
ordering. `--emit-prior-runs` prints `-R <run_id>` flags for carrying finished
trials into a new sweep.

It also reports `!! DIVERGED` for trials whose chains blew up numerically (§3.6)
— these are scored, so they are *not* unsynced, but they carry no usable ESS or
R-hat.

**Null sweep parameters do not survive the CLI.** A wandb agent passes each
sweep parameter to the program as a command-line argument, so a `value: null`
arrives as the *string* `None` and pyrallis rejects it:

```
pyrallis.utils.ParsingError: Failed when parsing value='None' into field
"TrainConfig.early_stop_acc_threshold" of type typing.Optional[float]
```

Use a value that both parses and is semantically inert (here `0.0`, since
accuracy is never negative). `null` is fine in a *config* yaml, which pyrallis
reads directly rather than through argv — the base eval configs use it.

**Unsynced trials.** The GPU box's wandb connection drops intermittently. A
trial can finish locally but fail to upload its final metrics; wandb marks it
crashed and often auto-flips it to finished when the next trial starts, so the
loss is easy to miss. The fingerprint is a finished/crashed trial that was *not*
legitimately early-stopped yet has no metric. Recover with `wandb sync <run-dir>`
on the box — otherwise the Bayes optimiser never sees the result and the trial
slot is wasted. Ground truth from the box is `wandb sync --sync-all --dry-run`.

---

## 9. Do-NOTs

- Do not select any hyperparameter on evaluation seeds 1–10, at any stage.
- Do not change a search range, gate, or fixed value in response to observed
  sweep behaviour. If something is genuinely broken, restart the affected
  sweeps from scratch and say so.
- Do not stop a sweep before its rule fires; do not read a winner from a sweep
  that has not fired.
- Do not select on accuracy anywhere in stages 1–3.
- Do not tune `map_eta` or `map_sig_*` on anything downstream.
- Do not judge stage 3 by `param_*` or bulk predictive diagnostics.
- Do not compare post-`bt_pool="mean"` reward magnitudes or CVaR values to
  pre-fix runs; only the scale-free convergence diagnostics carry over.

---

## 10. Current state and how to resume

Written for someone joining cold. Read §1 (seed discipline) and §3.1 (budget,
stopping rule, metric) first; everything else can be looked up as needed.

### 10.1 What is finished

| stage | family | state |
|---|---|---|
| 1 | MR | 4/4 fired; winners in `scripts_mr/antmaze_<v>_mr_antmaze_eval.yaml` |
| 1 | PT | 4/4 fired; winners in `scripts_pt/antmaze_<v>_pt_antmaze_eval.yaml` |
| 1 | BNN | **4/4 fired** (round 2, merged); winners in `scripts_bnn/antmaze_<v>_bnn_antmaze_eval.yaml` |
| 3 | BNN | **halted at `c16`, by result** — medium_play `c4`/`c8`/`c16` measured (§4.3.1, §4.3.2), plus the half-split (§4.3.3), the per-chain drift (§4.3.5) and the non-cyclical control (§4.3.6). The ladder's axis is orthogonal to the binding constraint: a drift common to 14/16 chains that shrinks with neither draws nor chains. No budget selected; `c32` is not to be run. The cyclical schedule is cleared (§4.3.6). The 8-chain `chain_init_jitter` test is inconclusive — 8 chains is below the resolution of both ALIGNMENT and `loc_sd` (§4.3.7). Next is the same test at **16 chains** |
| 4 | all | not started |

The BNN configs carry a round-2 provenance header recording the sweep, winner,
trial/trigger, predictive CE and accuracy, all three acceptance numbers, the
rejection count, and any applicable disclosure. Verified: all nine selected
fields match the winning wandb run for every variant, `burn_in_lr` is absent,
`num_samples` is 75 and `num_burn_in_steps` is 20000.

The `SUPERSEDED-ROUND1` markers are gone, so `train_rewards.sh` will now launch.
It should not be launched yet — `num_chains` / `chains_per_gpu` still hold
round-1 reference values that stage 3 sets.

### 10.2 Immediate next action

**Stage 3: the BNN draw budget (§4).** Stage 1 is complete for all three
families; nothing is waiting on a sweep.

Stage 3 raises **`num_chains` only**, with `num_samples` pinned at the sweep's
75 draws/chain — selection and production must run at the same horizon, which is
the round-1 mistake §3.7 exists to prevent. `chains_per_gpu` is placement.

Judge it on the CVaR tail diagnostics (§4), reading the median / 95th-pct /
`pct_over_1.01` variants rather than the `_max`/`_min` extremes (§4.5 explains
why those are sparse rather than censored), and **check `fn_drift_*` and
`param_clamp_sampling_pct` first**: a run that is not sampling the target
measure makes every tail number meaningless. Read the raw `loc_sd` alongside
the z-scores — the z's are not comparable across chain counts, and a PASS at a
low count is not evidence of stationarity (§4.2.1). The winners pass the
z-gate at 4 chains × 75 draws; §4.3.2 shows that PASS does not survive
contact with a longer ladder.

**Progress so far — and why the ladder stopped.** medium_play's `c4`, `c8` and
`c16` rungs are done (2026-08-16 / 08-17 / 08-17), recorded in §4.3.1 and
§4.3.2. `c16` changed the question. Raw `loc_sd` — the drift effect size, which
unlike the z-scores does not depend on chain count — went 0.4319 → 0.4222 →
0.6460 while stationarity requires it to fall as 1/√draws (0.4319 → 0.3054 →
0.2160). It never falls. **There is a real within-chain location drift at every
rung, and the `c4`/`c8` gate PASSes were low-power false negatives** (§4.2.1).

Because `function_space_drift` splits each chain's *own* draws in half, adding
chains cannot reduce that drift — it only measures it better. Stage 3 as
specified (§4.1: `num_chains` only, `num_samples` pinned at 75) therefore
ladders an axis orthogonal to the binding constraint. **No budget can be
selected from this ladder**, and the flat `cvar_mcse_rel_median` across 4× the
compute (0.220 → 0.271 → 0.223) is the same fact seen from the tail.

**Since then**, `--per-chain-drift` showed the drift is common to 14/16 chains
(alignment 0.7564, §4.3.5), and the compute-matched non-cyclical control cleared
the cyclical schedule while incidentally supplying the empirical support §3.6.2
lacked for it (§4.3.6). The `chain_init_jitter` test meant to separate the two
remaining hypotheses was run at 8 chains and is **inconclusive**: ALIGNMENT and
`loc_sd` both vary more between halves of one run than the effect being tested
(§4.3.7). The cause is still unidentified; the shared start and weight-space
diffusion are both still live, and the 16-chain jitter run distinguishes them.

**Do this next, in order:**

1. **Do not run `c32`, and do not start the other three variants' ladders.**
   A further doubling buys power to detect the same within-chain drift, not
   less drift, at the cost of a full run. The other three variants show the
   same signature at `c4` — raw `loc_sd` 0.21–0.43 while all four pass the
   z-gate (§4.3.2) — so laddering them would reproduce this result three more
   times rather than test it.

2. **Isolate which chains drift.** The half-split is **done** (§4.3.3): the
   upper half drifts 2.10× more on raw `loc_sd`, and the half that drifts more
   has the *better* tail statistics, so `c16`'s unresolved-count improvement is
   a drift artifact rather than progress. It also refuted the hypothesis it was
   built to test — `chain_init_jitter = 0` means there is no per-chain
   initialisation, so the chains are exchangeable and a lower-vs-upper gap
   cannot be a property of "later" chains.

   `--per-chain-drift` is **done** (§4.3.5): alignment 0.7564, 14/16 chains
   drifting the same way (sign test p = 0.0042), no outlier chains. The drift
   is common to essentially every chain, which settles §4.3.2's non-shrinking
   `loc_sd` on direct evidence. It also explains §4.3.3's 2.10× — two
   counter-drifting chains happened to land in the lower half — and closes the
   GPU question, since those two sit on different GPUs.

3. **The cyclical schedule is CLEARED — done 2026-08-19 (§4.3.6).** Turning it
   off at matched compute made the drift worse on every axis and cost 6.3
   accuracy points, so the drift is intrinsic to the sampler or model. Do not
   build on the `nocyc` configuration. Remaining levers, reordered on that
   evidence:

   - **`chain_init_jitter 0.1` at 16 chains, cyclical — do this first.** It is
     a *discriminating* experiment, not just an R-hat fix: if the common drift
     comes from the shared start, jittering must drop ALIGNMENT from 0.7564;
     if alignment holds, the shared start is dead and the cause is weight-space
     diffusion (§4.3.6). **The 8-chain version was run 2026-08-19 and is
     inconclusive** — 8 chains is below the resolution of both ALIGNMENT and
     `loc_sd` (§4.3.7). Use 16, and compare against `c16`, not `c8`.
   - **Test the diffusion hypothesis** if jitter does not move alignment: check
     whether `loc_sd` tracks ‖w‖ growth across draws (§3.6.2). §7.1 records the
     round-1 norm growing 4.09× in every chain, so this is the mechanism with a
     precedent in this project.
   - **`num_burn_in_steps`** — weak prior. A transient would have to survive
     burn-in and still move `f` by 0.65 sd between steps ~100 000 and ~206 000.
   - **`num_samples`** — last resort; it breaks the §3.7 selection/production
     horizon match, so it changes what stage 1 selected.

   Run on medium_play alone until something moves raw `loc_sd`, and judge on
   `loc_sd` and ALIGNMENT — never on the tail statistics, which §4.3.3 shows
   improve as drift gets worse.

4. **Only then** resume the ladder, transcribe `num_chains` / `chains_per_gpu`
   into `scripts_bnn/antmaze_<v>_bnn_antmaze_eval.yaml`, and re-run the
   field-by-field verification (§10.3).

5. **Then stage 4** (§5) — the 8-way normalization grid, which runs outside this
   repo in the surrounding `iqlpref` pipeline.

The §4.1 write-up conclusion stands and is strengthened: the tail is not
reachable by adding chains at a 75-draw horizon. The reason is now identified —
the sampler is not stationary at that horizon — which makes it a result about
the cyclical schedule composed with fSGHMC (§3.6.2's known deviation), and it
belongs in the paper.

**Always capture the diagnostic to a file next to the run.** The
unresolved-point count and the `--worst-k` listing are the two things §4.6 asks
to be recorded that are *not* logged to wandb, so the terminal is their only
copy otherwise, and `exp/` is gitignored — the file stays local and can be
pulled off the box for analysis. The `c4`/`c8`/`c16` captures are in `exp/` on
the analysis Mac.

Every stage-3 run needs a distinct `OUT_DIR`. The deterministic
`{OUT_DIR}_{seed}` path has already destroyed evidence three times (§10.3), and
two runs sharing a path will silently clobber each other.

### 10.3 The BNN production configs — done for round 2

Completed 2026-08-15 for all four variants. Kept because it documents a **trap**
that recurs whenever these configs are regenerated.

`map_amp2` is absent from `TrainConfig`'s defaults in spirit only — it defaults
to `1.0`, and `burn_in_lr` defaults to `None`. In round 1 the first was a trap
(a missing line silently discarded the prior-amplitude result) and the second
was a bug (a present line reinstated the burn-in mismatch that ended the round).
Round 2 inverts the second: **`burn_in_lr` must stay absent**, because burn-in
inherits the swept `sghmc_lr`, and `map_amp2` must be present.

Each config ends up with:

| field | source |
|---|---|
| all nine swept fields | the merged sweep's winner — one run, one provenance block (§6) |
| `burn_in_lr` | **absent** — adding it back breaks the merge (§3.7) |
| `num_samples: 75`, `num_burn_in_steps: 20000` | pinned to the sweep (§3.2, §4) |
| `num_chains`, `chains_per_gpu` | **stage 3 sets these** — currently round-1 reference values |
| `n_discarded` | owned by no stage; decide deliberately |
| `seed` | leave at 1; `train_rewards.sh` overrides per evaluation seed |

**Verification performed:** each config parsed with `yaml.safe_load` and compared
field-by-field against the winning wandb run — all nine matched for all four
variants, `burn_in_lr` absent, `num_samples`/`num_burn_in_steps` correct, and the
`SUPERSEDED-ROUND1` marker cleared (which is what re-arms `train_rewards.sh`).

`bt_pool: "mean"`, `clip_during_sampling: false`, `clip_grad_norm_value: 100.0`,
`samples_per_cycle: 1`, `chain_init_jitter: 0.0` and now `burn_in_lr: None` are
left to `TrainConfig` defaults, which carry the intended values. This is the one
place §3.4's "never read from a dataclass default" is knowingly relaxed, because
these are uniform project-wide and the sweep yamls say so explicitly.

**Do not run `train_rewards.sh` yet.** The marker is cleared so it will launch,
but `num_chains`/`chains_per_gpu` are still round-1 values until stage 3 sets
them.

### 10.4 Then: stage 3, then stage 4

**Stage 3 is specified in full in §4** — what it chooses, the stationarity
precondition, the measured starting point, the launch command, what to judge it
on, and when to stop. That section is the single source of truth; this one only
says where it sits in the sequence. Do not follow a summary of §4 written
elsewhere, including an earlier version of this paragraph, which told the reader
to check a `sampling_weight_growth` metric that no longer exists (§3.6.2
explains why weight-space statistics were removed).

Stage 4 (§5) is the 8-way normalization grid, selected on max mean IQL score at
seed 0. It runs outside this repo, in the surrounding `iqlpref` pipeline.

### 10.5 Sweep IDs

Entity `champlin-university-of-arizona`.

**In use:**

| family / stage | medium_play | medium_diverse | large_play | large_diverse |
|---|---|---|---|---|
| MR stage 1 | `70742ym5` | `vilrah4f` | `qkjet6r3` | `59czpdwf` |
| PT stage 1 | `z6nrw1vy` | `sridqxoj` | `1z6xo2u0` | `gjphiwvs` |
| BNN stage 1 (round 2) | `9gifb8sa` | `bq7ygeqe` | `ojk7k4vb` | `m5sp9bw9` |

Round-2 ids are cached in `exp/sweep_ids_bnn_round2.txt`. All four fired
(triggers 40 / 42 / 29 / 40) and their agents exited on their own.

**Superseded — do not read winners from these:**

| round-1 BNN tier | medium_play | medium_diverse | large_play | large_diverse |
|---|---|---|---|---|
| warm-up tier | `kk79h8xf` | `pyrz4qou` | `jhpdsl60` | `in2p7l17` |
| sampling tier | `ld9oi90s` | `o9g70yby` | `u5snid84` | `gnlrcb7y` |

Also superseded: the first clean-restart sampling-tier attempt (`zkkg4kdu`,
`jpu2vqce`, `7kfieu41`, `c05yyh72`), and everything from the pre-restart round
described in §0. The round-1 tiers above *did* fire and their numbers are real —
they are superseded because the design was discarded (§3.7), not because the
sweeps failed.

### 10.6 Monitoring habits worth keeping

Use `check_sweep_convergence.py` (§8) for every status check rather than reading
the wandb UI. It applies the stopping rule **in chronological order** — the wandb
API returns runs in name order, and getting this wrong silently reports the wrong
trigger point and the wrong winner. It also runs the unsynced scan and the
divergence scan (§3.6) in the same pass.

Two failure modes it catches that the dashboard hides:

- **Unsynced trials.** The GPU box's wandb connection drops intermittently; a
  trial can finish locally but never upload its metric, and the dashboard often
  auto-flips it from crashed to finished, so the loss is invisible. Recover with
  `wandb sync <run-dir>` on the box.
- **Divergent trials.** Scored normally, so not unsynced, but the chains blew up
  and every convergence diagnostic is NaN — unusable for stage 3.

Operational trap: killing a sweep agent does **not** kill its training process.
The child is orphaned and keeps running on the GPU. Kill by process group and
match the full config name (`antmaze_large_play_bnn_antmaze_eval`, not `large`),
or you will take down a neighbouring variant. This has happened.

### 10.6.1 Where things run

Two environments, and the split matters for anything you try to do:

- **Training runs execute on `leviathan`**, the lab GPU box (6× RTX A6000, conda
  env `pt`). Everything in §4.4, `launch_hp_sweeps.sh` and `train_rewards.sh`
  runs there. `data/` is gitignored and lives on the box (and locally); it is
  not in the repo.
- **Analysis runs locally** against the wandb API — `check_sweep_convergence.py`,
  `check_winner_eligibility.py`, and any ad-hoc query. On the analysis Mac use
  `/opt/anaconda3/envs/irl/bin/python`; on the box the `pt` env's `python` is
  fine. No GPU or box access is needed for any of it, which is why every reading
  in this project was produced without touching `leviathan`.

`diagnose_sampling_tail.py` is the exception: it needs a run's saved chains, so
it runs on the box (or wherever `OUT_DIR` was written).

If you are an assistant working on this: **do not log into `leviathan`.** Hand
the user the exact command to run and interpret the output they paste back. The
wandb API plus the repo answers more than it first appears — run `config`,
`summary` and `metadata` (which carries `host`, `cpu_count`, `gpu_count`) cover
most "what is the box doing" questions without any shell access.

### 10.7 Concurrent sweeps oversubscribe the CPU (~2.8×) — confirmed

Measured on medium_diverse (`o9g70yby`), 2026-08-08. **Compare trials by
h per 1000 sampling steps, not by wall-clock duration** — a stage-2 trial runs
`num_samples × cycle_length` sampling steps, and `cycle_length` is swept over
500–3000, so raw duration is dominated by the schedule under test and hides
throughput changes entirely.

| period | concurrent sweeps | h / 1k sampling steps |
|---|---|---|
| t1–t11 (Jul 31 – Aug 6) | 4 | 0.15–0.27 (median 0.16) |
| t13–t15 (Aug 6–7) | 3 → 2 | 0.090–0.103 |
| t16–t17 (Aug 7–8) | 1 | 0.056–0.060 |

The step-downs align with the other sweeps' last heartbeats (`ld9oi90s`
Aug 6 12:52, `u5snid84` Aug 7 05:47, `gnlrcb7y` Aug 7 20:35 UTC).

**It is not GPU contention.** `launch_hp_sweeps.sh:171` launches each agent under
`CUDA_VISIBLE_DEVICES="$gpu"` with a distinct GPU per sweep, and a stage-2 trial
needs exactly one (`num_chains: 4`, `chains_per_gpu: 4` → `rank // chains_per_gpu`
= 0 for all four chains). Four sweeps used four of `leviathan`'s six A6000s, none
shared. Nor is it raw core starvation: the box has 255 logical CPUs.

**It is CPU oversubscription.** Nothing in the repo sets `OMP_NUM_THREADS` /
`MKL_NUM_THREADS` or calls `torch.set_num_threads`, so every process defaults its
intra-op pool to the core count. Each trial spawns four chain *processes* via
`mp.spawn` (`optbnn/sgmcmc_bayes_net/f_pref_net.py:954`), and the BNN dataloaders
run `num_workers: 0` (ibid. :588), so CPU-side work stays in-process where that
pool applies.

Confirmed on the box during trial 18, uncontended, with
`ps -o pid,nlwp,pcpu,comm -C python --sort=-pcpu`:

| | NLWP | %CPU |
|---|---|---|
| 4 chain processes | 385 each | 3814 + 3791 + 3776 + 3772 = **15,153%** |
| parent | 389 | 1,337% |
| **one trial** | ~1,925 threads | ~16,490% ≈ **165 of 255 cores** |

So **a single trial already consumes 65% of the box**, and four concurrent trials
demand ~660 cores against 255 — **2.59× oversubscribed, against the 2.8× slowdown
measured from throughput**. The two figures are derived independently (CPU demand
vs h/1k sampling steps); the residual is context-switch and cache-thrash overhead
on top of pure queuing.

Note `ps` reports %CPU as a *lifetime average*, which is what makes this reading
usable: these processes lived entirely within the single-sweep period, so 165
cores is the trial's unconstrained demand — the right input for the arithmetic.
Read the same way during a contended period it would be suppressed, not
informative.

**Most of that CPU was waste — confirmed, and fixed.** 38 cores per chain is
anomalous for a GPU-resident sampler: the matmuls are small (width 64–1024,
batch 64), sizes at which a 255-thread BLAS pool is parallel overhead rather than
speed-up, and OpenMP's default active wait policy has idle threads spin instead
of sleeping. A/B on a single uncontended 8-chain medium_diverse job at seed 0
(`num_samples: 10`, `n_discarded: 0`), 2026-08-08:

| arm | wall time |
|---|---|
| default threads | 3 h 01 m 34 s |
| `OMP_NUM_THREADS=8`, `OMP_WAIT_POLICY=PASSIVE` | **2 h 12 m 10 s** (−27%) |

Capping is **faster**, not merely equal, which settles the question: the extra
threads were costing time. This is the single-job gain — an 8-chain job is ~8×
oversubscribed on its own, before any concurrency — and the ~2.8× multi-job
penalty above sits on top of it.

**Now set in `train_rewards.sh`** (`OMP_NUM_THREADS=8`, `MKL_*`/`OPENBLAS_*` to
match, `OMP_WAIT_POLICY=PASSIVE`), exported once so an entire campaign shares
them; each is overridable from the environment for deliberate experiments.

**Do not vary thread count within a campaign.** It alters floating-point
reduction order, so runs before and after are not strictly comparable. In
particular the stage-3 selection runs (seed 0) and the evaluation runs (seeds
1–10) must use the same setting.

*Disclosure:* stages 1 and 2 ran uncapped; stage 3 onward runs capped. The
selected hyperparameters therefore come from a slightly different numerical
environment than the runs that use them. This is reduction-order noise, far below
the differences the selection metric resolved, but state it rather than leave it
implicit.

This does not affect any selection result — throughput is not an input to any
stopping rule or metric — but it does affect every schedule estimate in this
document.
