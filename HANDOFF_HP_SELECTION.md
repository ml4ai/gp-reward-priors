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
| **`bt_pool: "mean"`** — the likelihood `Φ(f)` pools rewards by masked *mean* over timesteps, where the preference-learning literature uses the *sum* (return) | **Resolved 2026-08-11.** Applied identically in MR, PT and BNN, so cross-family comparability holds. Every segment is exactly T=100 valid timesteps (verified, all four variants, train and val), so mean = sum/100 *exactly*: the two are the same model up to a global temperature, and no length confound exists. One sentence in the paper, no further action.  **2026-08-20: tested directly.** A `bt_pool="sum"` run with `map_amp2` rescaled by 1e4 (the prior is zero-mean, `map_informed_prior.py:4`, so the kernel rescale is the whole correspondence) reproduced the model as claimed but not the sampler: gradients grow ~100x under sum pooling while `sghmc_lr` is a weight-space constant, so the effective step grew ~50x, every drift gate improved and predictive CE doubled (0.2076 -> 0.4158). The equivalence claim in this row is about the MODEL and it survives; the sampler is simply not scale-invariant. Retain `"mean"` in all three families -- switching gains nothing and would break the identical-pooling basis of this row (SS4.3.15). |

### 3.6.3 Winner acceptance criteria

Pre-registered 2026-08-12, **before any round-2 sweep fired**. A configuration
that samples something other than `P_{f|D}` is not a valid winner however good
its score, so selection is a *constrained* minimisation: the winner is the
**lowest `val_predictive_cross_entropy` among ELIGIBLE trials up to the stopping
trigger**.

> ### AMENDMENT 2026-08-24 — the drift criteria move to the centred component
>
> **What changes.** The two `fn_drift_*` criteria below are evaluated on
> `val_fn_drift_centred_*` instead of the raw metrics. Thresholds, null and
> reasoning are unchanged — the `|N(0,1)|` calibration applies identically to
> the centred statistic. The **offset** metrics are logged and reported but
> **never gate**.
>
> **Why.** The BT/CE likelihood is exactly invariant to `f → f + c` (§4.3.10):
> `Φ` pools by masked mean and cross-entropy on `[Φ₁, Φ₂]` depends only on
> `Φ₁ − Φ₂`, so the offset is unidentified by the data and drift along it
> cancels in every preference prediction — and a constant reward offset leaves
> the IQL greedy policy unchanged. **The raw statistic mixes that unidentified
> direction into the criterion.** §4.3.28 makes the consequence concrete: the
> best-sampling configuration produced in this project is stationary in the
> identified component (centred `loc_z` 0.6144, `scale_z` 0.6469, both at the
> 0.6745 null median) yet **fails the raw gate** at `scale_z` 2.2827, because
> all its residual drift sits in the offset. Under the criteria as written it
> would be rejected as ineligible, while trials that pass are disproportionately
> those whose offset is pinned by an over-tight prior — which §4.3.14 identified
> as the pathology that drove `map_amp2` to improperness in the first place.
> **The unamended criterion selects against the thing it exists to detect.**
>
> **This is a post-hoc change to a pre-registered criterion, and it is
> disclosed as one.** Under §0's standing rule the round is restarted rather
> than patched, and that is what happens here: the amendment governs the
> **redesigned stage-1 BNN sweep** (§10.2 step 3), not a re-adjudication of
> round 2. Round-2 winners were selected on the raw criterion and **cannot** be
> re-scored against the new one — the centred statistic needs the saved chains,
> and no sweep trial's chains survive (§4.3). Their provenance headers stand as
> a record of what was done, and §7.1 carries the disclosure.
>
> **Scope limits.** `param_clamp_sampling_pct` is unaffected: the momentum
> clamp is a hard nonlinearity in weight space with no offset interpretation.
> The divergence rule is unaffected. Nothing about the CE ranking changes here;
> that is §10.2 step 3's separate move to CVaR CE.
>
> **Implementation.** `util.function_space_drift` now returns
> `fn_drift_centred_*` and `fn_drift_offset_*` alongside the raw keys (24
> metrics, raw key names byte-identical), so every future trial logs all three
> and this criterion is evaluable from wandb without saved chains — the gap that
> made round 2 un-re-adjudicable.

**Eligibility.** All three must hold:

| criterion | threshold | why this number |
|---|---|---|
| `val_fn_drift_centred_loc_z_median` | ≤ 2.0 | The per-point stationary null is \|N(0,1)\|: median ~0.67, 95th ~2. A median at 2.0 means the *typical* point has shifted 3× the null median, so this is deliberately lenient — set to avoid rejecting on noise, not to be strict. **Centred** per the 2026-08-24 amendment. |
| `val_fn_drift_centred_scale_z_median` | ≤ 2.0 | Same null, same reasoning. |
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

> **Read these on the CENTRED metrics (amendment 2026-08-24, §3.6.3).**
> `fn_drift_centred_*` gates; `fn_drift_offset_*` is reported, never gated. Raw
> `f` mixes the identified shape with the likelihood-invariant offset, so a raw
> FAIL may be entirely in a direction that cancels in every preference
> prediction — §4.3.28 is exactly that case. Every raw figure quoted in §4.3
> predates the amendment; where a section quotes raw and centred side by side,
> the centred column is the one the criterion now reads.

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

> **Headline superseded by §4.3.11.** The `loc_sd` trajectory below is
> raw, and about half of it is the likelihood-invariant offset (§4.3.9).
> On the identified component `loc_sd` DOES fall with chains (0.3728 →
> 0.3232, obs/req 1.23 rather than 2.16), so "the drift never shrinks" is
> not right as stated. The conclusion that stage 3's axis is orthogonal to
> the binding constraint stands, for the reason §4.3.11 gives instead. The
> low-power-false-negative argument also stands — but for SCALE, where the
> effect size is identical at `c8` and `c16` (1.5913 vs 1.6026) while the
> gate flips PASS→FAIL.

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

> **DONE, and hypothesis 1 is REFUTED — see §4.3.8.** First run at 8 chains,
> which was uninformative (§4.3.7: ALIGNMENT varies by 0.41 between halves of a
> single jitter-0 run, against a jitter effect of 0.046). Redone at 16 chains:
> **ALIGNMENT held at 0.7216 against `c16`'s 0.7564**, where this hypothesis
> predicted a collapse toward 0.25. The shared start is not the cause, and
> hypothesis 2 — weight-space diffusion — is the only one left.

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

### 4.3.8 The shared start is refuted — measured 2026-08-19

`chain_init_jitter 0.1` at **16 chains**, cyclical, against `c16` (identical but
jitter 0). This is the matched comparison §4.3.7 said was needed.

| metric | `c16` jitter 0 | `jit16` jitter 0.1 | |
|---|---|---|---|
| **ALIGNMENT** | **0.7564** | **0.7216** | −0.035 |
| signed shift | 14/16, p = 0.0042 | 12/16, p = 0.0768 | |
| raw `loc_sd` | 0.6460 | 0.5714 | −11.5% |
| `loc_z` | 2.5152 | 2.2037 | still FAILS |
| raw `scale_ratio` | 1.4664 | **1.5213** | worse |
| `scale_z` | 1.9962 | **2.2083** | PASS → **FAIL** |
| per-chain `loc_sd` max/median | 1.33× | 1.59× | |
| `rhat_bulk` median | 1.7037 | 1.6685 | |
| `ess_bulk` median | 26.83 | 27.51 | |
| unresolved points | 0.66% | 0.58% | |

**Hypothesis 1 is dead.** The shared-start hypothesis predicted that jittering
the starts would break the common component and drive ALIGNMENT toward 0.25.
It moved from 0.7564 to **0.7216** — 4.6% relative, still firmly in COMMON
territory. There is no C=16 replicate to bound the noise directly, but the
predicted effect was a collapse of two thirds and the observed change is a
rounding error against it, so the conclusion is robust to any plausible noise
level. Overdispersing the starts does not remove the common drift, therefore
the common drift does not come from the shared start.

**Jitter trades location drift for spread drift, and is not a fix.** `loc_sd`
improved 11.5% while `scale_ratio` worsened and `scale_z` crossed from PASS to
FAIL. This is the second independent run showing that pattern — the 8-chain
test did the same (§4.3.7, `scale_ratio` 1.4361 → 1.5750) — so unlike the
alignment reading it is replicated. Net stationarity is **worse**, not better.

**Its R-hat rationale is also not supported here.** Overdispersed starts are
supposed to *raise* R-hat, making it an honest convergence check rather than a
flattered one (`f_pref_net.py:130`). `rhat_bulk` median instead fell slightly,
1.7037 → 1.6685. The change is small enough to be noise, but it is the wrong
direction, and nothing in this run argues for keeping jitter on. **Leave
`chain_init_jitter` at 0** absent a reason beyond this evidence.

> **Basis corrected by §4.3.11.** The `scale_z` worsening cited here is on
> RAW `f`, which is contaminated by the offset. On the identified
> component jitter is mildly BETTER on both axes (centred `ratio` 1.5047
> vs 1.6026, centred `loc_sd` 0.2764 vs 0.3232). It still does not fix the
> widening, so leaving it at 0 remains defensible — but not for the reason
> stated here.

**What remains: weight-space diffusion that is only approximately
f-preserving** (§3.6.2). Every alternative has now been eliminated by direct
test — the cyclical schedule (§4.3.6), the shared start (this section) — and
burn-in was argued down on the 11:1 sampling-to-burn-in ratio (§4.3.5). It also
has a precedent in this project: §7.1 records round 1's weight norm growing
4.09× in *all eight* chains, which is a common-across-chains mechanism and so
survives the alignment test that killed the shared start.

**`--weight-f-coupling` tests it, and needs no new sampling.** It correlates
per-chain ‖w‖ growth against per-chain f drift **across chains**: under the leak
hypothesis, chains that grew their weights more should have drifted more in f.
Significance is by permutation.

> **Ran 2026-08-19 and came back UNINFORMATIVE, not negative — see §4.3.9.**
> ‖w‖ growth is 1.51× in every chain to within ±1%, so the regressor has no
> variation and the across-chain test has no leverage. A common cause of a
> common effect cannot be detected between chains. The same output did show
> that most of the drift lies in the *global offset* of `f`, a direction the
> BT/CE likelihood is exactly invariant to, which is where §4.3.9 goes next.

This is *not* a reinstatement of the removed `--weight-trace`. §3.6.2 remains
correct that weight-space statistics say nothing about convergence on their
own; here ‖w‖ is used only as a **regressor** against an f-space quantity
measured independently, and no claim is read off ‖w‖ by itself. The tool also
prints a within-chain correlation and explicitly refuses it as evidence: if
‖w‖ and f are both monotone in draw index — which is what a drifting chain
looks like — they correlate near 1 whether or not one causes the other.
Validated on synthetic chains, within-chain r was 0.95 under a true leak and
0.97 under no leak, while the across-chain test gave r = 0.995 (p < 0.001) and
r = 0.117 (p = 0.67) respectively. **Read only the across-chain line.**

```
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_play_c16_0 --weight-f-coupling --device cuda \
    2>&1 | tee exp/stage3_medium_play_c16_0_wcoupling_diag_tail.txt
```

Run it on `c16` (jitter 0, the cleaner baseline) and on `jit16`. **Check the
`wGrowth` range before reading the result**: if the chains barely differ in
growth the regressor has no variation and the test is uninformative rather than
negative. With n = 16 only a large effect will clear the permutation test, so a
null is weak evidence of absence — the tool says so in its own output.

### 4.3.9 The coupling test is uninformative — and it pointed somewhere better

`--weight-f-coupling` on `c16` and `jit16`:

| | `wGrowth` median | `wGrowth` range | across-chain r(wGrowth, \|fShift\|) |
|---|---|---|---|
| `c16` | 1.5132 | 1.4716–1.5413 (**±1.1%**) | 0.0367, p = 0.8943 |
| `jit16` | 1.5104 | 1.4770–1.5410 | 0.0173, p = 0.9502 |

**This is the uninformative case, not a negative result** — exactly what the
tool's own caveat says to check. The regressor varies by ~1% across chains
while `fShift` varies by ~88% (−5.9 to +17.3 on `c16`). With essentially no
variation in ‖w‖ growth there is no leverage to correlate against, so the null
means "no power", not "no leak".

**The design was wrong for the question, and that is worth recording.** An
across-chain correlation can only detect a mechanism that *varies* across
chains. The drift we are chasing is COMMON to all chains (§4.3.5), and weight
growth turns out to be common too — 1.51× in every chain, to ±1%. A common
cause of a common effect is invisible to a between-chain test by construction.
The test would only ever have worked if chains happened to differ in growth.

**But the table exposed the real structure.** `fShift` there is the *mean of f
over the diagnostic points* — the global offset. It moves by up to 17.3 reward
units, against a per-point `pred_sd` on the order of 10–16. So most of the
measured drift is a shift in the overall level of `f`.

**And the likelihood is exactly invariant to that shift.** `bt_pool_logit` with
`mode: "mean"` returns `sum(f*mask)/n` (`util.py:356–364`), so a segment's
logit is the mean of `f` over its timesteps; `LikCE` is `CrossEntropyLoss` on
`[Φ₁, Φ₂]`, and softmax depends only on `Φ₁ − Φ₂`. Therefore

> **f → f + c leaves every preference probability unchanged. The data carry no
> information whatsoever about the global offset of f.**

Only the functional GP prior constrains it, weakly, through
`-J^T K^{-1}(f - m)` at 35 measurement points. A chain wandering along the
offset is not a broken sampler — it is exploring a direction the likelihood
does not pin down. Note this is a *consequence of* the `bt_pool: "mean"`
choice recorded in §3.6.2, which was analysed there for length confounds and
cross-family comparability; the offset non-identifiability was not noticed.
The same invariance holds for `bt_pool: "sum"`.

**`--offset-shape-split` measures how much of the drift this accounts for.** It
recomputes the §4.2 gate three ways — raw `f`, centred `f` (each draw minus its
own mean, the identified *shape*), and the offset alone. Validated on synthetic
chains: a pure-offset drift reports 92.2% removed with the centred `loc_z`
passing at 0.80 against raw 10.12; a pure-shape drift reports −0.6% removed; and
a stationary control is caught by a guard that refuses to decompose when raw `f`
already passes, since otherwise the fraction is noise over noise.

```
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_play_c16_0 --offset-shape-split --device cuda \
    2>&1 | tee exp/stage3_medium_play_c16_0_offsplit_diag_tail.txt
```

**What each outcome means.** If centring removes most of the drift and the
shape passes the gate, the stationarity failure is confined to a direction the
data does not identify — preference predictions are unaffected by construction,
and a constant reward offset also leaves the IQL greedy policy unchanged, so
the downstream consequence is small. **The tail statistics are the exception:
CVaR of `f` is offset-sensitive and moves with `c`, so §4.3's whole table would
need recomputing on centred `f` before anything is selected on it.** If the
drift survives centring, it is in the identified part of `f`, the offset
invariance excuses nothing, and this becomes a genuine sampling failure again.

**Do not treat the offset invariance as a licence in advance of the
measurement.** It is a reason the drift *may* be benign, not evidence that it
is; §4.3.5 established the drift is real and common regardless of which
direction it lies in.

### 4.3.10 The split — measured 2026-08-19. The defect is a widening, not a wandering.

`--offset-shape-split` on `c16`:

| | `loc_sd` | `ratio` | `loc_z` | `scale_z` |
|---|---|---|---|---|
| raw `f` | 0.6460 | 1.4664 | **2.5152** FAIL | 1.9962 |
| centred (shape) | 0.3232 | **1.6026** | **1.2566** PASS | **2.5906** FAIL |
| offset only | 0.8949 | 1.3476 | 3.6054 | 1.5144 |

**Location: the drift is largely in the direction the likelihood cannot see.**
Centring halves `loc_sd` (0.6460 → 0.3232) and the centred `loc_z` of 1.2566
**passes** a gate raw `f` fails at 2.5152, while the offset alone carries the
worst location drift of the three at 3.6054. Preference predictions are
unaffected by construction, and a constant reward offset leaves the IQL greedy
policy unchanged.

**Scale: centring made it worse, and that is the real finding.** The centred
`scale_z` is 2.5906 against raw's 1.9962, and the centred ratio 1.6026 against
raw's 1.4664. **The offset drift was partly masking a widening of the
identified shape** — the offset's own ratio (1.3476) is lower than the shape's,
so mixing them diluted the signal. `f → a·f` changes every preference
probability, so scale is *identified*: no invariance excuses this.

**The problem has changed shape.** Nine subsections have chased a *location*
drift; on the identified component the location gate passes, and what remains
is that **the identified part of `f` widens 1.60× between the first and second
halves of sampling.** Unlike `loc_z`, the ratio is an effect size with no
chain-count dependence (§4.2.1), so 1.60× is not a power artifact.

**Three consequences, in order of cost.**

1. **§4.3.2's headline test was run on the wrong quantity.** The 1/√draws
   stationarity test used raw `loc_sd` (0.4319 / 0.4222 / 0.6460), which this
   section shows is about half unidentified offset. It must be redone on
   centred `loc_sd`. `c8` and `c16` both have saved chains, giving two points
   and a 2× draw ratio — enough to test whether the *identified* location drift
   falls as √2 the way stationarity requires. `c4` has no saved chains (§4.3),
   so that rung cannot be recovered.
2. **The §4.3 tail table is contaminated twice over.** CVaR of `f` is sensitive
   to both the offset and the scale. Centring removes the first; the second
   survives centring, so **even a recomputed tail table is an estimate of a
   still-moving target.** No budget should be selected from either version.
3. **The remaining defect is a variance non-stationarity**, which is a
   different problem from everything tried so far. Burn-in, jitter and the
   schedule were all aimed at a location transient. A chain whose spread is
   still growing after 220 000 sampling steps has not equilibrated its
   variance, and §4.3.6 already noted the constant-LR run widened *more*
   (2.13×), so this is not schedule-induced.

**Do this next — no new sampling:**

```
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_play_c8_0 --offset-shape-split --device cuda \
    2>&1 | tee exp/stage3_medium_play_c8_0_offsplit_diag_tail.txt
```

and the same on `stage3_medium_play_jit16_0` and
`stage3_medium_play_nocyc_0`. Then compare **centred** `loc_sd` between `c8`
and `c16` for the 1/√draws test, and **centred** `ratio` across all four for
whether the widening responds to anything tried so far.

**Tool correction.** The verdict logic branched on `frac > 0.5`, and `c16`
lands on exactly 0.500 — so the first run of this test printed "NOT mainly the
offset … a genuine sampling failure" despite the centred location gate passing.
It also ignored scale entirely, which is where the actual defect turned out to
be. It now keys on the gate outcomes, judges location and scale separately, and
reports the fraction as descriptive only. Re-validated on four synthetic cases
including a mixed offset-drift-plus-shape-widening case matching `c16`.

### 4.3.11 The centred ladder — measured 2026-08-19. §4.3.2's headline does not survive.

All four runs, **centred** (identified shape only):

| run | chains | raw `loc_sd` | centred `loc_sd` | centred `ratio` | centred `scale_z` |
|---|---|---|---|---|---|
| `c8` | 8 | 0.4222 | 0.3728 | 1.5913 | 1.8603 PASS |
| `c16` | 16 | 0.6460 | **0.3232** | **1.6026** | 2.5906 FAIL |
| `jit16` | 16 | 0.5714 | 0.2764 | 1.5047 | 2.2526 FAIL |
| `nocyc` | 8 | 0.5443 | 0.4809 | **2.2478** | 2.8559 FAIL |

**§4.3.2's central finding was an artifact of the unidentified offset.** That
section's headline was that `loc_sd` never falls, so the location drift is real
at every rung and does not shrink. On the identified component it *does* fall:

| | `c8` → `c16` | required (1/√2) | obs/req |
|---|---|---|---|
| raw `loc_sd` | 0.4222 → 0.6460 | 0.2985 | **2.16** |
| centred `loc_sd` | 0.3728 → **0.3232** | 0.2636 | **1.23** |

Raw said the drift *grew* when it should have shrunk by √2. Centred, it shrinks
— just not quite fast enough. Fitting `loc_sd² = d² + n²/C` across the two
points gives a common identified location drift of **d ≈ 0.26 sd** against
noise `n ≈ 0.74`; the same fit is impossible on raw `loc_sd` (it would need
negative noise), which is itself a sign that raw is a random walk along the
free offset rather than drift-plus-noise. **Two points, no error bars — treat
d ≈ 0.26 as indicative, not measured.**

**The test was also mis-specified, in a way worth recording.** `num_samples` is
**75 in every run this project has ever done**, so `function_space_drift`
always compares 37 draws against 37 draws *within* each chain. Adding chains
cannot shrink a per-chain drift; it only estimates the common component more
precisely, so `loc_sd` converges to `|d|`, not to zero. A 1/√draws test is only
valid where the drift is pure noise. §4.3.2 was right that stage 3's axis is
orthogonal to the binding constraint — but the reason is this, not the raw
`loc_sd` trajectory it cited.

**Scale is the defect, and here §4.3.2's "low-power false negative" claim is
vindicated.** The centred ratio is an effect size with no chain-count
dependence: `c8` 1.5913 and `c16` 1.6026 differ by **0.7%** — the same
widening — yet `scale_z` reads 1.8603 (PASS) at 8 chains and 2.5906 (FAIL) at
16. That is §4.2.1 in its purest form, and it means **`c8` passed the scale
gate only for want of chains.**

**What moves the widening, and what does not.** Ranking on centred `ratio`:

    jit16 1.5047  <  c8 1.5913  ~  c16 1.6026  <<  nocyc 2.2478

- **The cyclical schedule helps substantially** — removing it takes the
  widening from 1.60× to 2.25×. This corroborates §4.3.6 on the identified
  component, not merely on raw `f`.
- **Jitter helps slightly** (1.5047 vs 1.6026, and centred `loc_sd` 0.2764 vs
  0.3232). **This reverses §4.3.8's recommendation's basis:** that section said
  "leave `chain_init_jitter` at 0" because raw `scale_z` worsened, but raw
  `scale_z` was contaminated by the offset. On the identified component jitter
  is mildly *better* on both axes. It does not fix the widening and the choice
  is not decisive either way, so leaving it at 0 remains defensible — but not
  for the reason given there.
- **Nothing tried so far removes it.** Every configuration widens by ≥1.50×.

**The one axis never varied is `num_samples`.** Every run in this project uses
75 draws per chain, so nothing measured here can say whether the widening is a
chain still equilibrating its variance — which more draws would resolve — or a
genuine instability that more draws would compound. §4.1 pinned `num_samples`
to preserve §3.7's selection/production horizon match, and that pin is exactly
why the question is unanswerable from existing data.

**Next: one diagnostic run at double the draws.**

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --num_samples 150 \
    --OUT_DIR ./exp/stage3_medium_play_d150 > ../exp/stage3_medium_play_d150.log 2>&1 &
```

**This is a DIAGNOSTIC run and must not be used for selection.** It breaks
§3.7's horizon match by construction — that is the point of it — so its CE and
accuracy are not comparable to any selection number, and §3.7 exists precisely
to stop such a run leaking into a winner. Label the output accordingly.

Read it on centred `ratio` against `c16`'s 1.6026. Falling toward 1 means the
variance is still equilibrating and the horizon is simply too short; holding at
~1.6 or rising means an instability that no budget fixes, and the sampler
itself needs revisiting. Note the halves being compared grow with the run
(75 vs 37 draws), so this is not an apples-to-apples ratio — it is the right
comparison anyway, because the question is whether a *longer* chain equilibrates.

### 4.3.12 The `d150` run — measured 2026-08-20. Both budget axes are exhausted.

`num_samples 150`, 16 chains, otherwise identical to `c16`. **Diagnostic only —
it breaks §3.7's horizon match and its CE/accuracy must never feed selection.**

| metric | `c16` (75 draws) | `d150` (150 draws) | |
|---|---|---|---|
| centred `loc_sd` | 0.3232 | 0.3243 | **1.003×** — unchanged |
| centred `ratio` | 1.6026 | 1.5470 | 0.965× |
| centred `scale_z` | 2.5906 | 2.3741 | still **FAIL** |
| raw `ratio` | 1.4664 | 1.5063 | worse |
| raw `scale_z` | 1.9962 | 2.0226 | now **FAIL** too |

§4.3.11 posed the test as: falling toward 1 means the horizon was simply too
short; holding at ~1.6 means an instability no budget fixes. **It held.** A 3.5%
reduction on double the sampling is not equilibration.

**The budget bought nothing at all.** This is the part that settles stage 3:

| | `c16` | `d150` | |
|---|---|---|---|
| `ess_bulk` median | 26.83 | 27.98 | **+4%** on 2× the draws |
| sampling efficiency | 2.24% | 1.17% | **halved** |
| `cvar_ess_median` | 89.42 | 83.59 | **0.935× — went DOWN** |

Doubling the draws per chain produced 4% more effective samples and *fewer*
effective CVaR draws. §4.6's failure mode — ESS flat while the estimand is still
moving — now holds on the **draws** axis as well as the chains axis. Stage 3
raises `num_chains`; §4.1's other lever is `num_samples`; **neither works.**

**Why: the spread grows as a scale-free power law.** From the block `sd/first`
column, fitting `sd(t) ∝ t^α` at block centres 15/45/75/105/135 draws:

| t | `sd/first` | α |
|---|---|---|
| 45 | 1.4337 | 0.328 |
| 75 | 1.8282 | 0.375 |
| 105 | 2.2232 | 0.411 |
| 135 | 2.3256 | 0.384 |

α ≈ 0.37–0.41, against **0.5 for free diffusion**. The half-split ratio implies
α = 0.397 independently. A power law is **scale-invariant**: `sd2/sd1` over
halves does not depend on the horizon, which is exactly why 1.6026 at 37-vs-37
draws became 1.5470 at 75-vs-75. **No draw budget can fix a scale-free
widening** — that is a property of the process, not of the window. The spread
reaches 2.33× its initial value within a single chain's 150 draws.

The identified component is therefore **diffusing, not mixing**, at a rate
slightly sub-diffusive. That is the concrete form of the §3.6.2 hypothesis: the
weight-space diffusion is *not* purely f-preserving; a component of it leaks
into the spread of the identified shape.

**The block location test still cannot resolve, and now we know it never will.**
Max block step is 0.89× the noise floor even at 150 draws, because the floor
depends on **ESS, not draws** — and ESS did not grow. The location shifts do
show a suggestive decay (0.345, 0.358, 0.137, 0.147), and the last `sd/first`
increment is much smaller than its predecessors (+0.102 against ~+0.40), which
could be the beginnings of saturation. Both are below resolution. **Do not
build on either**; recording them only so a future reading is not mistaken for
confirmation.

**Location remains sound throughout.** Centred `loc_z` is 1.2344 here, 1.0843 at
`jit16`, 1.0609 at `c8` — every configuration passes. Combined with §4.3.9,
`E[f]` is stationary and the drift that fails the gate is the likelihood-
invariant offset. **The defect is confined to the spread.**

### 4.3.13 Stage 3 terminates without a budget — the decision this forces

Both levers §4.1 gives stage 3 are now measured and neither resolves the tail:
chains (§4.3.2, §4.3.11) and draws (§4.3.12). Selecting `num_chains` on tail
statistics is therefore not possible, and continuing to ladder is not a
research plan. **Stage 3 should be closed as a negative result and the budget
set on cost plus the §4.2 gate.**

What that costs depends on one thing this document cannot settle:

- **If the downstream quantity is `E[f]`** — the reward mean that IQL's greedy
  policy uses — the position is strong. Centred location passes at every
  configuration, the offset is invariant to both the BT likelihood and the
  greedy policy, and the widening affects only the uncertainty. Select on cost,
  disclose the widening as a limitation on interval estimates, and proceed.
- **If the downstream quantity is CVaR** — which `diagnose_sampling_tail.py`
  calls "the downstream quantity" throughout, and which §4 has been steering on
  — the position is weak. CVaR is a functional of the lower tail, the tail is
  exactly what the widening inflates, and `cvar_ess_median` *fell* when the
  budget doubled. A CVaR-based claim rests on a spread that is still growing at
  the end of every run measured.

**Resolve which of these the paper actually claims before setting the budget.**
If it is CVaR, the honest options are to fix the sampler (a research task, not
an HP-selection task) or to restate the claim in terms of `E[f]`. Do not select
a `num_chains` and let the CVaR numbers inherit an unstated caveat.

**For the write-up**, this is a substantive negative result and belongs in the
paper: composing a cyclical step-size schedule with fSGHMC at this scale yields
a sampler whose identified-component spread grows as `t^0.4` over the full
sampling horizon, so its uncertainty estimates do not converge in either the
chain or the draw budget, while its posterior *mean* is stationary. §7.1's
existing lesson — diagnostics can be actively misleading — extends: here R-hat,
ESS and the drift z-gate were each individually reassuring at some
configuration, and only the raw effect size on the centred component exposed it.

### 4.3.14 Root cause — the selection objective drives the prior to improperness

**Decision recorded 2026-08-20.** §4.3.13's fork is resolved: the paper's claim
is conservative reward prediction via **CVaR**, hypothesising that induced
conservatism preserves policy performance under reduced data and increased label
noise better than a non-Bayesian baseline. There is no theory that the posterior
*mean* would beat a non-Bayesian method under those conditions — if there were,
the experiments in question would be unnecessary. **The mean-based fallback is
not available, so the sampler must be fixed.** §4.3.12's widening is therefore
load-bearing, not a disclosed limitation.

**The sampler is not the root cause. The selected target is.** The four stage-1
winners:

| variant | `n_meas` (range 0–64, default 256) | `map_amp2` (range 1–1e6) | % of log range |
|---|---|---|---|
| medium_play | 35 | 168 940 | 87% |
| medium_diverse | 17 | 119 681 | 85% |
| large_play | 29 | **925 895** | **99.5%** |
| large_diverse | **7** | 94 945 | 83% |

Both knobs govern how tightly the functional prior constrains `f`, and **the
sweep pushed both toward removing it**: amplitude near the top of its range,
measurement points near the bottom of theirs (`n_meas: 0` disables the
functional prior outright, and large_diverse selected 7).

**`map_amp2` has no interior optimum under CE — it chases the cap.** Across
three rounds, the winner has sat at 83–99% of the log range every time, and each
cap expansion moved it:

| round | cap | winners | % of log range |
|---|---|---|---|
| 1 | 1e3 | 313–773 | 83–96% |
| 2 | 1e4 | 6 647, 8 699 | 96–98% |
| 3 | 1e6 | 94 945–925 895 | 83–99.5% |

The sweep yaml raised the cap to 1e6 specifically so the search would not be
"boundary-limited a third time". It was anyway — large_play sits at 99.5%.
**Validation CE improves monotonically as the prior flattens, so there is no
value of `map_amp2` the objective will settle on.** A flatter prior fits better
at 75 draws; the cost only appears at horizons the selection never reaches. This
is §3.7's pathology exactly, on a parameter §3.7 was not written about.

**Why this makes sampling intractable.** `map_amp2` 168 940 puts the prior
reward sd at √168 940 ≈ **411×** the base kernel scale, against observed
`pred_sd` of ~10–20. Along directions the preference data identifies weakly, the
posterior inherits close to that prior width, and the chain must diffuse across
it. At the measured `t^0.4`, growing from sd ~15 to ~100 needs **115×** the
current horizon; to 411, **~3 900×** — 590 000 draws per chain. Those are
ceilings, not forecasts, since the likelihood does constrain identified
directions and the true equilibration target is unknown. But the order of
magnitude is the point: **the chain is nowhere near equilibrium and no feasible
budget gets it there.** That is why §4.3.12 found both budget axes exhausted,
why ESS is ~28 regardless of draws, and why the location (well-identified)
passes while the spread does not.

**Why §3.6.3's eligibility gate did not catch it.** It did exactly what it was
written to do, at 4 chains × 75 draws. §4.2.1 shows the drift z-scores have
power growing as √(chains), so at `c4` the gate cannot resolve a drift that
`c16` reports at `scale_z` 2.59. The gate was passing configurations that drift,
and doing so *systematically* in favour of the flattest priors, because those
score best on CE. Selection and eligibility were pulling the same direction.

**The plan.** This is a re-selection problem, not a sampler-engineering one, and
it interacts with the `bt_pool` decision already queued:

1. **Do not sweep `map_amp2` on CE.** The objective has no interior optimum in
   it. Either fix it at the logit-matching value the sweep yaml derives
   (~1e4 under mean pooling) or, better, resolve `bt_pool` first: under `"sum"`
   the natural amplitude is ~1 and the enormous multiplier is unnecessary. The
   two are the same model up to a global temperature (§3.6.2), but *not* the
   same weight-space geometry at a fixed step size, so the choice affects
   sampling even where it does not affect the model.
2. **Raise the `n_meas` floor.** A range whose lower end disables the prior is
   the wrong range for a method whose contribution *is* the prior. The default
   is 256; the sweep caps at 64 and winners sit at 7–35.
3. **Re-specify eligibility on centred effect sizes with power.** Gate on
   centred `ratio` and centred `loc_sd` (§4.3.10–§4.3.11), not raw `z` at 4
   chains. Raw `z` is contaminated by the free offset and low-powered at `c4` —
   the two failure modes this document spent nine subsections isolating.
4. **Then re-run stage 1**, and only then stage 3.

**For the write-up**, this is a stronger result than §4.3.13's. The headline is
not "our sampler mixes poorly" but: **selecting a functional-prior BNN on
validation CE at a short sampling horizon drives the prior toward improperness,
which destroys the calibrated uncertainty the method exists to provide, while
every standard diagnostic looks acceptable.** §7.1 already carries the
short-horizon lesson for step size; this extends it to the prior itself, and it
is the sharper instance.

### 4.3.15 The `bt_pool="sum"` run — stationarity can be bought, and it was

Ran `bt_pool sum` with `map_amp2` rescaled 168939.82 → 16.894 (rewards go to
1/100 under sum, so sd ×1/100, variance ×1/1e4), 16 chains, everything else as
`c16`:

| metric | `c16` (mean) | `btsum` (sum) | |
|---|---|---|---|
| raw `loc_z` / `scale_z` | 2.5152 / 1.9962 | **0.3740 / 1.4966** | both now PASS |
| raw `loc_sd` | 0.6460 | **0.1075** | 6× better |
| centred `ratio` | 1.6026 | **1.0661** | **the widening is gone** |
| centred `scale_z` | 2.5906 | **0.8412** | passes comfortably |
| **predictive CE** | **0.2076** | **0.4158** | **2× WORSE** |
| predictive accuracy | 0.9148 | 0.9002 | worse |
| `gradnorm_burnin_mean` | 0.2034 | 10.8715 | **53×** |
| `gradnorm_sampling_mean` | 0.2098 | 3.6421 | **17×** |

**The rescaling was correct and the equivalence holds at the model level.** The
map-informed prior is **zero-mean** — `map_informed_prior.py:4`, "informativeness
lives entirely in its kernel (zero mean)", confirmed at lines 244 and 293 — so
there is no prior mean needing a matching rescale, and `amp2` multiplies the
whole kernel including jitter (`:163–176`), giving `K → K/1e4` exactly. CE is a
function of `Φ₁ − Φ₂`, which the reparameterisation leaves invariant. **A
correct equivalence must therefore give identical CE. It gave 2× worse.**

**The sampler is not scale-invariant, and that is the whole explanation.**
`sghmc_lr`, `mdecay`, `max_param_step` and `clip_grad_norm_value` are all fixed
constants in *weight* space. Under sum-pooling `∂Φ/∂r_t` is 1 rather than
`1/T`, so gradients grow ~100× — measured 53× in burn-in, 17× in sampling — and
with `sghmc_lr` unchanged the effective step grows with them. Neither the
`max_param_step` clamp nor the gradient clip fired in either run
(`param_clamp_sampling_pct` 0, `gradnorm_*_pct_over_clip` 0), so nothing
absorbed it. The sampler simply took ~50× larger steps through the same target.

**Which is why every stationarity number improved while the answer got worse.**
A step size that large equilibrates fast to a measure that is not the posterior:
the discretisation error is O(ε), so the chain settles quickly onto something
too broad and stays there. `function_space_drift` asks whether the sampled
measure is *stationary*, not whether it is the *right* measure — and a fast,
stationary, wrong chain passes every gate in §4.2.

> **This is the round-1 failure mode inverted, and it is the most important
> methodological result in §4.3.** §7.1 records diagnostics looking *good*
> because a chain was drifting. Here they look good because a chain is moving
> too fast. In both cases the diagnostic is measuring something real and the
> inference from it is wrong. **Stationarity is necessary, not sufficient**, and
> centred `ratio` alone is therefore *not* a valid objective for a sampler fix:
> it can be driven to 1.0661 by breaking the sampler. Judge any candidate fix
> **jointly on centred `ratio` and predictive CE**, and treat a fix that improves
> drift while degrading CE as having made things worse, not better.

**Decision: keep `bt_pool="mean"` in all three families (option (a)).** The
comparability objection stands on its own — `bt_pool` is shared across
`scripts_mr/`, `scripts_pt/` and BNN, and §3.6.2 rests cross-family
comparability on it being identical — and the run adds a second, independent
reason: the mean-pooled parameterisation is where the tuned sampler
hyperparameters are actually valid. A fair sum-pooled comparison would need
`sghmc_lr` re-tuned by roughly the same factor, which at best re-derives the
posterior we already have. **Nothing is to be gained by switching, and §3.6.2's
equivalence claim survives** — it is a statement about the model, and the model
behaved exactly as claimed.

**`map_amp2` is still untested.** `btsum` used the *same* prior amplitude
expressed in different units, not a smaller one, so §4.3.14's root-cause
hypothesis — that selection on CE drove the amplitude to 1.69e5 and an
improper-in-practice prior — has had no experiment aimed at it yet. That is the
next run, and it should be **mean-pooled**:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 8 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --OUT_DIR ./exp/stage3_medium_play_amp1e4 > ../exp/stage3_medium_play_amp1e4.log 2>&1 &
```

**Eight chains is the right size here**, and this is not the §4.3.7 mistake
repeated: centred `ratio` is an *effect size* with no chain-count dependence,
and `c8` 1.5913 against `c16` 1.6026 measures its run-to-run stability at
**0.7%**. What §4.3.7 showed to be unresolvable at 8 chains was ALIGNMENT and
pooled `loc_sd`, neither of which is the objective here. That halves the cost
per rung.

Judge each rung on **centred `ratio` AND CE together**, per the box above. The
outcome that would confirm §4.3.14 is `ratio` falling toward 1 while CE stays
near 0.2076; `ratio` falling while CE degrades is the `btsum` failure again and
means the amplitude is not the mechanism.

### 4.3.16 The amplitude ladder — §4.3.14 confirmed, and it is a real lever

`map_amp2` 168939.8 → 16894.0 (10× down, to the principled scale), 8 chains,
mean-pooled, everything else as `c8`:

| metric | `c8` (1.69e5) | `amp1e4` (1.69e4) | |
|---|---|---|---|
| **centred `ratio`** | **1.5913** | **1.3200** | **46% of the excess removed** |
| centred `scale_z` | 1.8603 | 1.1812 | large drop |
| centred `loc_sd` | 0.3728 | 0.3173 | better |
| centred `loc_z` | 1.0609 | 0.9576 | better |
| offset `ratio` | 1.2903 | **1.4677** | **rose** |
| raw `ratio` | 1.4361 | 1.4422 | flat — the two cancel |
| raw `loc_sd` | 0.4222 | 0.1744 | 2.4× better |
| predictive CE | 0.2029 | 0.2232 | +10% |
| accuracy | 0.9172 | 0.9107 | −0.7 pt |
| `gradnorm_sampling_mean` | 0.2252 | 0.1856 | no blow-up |

**This is not the `btsum` failure mode.** Gradient norms *fell*, the clamp never
fired, and CE cost 10% rather than doubling. The stationarity gain is bought
with a better-specified prior, not with a broken sampler — which is exactly the
distinction §4.3.15 says to test for.

**Raw `ratio` is not a valid read-out for this question.** It moved 1.4361 →
1.4422 and reads as "no change", while the centred shape improved 17% and the
offset *worsened* — the two cancel inside the mixture. Any amplitude comparison
must be made on the split; the raw number will hide the effect. (This caught me
out: the wandb-only preliminary read of this run concluded the widening was
untouched, and it was wrong.)

**§4.3.14's root cause is confirmed, and the sweep yaml documents the mechanism
itself.** The principled amplitude is derivable, not empirical: every segment is
exactly T=100 timesteps, so mean pooling divides the BT logit by 100, matching
that logit scale needs rewards 100× larger, and since `map_amp2` scales the
kernel (sd as `sqrt(map_amp2)`) the natural mean-pooled amplitude is ~100² =
**1e4** (`sweep_antmaze_medium_play_bnn_antmaze_eval.yaml:149–157`). The
selected winner sits **16.9× above** it. And the same comment records the winner
chasing every cap it was given:

| cap | winner(s) |
|---|---|
| 1e3 (round 1) | 313 – 773 |
| 1e4 (round 2, early) | 6647 / 8699 |
| 1e6 (round 2, final) | **168940** |

**CE has no interior optimum in `map_amp2`.** Three caps, three winners at or
near the boundary, each an order of magnitude apart. That is the improper-prior
limit reached numerically, and it is direct evidence for §4.3.14 sitting in the
repo's own configuration files — the cap was raised twice *because* the winner
kept hitting it, which is the selection objective doing exactly what §4.3.14
says it does.

**But the widening is reduced, not eliminated.** 1.3200 is not 1.0, and at the
principled amplitude there is still a 32% spread growth between chain halves.
Amplitude is *a* mechanism, not the only one. Whether it can reach 1.0 at all
is the next question, and it needs one more rung **below** the principled value
— which is a diagnostic, not a candidate configuration:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 8 --chains_per_gpu 4 --map_amp2 1689.3982289052463 \
    --OUT_DIR ./exp/stage3_medium_play_amp1e3 > ../exp/stage3_medium_play_amp1e3.log 2>&1 &
```

Read it on centred `ratio` and CE jointly (§4.3.15), and on the split, never on
raw `ratio`. A continued fall toward 1.0 means amplitude is the dominant
mechanism and the question becomes where to stop on the CE/stationarity curve —
a question that must be answered on calibration grounds, since §4.3.14 is
precisely that CE cannot answer it. A plateau near 1.3 means a second mechanism
holds the floor, and the residual is what the sampler repair has to target.

**Selection implication, independent of that rung.** `map_amp2` must stop being
swept on validation CE. The principled value is derivable from the pooling
convention and the segment length; fix it there, and let the sweep spend its
budget on parameters that have interior optima.

### 4.3.17 The ladder completed — amplitude plateaus, and the tail improves

Three rungs, 8 chains each, mean-pooled, everything else as `c8`:

| `map_amp2` | centred `ratio` | excess | centred `scale_z` | CE | acc | `cvar_ess` | relMCSE med | relMCSE max | unresolved |
|---|---|---|---|---|---|---|---|---|---|
| 1.69e5 (selected) | 1.5913 | 0.5913 | 1.8603 | 0.2029 | 0.9172 | 59.21 | 0.2706 | 2.3862 | **2.25%** |
| 1.69e4 (**principled**) | 1.3200 | 0.3200 | 1.1812 | 0.2232 | 0.9107 | 46.78 | 0.3064 | 1.9029 | **1.44%** |
| 1.69e3 (diagnostic) | 1.2308 | 0.2308 | 0.9483 | 0.2792 | 0.8864 | 31.03 | 0.3406 | 1.4390 | **0.25%** |

**Amplitude plateaus short of stationarity.** Successive decades remove 0.2713
then 0.0892 of the excess widening — gains shrinking 0.329× per decade. If that
holds, the remaining excess is ~0.044 and the floor is **centred `ratio` ≈ 1.19,
not 1.0**. Three points and a geometric extrapolation, so indicative rather than
measured, but the direction is unambiguous: **a second mechanism holds the
floor, and amplitude cannot reach stationarity on its own.** This is §4.3.16's
plateau branch, and that residual is what the sampler repair must target.

**The returns diminish sharply, which fixes the stopping point.** Excess removed
per unit of CE paid: **13.36** in the first decade, **1.59** in the second — an
8.4× collapse in efficiency. Combined with the derivation, that makes the
principled ~1e4 the defensible stopping point: it captures 46% of the excess for
10% CE, and everything past it costs roughly eight times more per unit gained.

> **Correction to the `cvar_ess` reading.** The falling `cvar_ess` median (59.21
> → 46.78 → 31.03) was flagged as a degradation of the paper's core quantity.
> That was wrong. The metric that decides whether a point's CVaR is *usable* is
> whether its MC error exceeds the posterior sd, and unresolved points fall
> **2.25% → 1.44% → 0.25%, a 9× improvement**, while relMCSE *max* falls 2.3862
> → 1.4390. Only the *median* worsens (0.2706 → 0.3406). So the difficulty
> distribution compresses: the typical point gets slightly noisier while the
> points that were unusable become usable. For a method whose claim is
> per-point conservative reward estimates, that is the right direction, and it
> is the opposite of the conclusion drawn from the median alone.
>
> Part of the median move is likely §7.1's inflation unwinding — a drifting
> chain inflates ESS, so the higher `cvar_ess` at 1.69e5 was partly an artifact
> of the drift being removed. These runs cannot separate that from a genuine
> efficiency cost, and no claim either way should be made from them.

**Checkable prediction.** The `c8` → `c16` centred `scale_z` factor was measured
at 1.393 (1.8603 → 2.5906), close to the √2 §4.2.1 predicts. Applying it:

| `map_amp2` | 8-chain | 16-chain (est.) | |
|---|---|---|---|
| 1.69e5 | 1.8603 | 2.5906 | FAIL (measured) |
| 1.69e4 | 1.1812 | **1.6449** | **PASS** |
| 1.69e3 | 0.9483 | 1.3206 | PASS |

**At the principled amplitude the run is predicted to pass §4.2's scale gate at
16 chains** — the first configuration in this investigation that would. Verify
it rather than assume it; §4.2.1 exists because chain-count extrapolations of
z-scores are exactly what goes wrong here.

**Recommendation.** Fix `map_amp2` at the principled value and remove it from
the sweep. It has no interior optimum under CE (§4.3.16's cap history), it is
derivable from the pooling convention and segment length, and the sweep budget
it consumes is better spent on parameters that CE can actually select. Then
re-run the 16-chain confirmation above before anything downstream depends on it.

### 4.3.18 The 16-chain confirmation — gates pass, and the common drift is gone

`map_amp2` 16894.0 at 16 chains, against `c16` (1.69e5, 16 chains):

| metric | `c16` | `amp1e4_c16` | |
|---|---|---|---|
| raw `loc_z` / `scale_z` | 2.5152 **FAIL** / 1.9962 | **1.2801 / 1.9308** | both PASS |
| centred `loc_z` / `scale_z` | 1.2566 / 2.5906 **FAIL** | **1.0251 / 1.7532** | both PASS |
| centred `ratio` | 1.6026 | **1.3490** | |
| **ALIGNMENT** | **0.7564** | **0.4593** | |
| signed shift | 14/16, p = 0.0042 | **10/16, p = 0.4545** | |
| unresolved | 0.66% | **0.17%** | 3.9× better |
| relMCSE max | 1.6749 | **1.2697** | better |
| relMCSE median | 0.2234 | 0.2437 | 9% worse |
| CE | 0.2076 | 0.2232 | 7.5% worse |

**Both §4.3.17 predictions hold.** Centred `scale_z` came in at **1.7532**
against a predicted 1.6449 — 6.6% off, and on the right side of the gate. And
the effect-size stability that justified running the ladder at 8 chains holds:
centred `ratio` 1.3200 (8ch) vs 1.3490 (16ch), **2.2% apart**. Slightly looser
than the 0.7% seen at 1.69e5 but well inside what the ladder's conclusions rest
on. §4.3.16's 8-chain methodology is sound.

**The common drift is gone.** ALIGNMENT collapses 0.7564 → **0.4593** and the
sign test goes from p = 0.0042 to p = 0.4545 — from "every chain carries the
same shift" to no detectable common direction, both measured at 16 chains so
directly comparable. **§4.3.5's common drift was largely an artifact of the
excessive amplitude**, i.e. of the free offset wandering under a prior too weak
to pin it. Nine subsections chased a shared cause — start, schedule, GPU
placement — for something that was mostly a mis-specified prior.

**This changes what the residual is.** The widening that remains
(`ratio` 1.3490) is *not* shared across chains: each chain's variance grows
independently. Shared-cause hypotheses are therefore the wrong place to look
for it. `chain_init_jitter`, the cyclical schedule and burn-in length all act
on mechanisms common to every chain, and none of them can explain a per-chain
independent variance growth. **The sampler repair should target within-chain
equilibration** — the chain's own variance still growing over its 75 draws —
which is the `num_samples` axis §4.3.12 found exhausted at 150 draws and the
step-size/friction geometry that §4.3.15 showed the sampler is sensitive to.

> **The gate PASS here is itself power-dependent — do not read it as
> stationarity.** §4.2.1 cuts both ways: `scale_z` grows with chain count, so
> passing at 16 says nothing about 32. Applying the measured 1.393 factor,
> centred `scale_z` at 32 chains projects to **2.44 — a FAIL.** The effect size
> is what is invariant, and it is **1.3490**, a 35% widening between chain
> halves. That is a real non-stationarity that this configuration has reduced,
> not removed, exactly as §4.3.17's plateau at ~1.19 predicts. A configuration
> that passes §4.2 at the chain count you happen to run is not the same as a
> sampler that is stationary.

**Tail behaviour repeats §4.3.17's pattern and is the best seen so far.**
Unresolved points 0.66% → **0.17%** (3.9× better, and the lowest of any run in
this investigation), relMCSE max 1.6749 → 1.2697, against a 9% worse median.
The difficulty distribution compresses again: typical points slightly noisier,
unusable points become usable.

**Status of this configuration.** At the principled amplitude and 16 chains,
medium_play passes every §4.2 gate raw and centred, resolves 99.83% of points,
and costs 7.5% CE against the CE-selected amplitude. It is the best-behaved
configuration produced in this investigation and the first to clear the gate.
It is **not** stationary, and §7.1's disclosure list should say so plainly
rather than resting on the gate result.

### 4.3.19 The residual — `n_meas` is a second lever, and both plateau together

Two runs at the principled amplitude, 8 chains, against the `amp1e4` baseline:

| run | `n_meas` | draws | centred `ratio` | centred `scale_z` | CE | acc | `cvar_ess` | unresolved |
|---|---|---|---|---|---|---|---|---|
| `amp1e4` | 35 | 75 | 1.3200 | 1.1812 | 0.2232 | 0.9107 | 46.78 | 1.44% |
| `nmeas256` | **256** | 75 | **1.2297** | 0.9372 | 0.2865 | 0.8831 | 27.39 | **0.28%** |
| `d150` (first 75) | 35 | 75 | 1.3200 | 1.1812 | — | — | — | 1.44% |
| `d150` (full) | 35 | **150** | 1.2975 | 1.1114 | 0.2134 | 0.9107 | **17.19** | **42.09%** |

`d150`'s first 75 draws reproduce `amp1e4` **digit-for-digit** — same centred
`ratio`, `loc_sd`, `scale_z` and unresolved count — so the two are properly
nested and every comparison below is controlled.

**Longer chains do not equilibrate the variance.** Under `sd(t) ∝ t^α` a
half-split ratio is `3^α`, **independent of window length**, so a saturating
variance must show a *falling* ratio as the window grows. Doubling the window
moved it only −1.7% (1.3200 → 1.2975; α = 0.2527 → 0.2371). The growth is
essentially scale-free: a power law that does not saturate. **§4.3.12's
conclusion survives correction of the amplitude** — this was worth re-testing,
because it was originally measured under a mis-specified prior, but the answer
is unchanged.

> **The non-stationarity actively destroys CVaR, and more draws makes it
> worse.** `d150` doubles the draws and unresolved points go **1.44% → 42.09%**
> while `cvar_ess` *falls* 46.78 → 17.19. Pooling draws from a chain whose
> variance is still growing mixes narrow early draws with wide late ones, and
> the between-draw heterogeneity inflates the MCSE faster than the extra draws
> reduce it. This is the concrete cost of the widening to the quantity the
> paper's claim rests on, and it means **the tail cannot be bought with
> budget**: spending more draws on this sampler makes the CVaR estimate worse,
> not better. It also retires any lingering idea that the widening is a
> cosmetic diagnostic complaint.

**`n_meas` is confirmed as a second lever, as §4.3.19's hypothesis predicted.**
256 measurement points cut the excess widening 0.3200 → 0.2297 (28% of what
remained), took centred `scale_z` to 0.9372, and improved unresolved points
**5× to 0.28%** — the best in the investigation. The cost is real: CE +28%
(0.2232 → 0.2865), accuracy −2.8 pt, `cvar_ess` −41%. No sampler pathology
(gradient norms flat, clamp and clip at zero), so this is a prior-strength
effect, not a §4.3.15 artifact.

**Both prior-strength levers plateau on the same floor.** Independently:

| configuration | centred `ratio` |
|---|---|
| `map_amp2` 1.69e3, `n_meas` 35 | 1.2308 |
| `map_amp2` 1.69e4, `n_meas` 256 | **1.2297** |
| §4.3.17 projected plateau | ~1.19 |

Two different knobs, pushed hard in opposite parameterisations, landing within
0.1% of each other and just above the projected floor. Both are strengthening
the same thing — the functional prior's grip on `f` — and they saturate
together. **The residual below ~1.23 is a third mechanism that prior strength
cannot reach.**

**What that third mechanism is likely to be.** With the common drift gone
(§4.3.18) and prior strength exhausted, what remains is per-chain,
scale-free variance growth at α ≈ 0.24 — the signature of the sampler's own
discretisation rather than of the target. §4.3.15 already demonstrated this
sampler is acutely sensitive to effective step size: a ~50× step increase
passed every stationarity gate while doubling CE. The untested geometry knobs
are `sghmc_lr` and `mdecay` (friction), neither of which has been varied at the
corrected amplitude, and both of which were selected by the same CE objective
that mis-set `map_amp2` and, on this evidence, `n_meas`.

**Do not read ALIGNMENT on these runs.** They are 8-chain, and §4.3.7 measured
the 8-chain ALIGNMENT spread within a single run at 0.41. The values here
(0.0864, 0.1581, 0.2538) are inside that noise and carry no information.

**Implication for the sweep redesign.** `n_meas` now looks like a second
instance of §4.3.14's pathology: the codebase default is 256, the sweep caps it
at 64, round-1 winners were 10–35 and round 2 chose 35 — CE consistently
prefers the weakest prior it is offered. If that reading holds, **both
`map_amp2` and `n_meas` should be fixed on principled grounds and removed from
the sweep**, which frees a substantial share of the 130-run budget. But settle
the geometry first: `sghmc_lr` and `mdecay` are swept, and if either turns out
to be mis-set by the same mechanism, the re-sweep would need doing a third time.

### 4.3.20 Friction goes the wrong way — and that identifies the mechanism

`mdecay` 0.1946 → 0.6 at the principled amplitude, 8 chains:

| metric | `amp1e4` | `mdecay06` | |
|---|---|---|---|
| **centred `ratio`** | **1.3200** | **1.5150** | **worse** — excess +61% |
| centred `scale_z` | 1.1812 | 1.6293 | worse |
| centred `loc_sd` | 0.3173 | 0.4063 | worse |
| CE | 0.2232 | 0.2710 | 21% worse |
| accuracy | 0.9107 | 0.8880 | −2.3 pt |
| unresolved | 1.44% | **0.84%** | **better** |
| relMCSE median | 0.3064 | **0.2451** | **better** |
| `cvar_ess` | 46.78 | **53.54** | **better** |
| `ess_bulk` median | — | 14.43 | unchanged vs `c8` 14.57 |

**Raising friction was the wrong direction, and the sampler code says why.**
`adaptive_sghmc.py:147` sets

    epsilon_var = 2 * lr² * mdecay * minv_t - lr⁴

so **the injected noise variance is proportional to `mdecay`**. Tripling the
friction tripled the thermal noise (3.08×). The `-lr⁴` term is the
gradient-noise correction; `v_hat` enters only through the preconditioner
`minv_t`, not as a subtraction from the injected noise. More friction here does
not mean more damping — it means a hotter chain.

**This makes every result in §4.3.16–19 one mechanism.** Rank the
configurations by how much noise reaches the chain:

| configuration | noise | centred `ratio` | α |
|---|---|---|---|
| `n_meas` 256 — least prior-gradient noise | low | 1.2297 | 0.188 |
| `map_amp2` 1.69e3 — tightest prior | low | 1.2308 | 0.189 |
| baseline (`amp1e4`) | mid | 1.3200 | 0.253 |
| `mdecay` 0.6 — 3.1× injected noise | high | 1.5150 | 0.378 |
| `map_amp2` 1.69e5 — loosest prior | high | 1.5913 | 0.423 |

Monotone in noise, across three unrelated knobs. **The proposed mechanism: the
chain's stationary variance is inflated by noise the `-lr⁴` correction does not
fully absorb, and mixing is far too slow (`ess_bulk` ~14 of 600 draws, 2.4%) for
the chain to reach that inflated target inside 75 draws. The observed widening
is the climb toward it.** That explains why the growth is scale-free and does
not saturate (§4.3.19) — the chain is nowhere near its stationary variance at
any point in the run — and why every noise-reducing knob helps while the one
noise-raising knob hurts.

**Sharp prediction, and it is cheap to test: LOWER `mdecay` should reduce the
widening.** If the mechanism is right, halving the injected noise should move
centred `ratio` below 1.3200, and possibly below the ~1.23 floor that prior
strength alone could not pass — because friction reduces noise by a route
independent of the prior:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 8 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --mdecay 0.08 --OUT_DIR ./exp/stage3_medium_play_mdecay008 > ../exp/stage3_medium_play_mdecay008.log 2>&1 &
```

`epsilon_var` stays positive with enormous margin (`lr²` ≈ 6.2e-8, so the
`-lr⁴` term is ~1e-15), so 0.08 is numerically safe. Watch `ess_bulk` and
`gradnorm`: less friction is more underdamped, so mixing could degrade even as
stationarity improves.

**A useful dissociation, worth keeping whatever the mechanism turns out to be.**
Friction *improved* every tail metric — unresolved 1.44% → 0.84%, relMCSE
median 0.3064 → 0.2451, `cvar_ess` 46.78 → 53.54 — while making stationarity
worse. Higher friction damps per-draw jitter (lower MCSE) while raising the
target variance (worse widening). **Tail precision and stationarity are
therefore separately controllable**, which means the final configuration need
not trade them off against each other the way §4.3.17 assumed. If low `mdecay`
fixes the widening, the tail cost might be recoverable elsewhere.

### 4.3.21 Low friction does nothing — the asymmetry localises the noise source

> **MECHANISM RETRACTED — see §4.3.26.** The `-lr⁴` term is not inert: it is
> O(ε⁴) because the gradient-noise contribution it cancels is also O(ε⁴),
> against an O(ε²) thermostat. Gradient noise is ~1e-7 of the thermostat and
> is correctly cancelled. The asymmetry below has a simpler explanation:
> in exact SGHMC the stationary distribution is **independent of C**, so
> `mdecay`↓ producing no change is correct behaviour, and `mdecay`↑ acts
> only through discretisation error. **The measurements stand; the
> attributed mechanism does not.**

`mdecay` 0.08 at the principled amplitude, 8 chains:

| `mdecay` | injected noise | centred `ratio` | α | CE | unresolved | relMCSE med | `cvar_ess` |
|---|---|---|---|---|---|---|---|
| 0.08 | **0.41×** | **1.3230** | 0.2548 | **0.2160** | 2.39% | 0.4693 | 26.00 |
| 0.1946 (baseline) | 1.00× | 1.3200 | 0.2527 | 0.2232 | 1.44% | 0.3064 | 46.78 |
| 0.6 | 3.08× | 1.5150 | 0.3781 | 0.2710 | **0.84%** | **0.2451** | **53.54** |

**§4.3.20's prediction failed.** Halving the injected noise moved centred
`ratio` by +0.2% — nothing. And `ess_bulk` median is 14.81 against `c8`'s 14.57,
so mixing did **not** collapse: the flat result is genuine, not a chain moving
too little to reveal its own growth.

**But the asymmetry rescues the mechanism in sharper form.** The response is not
monotone in injected noise — **flat below the baseline, steeply worse above it**
(−59% noise → +0.2%; +208% noise → +14.8%). That is the signature of a
*saturating* sum: at `mdecay` ≤ 0.1946 the injected thermal noise is **not the
binding source**, so cutting it further changes nothing; tripling it makes it
binding, and the widening follows. **The dominant noise at baseline is
gradient noise, not the thermostat.**

That is exactly why the two knobs that worked, worked. `n_meas` 35 → 256 and
`map_amp2` 1.69e5 → 1.69e4 both reduce *gradient* noise — the first by
averaging the functional-prior gradient over more measurement points, the second
by shrinking the prior gradient's magnitude. `mdecay` down reduces only the
thermostat, which was never binding. §4.3.20's ranking was right about noise
driving the widening and wrong about which noise.

**The prediction this makes, and the knob is untouched.** `batch_size` is
**64** in every run in this investigation, against a codebase default of **256**
(`run_bnn_training_antmaze_eval.py:122`) — and it is **not in the sweep**, so it
was never selected by anything. It is the purest available gradient-noise knob:
4× the batch cuts the minibatch gradient-noise variance 4× with **no
prior-strength side effect**, unlike `n_meas` and `map_amp2`, which buy noise
reduction by stiffening the prior and cost CE for it.

**If minibatch noise is what sets the ~1.23 floor, `batch_size` is what passes
it** — and it is the one lever so far that should not cost CE:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 8 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --batch_size 256 --OUT_DIR ./exp/stage3_medium_play_bs256 > ../exp/stage3_medium_play_bs256.log 2>&1 &
```

**A pattern worth naming.** Three parameters now sit far below their codebase
defaults, all in the noise-increasing direction: `n_meas` 35 vs 256,
`batch_size` 64 vs 256, and `map_amp2` 17× above its principled value (a looser
prior, same direction). Two were driven there by CE selection (§4.3.14); the
third was never selected at all. **The sweep redesign should treat "is this
value below its default, and does raising it reduce chain noise?" as a
first-class check**, not just re-run the search.

**`mdecay` 0.08 is not a configuration to adopt.** It buys the best CE of any
principled-amplitude run (0.2160) and no stationarity gain, while costing the
tail badly: unresolved 1.44% → 2.39%, relMCSE median 0.3064 → 0.4693,
`cvar_ess` 46.78 → 26.00.

> **The single-knob search has reached its limit, and the tail objective is not
> monotone in any one lever.** `mdecay` moves stationarity and tail efficiency
> in *opposite* directions (0.6: best tail, worst widening; 0.08: worst tail, no
> widening gain), while `n_meas` 256 improves unresolved points to 0.28% *and*
> the widening, yet degrades `cvar_ess` and CE. No single knob dominates, and
> the interactions are the point rather than a nuisance. **Once `map_amp2` and
> `n_meas` are fixed on principled grounds, the remaining parameters should be
> re-swept jointly against a tail-aware objective** — not validation CE, which
> §4.3.14 showed selects for a weak prior and which no longer represents what
> the method is for. That re-sweep is the right vehicle for recovering tail
> efficiency, and it should wait until `batch_size` has been checked, since a
> fourth fixed lever changes what the sweep is searching over.

### 4.3.22 CVaR CE validated offline — the two objectives rank in opposite order

`--cvar-ce` (α = 0.05) across all eight saved configurations. No new sampling:

| run | centred `ratio` | mean CE | **CVaR CE** | SE | CVaR acc |
|---|---|---|---|---|---|
| `amp1e3` | 1.2308 | 0.2758 | **0.3093** | 0.0077 | 0.8701 |
| `nmeas256` | 1.2297 | 0.2826 | **0.3127** | 0.0070 | 0.8701 |
| `mdecay06` | 1.5150 | 0.2639 | 0.3451 | 0.0621 | 0.8571 |
| `amp1e4` | 1.3200 | 0.2167 | 0.3931 | 0.0274 | 0.8701 |
| `amp1e4_c16` | 1.3490 | 0.2157 | 0.4081 | 0.0279 | 0.8701 |
| `mdecay008` | 1.3230 | 0.2101 | 0.4105 | 0.0715 | 0.8701 |
| `c16` | 1.6026 | 0.1946 | **0.7345** | 0.0817 | 0.7922 |
| `c8` | 1.5913 | **0.1919** | **0.7463** | 0.1329 | 0.7662 |

> **At the amplitude mean CE selected, the CVaR reward predicts preferences
> WORSE THAN CHANCE.** `log 2 = 0.6931`; `c8` scores **0.7463** and `c16`
> **0.7345**. The conservative reward — the mechanism the paper's claim rests
> on — is worse than uninformative at the configuration the selection procedure
> chose, while its *mean* CE (0.1919) is the best of all eight. CVaR accuracy
> 0.7662 with CE above `log 2` means confidently wrong, i.e. badly calibrated
> rather than merely noisy.

**The objectives are anti-correlated with stationarity in opposite directions.**

| | correlation with centred `ratio` |
|---|---|
| mean CE | **−0.607** — better stationarity, *worse* mean CE |
| CVaR CE | **+0.833** — right sign |
| CVaR CE jackknife SE | **+0.855** — the objective is noisiest exactly where the sampler is worst |

Mean CE's best three are `c8`, `c16`, `mdecay008`; CVaR CE's best three are
`amp1e3`, `nmeas256`, `mdecay06`. **Nearly reversed.** Mean CE does not merely
fail to see sampler quality — it actively rewards its absence, which is §4.3.14
in one number and explains why three successive amplitude caps were chased.

CVaR CE also has a **2.41× dynamic range against mean CE's 1.47×**, so it
separates configurations that mean CE compresses.

**It balances stationarity against tail precision rather than proxying for
either.** `mdecay06` ranks 3rd on CVaR CE despite the 3rd-*worst* centred
`ratio` (1.5150), because its tail precision is the best measured (unresolved
0.84%, `cvar_ess` 53.54). A pure stationarity proxy would have ranked it 6th.
That is the behaviour wanted from a selection objective and it is not something
either diagnostic delivers alone.

**Selectability at 30 tail draws — the budget question, answered.** Resolvable
differences are ~2·SE:

- `c8` vs `amp1e3` (0.7463 vs 0.3093): trivially resolved.
- `amp1e4` vs `amp1e3` (0.3931 vs 0.3093, gap 0.084 against 2·SE ≈ 0.055):
  **resolved.**
- `amp1e3` vs `nmeas256` (0.3093 vs 0.3127, gap 0.0034 against 2·SE ≈ 0.015):
  **not resolved — a statistical tie.**

So the objective discriminates the differences that matter and cannot separate
near-ties. Closing that last gap would need SE ~4.4× smaller, i.e. ~19× the
draws (tail ~570, total ~11 400) — not worth it, since those two configurations
are independently known to sit on the same floor (§4.3.19). **α = 0.05 at
8 × 75 is adequate for coarse selection.** At 4 × 75 the tail is 15 draws and
the tool warns; do not select at that budget.

> **The one thing NOT yet established: an interior optimum.** CVaR CE is
> *monotone decreasing* across the whole tested amplitude range
> (1.69e5 → 0.7463, 1.69e4 → 0.3931, 1.69e3 → 0.3093). It reverses the
> direction of §4.3.14's pathology but has not been shown to *bound* it. Theory
> says a minimum must exist — as `map_amp2 → 0` the prior forces `f → 0`,
> every logit difference vanishes and CE → `log 2` = 0.6931 — so the curve must
> turn. **It is not bracketed, and until it is, the possibility that CVaR CE
> simply prefers ever-narrower posteriors is not excluded empirically.** That
> failure mode matters: a selection objective that drives the posterior to a
> point mass degenerates the BNN into MR and removes the conservatism the paper
> is about. **Bracket it before adopting the objective**, with one run below the
> current floor:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 8 --chains_per_gpu 4 --map_amp2 168.93982289052463 \
    --OUT_DIR ./exp/stage3_medium_play_amp1e2 > ../exp/stage3_medium_play_amp1e2.log 2>&1 &
```

CVaR CE rising at 1.69e2 brackets the minimum near 1.69e3 and the objective is
safe to adopt. Still falling means it is tracking posterior width rather than
sampling quality, and the objective needs a calibration term before it can be
used for selection.

### 4.3.23 The bracket closes — CVaR CE has an interior optimum

`map_amp2` 168.9, 8 chains, completing the amplitude curve:

| `map_amp2` | CVaR CE | mean CE | centred `ratio` | |
|---|---|---|---|---|
| 1.69e5 | 0.7463 | 0.1919 | 1.5913 | worse than `log 2` |
| 1.69e4 (**principled**) | 0.3931 | 0.2167 | 1.3200 | |
| 1.69e3 | **0.3093** | 0.2758 | 1.2308 | **CVaR CE minimum** |
| 1.69e2 | 0.4102 | 0.4021 | 1.1770 | turns back up |

**The objective is bounded on both sides and safe to adopt.** Too wide and the
posterior is badly sampled — CVaR CE 0.7463, worse than chance. Too narrow and
it degenerates. The minimum sits at 1.69e3.

**The turn happens for exactly the predicted reason.** At 1.69e2 the three
metrics collapse onto each other — plug-in 0.4021, predictive 0.4066, CVaR
0.4102, a spread of **0.008** — against a spread of **0.191** at 1.69e4. The
posterior has narrowed until CVaR ≈ mean, so CVaR CE inherits mean CE's
degradation as `f → 0`. **That is the mechanism that bounds the objective**, and
it is now observed rather than argued: the failure mode §4.3.22 worried about
(driving the posterior to a point mass) is self-limiting, because the objective
stops rewarding narrowing at precisely the point where conservatism disappears.

**Stationarity and CVaR CE do *not* share an optimum, and that is correct
behaviour.** Centred `ratio` keeps falling past the CVaR minimum (1.2308 →
1.1770 at 1.69e2), so a pure stationarity objective would keep tightening the
prior while CVaR CE turns back. **The objective balances conservatism quality
against sampling quality rather than maximising either** — the property
§4.3.22 wanted and could not yet demonstrate.

Incidentally the extra point revises §4.3.17's plateau estimate: excess widening
0.5913 → 0.3200 → 0.2308 → 0.1770, with decay ratios 0.329 then 0.603, so the
gains shrink more slowly than the two-point geometric fit assumed. Updated
floor ≈ **1.10**, against the 1.19 projected. Still not 1.0, and still not
reachable by prior strength alone.

> **An unresolved tension worth carrying forward.** The *derived* amplitude is
> ~1.69e4 (logit-matching, §4.3.16) but CVaR CE prefers **1.69e3** — one decade
> tighter, by 0.0838 against a 2·SE of 0.0548, so the preference is
> **resolvable, not noise**. Two readings, and they are distinguishable:
>
> 1. The derivation is right and the empirical optimum is compensating: an
>    over-tight prior is absorbing the sampler's uncorrected excess heat
>    (§4.3.21 — the `−lr⁴` correction is numerically inert), buying back what
>    the thermostat is adding.
> 2. The derivation's logit-scale argument is approximate and the true prior
>    scale really is a decade lower.
>
> **These make opposite predictions under a sampler fix.** If (1), then
> correcting the gradient noise — full batch, or a fixed measurement set, or a
> real `B̂` subtraction — should move the CVaR-optimal amplitude *up* toward the
> derived 1.69e4 and close the gap. If (2), the gap persists. **That is the
> cleanest available test of whether the sampler fix actually worked**, and it
> costs nothing beyond re-running this four-point curve afterwards. Do not fix
> `map_amp2` in the sweep at either value until it is settled.

### 4.3.24 The fixed measurement set fails — and the reason kills the route

`--fix_meas_set True` at the principled amplitude, 8 chains, every other field
identical to `amp1e4` (verified against the logged config, so this is a clean
single-variable comparison):

| metric | `amp1e4` (resampled) | `fixmeas` (fixed) | |
|---|---|---|---|
| centred `ratio` | 1.3200 | **1.5977** | much worse |
| centred `scale_z` | 1.1812 | 1.8867 | worse |
| **CVaR CE** | 0.3931 | **0.5906** | much worse |
| CVaR accuracy | 0.8701 | 0.8442 | worse |
| unresolved | 1.44% | 3.28% | worse |
| plug-in CE | 0.2021 | **0.1777** | **better** |
| predictive CE | 0.2167 | **0.1951** | **better** |
| `ess_bulk` median | ~14.5 | 14.66 | unchanged |

**Mean CE improved while everything that matters degraded** — §4.3.22's
signature of a wider, more weakly constrained posterior.

**The premise was wrong, and the pool size shows why.** The measurement pool is
**999,000 observations**. Resampling 35 of them per step over ~240 000 steps is
**8.4 million point-visits**, and its time-average is the *full-pool* prior
gradient: the resampling is a **stochastic approximation of a prior over the
whole pool**, not merely noise added to a fixed one. Freezing the set replaces
that with a prior supported on **35 points — 0.0035% of the pool** — for the
entire run. That is not a variance reduction, it is **a different and far weaker
prior**, so `fix_meas_set` changes the target distribution rather than the
sampler's noise.

The magnitude confirms it. Freezing 35 points at `map_amp2` 1.69e4 reproduces
the loose-prior regime almost exactly:

| configuration | centred `ratio` |
|---|---|
| `fixmeas` — amp 1.69e4, **fixed** | 1.5977 |
| `c8` — amp 1.69e5, resampled | 1.5913 |

**Freezing the measurement set is worth roughly a 10× looser prior.**

> **This retires route (a) of §10.2 step 1, and the correction runs deeper.**
> The functional-prior gradient's stochasticity **cannot be removed without
> changing the prior**: an exact full-pool gradient needs a 999 000² kernel and
> a 999 000³ Cholesky, which is not merely expensive but impossible. **At this
> pool size the prior-gradient noise is intrinsic to the method.** It therefore
> has to be *corrected for*, not eliminated — which makes a real `B̂`
> subtraction (route (c)) the only route that addresses the dominant noise
> source, and demotes full batch (route (b)) to removing the *minibatch*
> component only.
>
> It also revises §4.3.19 and §4.3.21. `n_meas` 35 → 256 was read there as
> reducing gradient noise; it does that, but it **also quadruples prior
> coverage**, and this run shows coverage is the dominant term — freezing at 35
> costs far more than the noise it removes. The `n_meas` result stands; the
> attributed mechanism was incomplete.

**One run would settle how much of §4.3.19's gain was coverage versus noise**,
and it is worth having before the sweep fixes `n_meas`:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 8 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --n_meas 256 --fix_meas_set True \
    --OUT_DIR ./exp/stage3_medium_play_fixmeas256 > ../exp/stage3_medium_play_fixmeas256.log 2>&1 &
```

Against `nmeas256` resampled (centred `ratio` 1.2297, CVaR CE 0.3127). Landing
near 1.23 would mean coverage at 256 points is already sufficient and the
remaining noise is what `n_meas` was buying — which would revive fixed sets at
larger `n`. Landing near 1.6 again means coverage dominates at any feasible
`n_meas` and **fixed sets are dead**, leaving `B̂` as the only route.

`--fix_meas_set` stays in the code at **default False** — every prior run is
bit-identical — as the instrument for that question, not as a fix.

### 4.3.25 The 2×2 completes — fixed sets are dead, with a bound

`--fix_meas_set True --n_meas 256`, principled amplitude, 8 chains. Centred
`ratio`, all four cells:

| | resampled | fixed | freezing penalty |
|---|---|---|---|
| `n_meas` 35 | 1.3200 | 1.5977 | **+0.2777** |
| `n_meas` 256 | **1.2297** | 1.3230 | **+0.0933** |

**The interaction predicted by §4.3.24 is real — and insufficient.** The
freezing penalty falls **3×** between `n_meas` 35 and 256, exactly as the
coverage account requires. It does not vanish. Freezing at 256 is still worse
than resampling at 256 on centred `ratio` (1.3230 vs 1.2297), and that gap is
**resolvable**: the ratio's run-to-run stability is 0.7–2.2% (§4.3.16, §4.3.18),
so 0.0933 is several times the noise floor.

**The penalty scales as `n^-0.548`** — essentially `n^-0.5`, the Monte-Carlo
rate, which is what a coverage deficit should obey. Extrapolating: driving the
penalty under 0.01 needs **`n_meas` ≈ 15 000**, whose Cholesky alone is
`3.4e12` flops/step, about **23 GPU-hours per chain** on top of everything else.
**Fixed measurement sets are not merely unhelpful here, they are unreachable**:
the `n` at which freezing becomes free is computationally out of range by orders
of magnitude. Route (a) is closed with a bound, not just a failed experiment.

**One honest qualification.** On the *deployed* objective the difference is a
tie, not a loss: CVaR CE 0.3568 (fixed) vs 0.3127 (resampled), a gap of 0.0441
against a resolvable threshold of **0.0695** (jackknife SE 0.0347, 30 tail
draws). The kill comes from centred `ratio`, where the gap *is* resolvable.
And freezing does deliver real noise reduction — `cvar_ess` 27.39 → 36.14, up
32% — it simply costs more in coverage than it returns in noise, at every
feasible `n_meas`.

> **Consequence for §10.2 step 1.** Route (a) is dead and route (b) (full batch)
> only ever addressed the *minibatch* component, which §4.3.24 showed is not the
> dominant source. **Route (c) — a real `B̂` subtraction in
> `adaptive_sghmc.py` — is the only remaining route to the dominant noise, and
> it is now the critical path.** That is also the more principled fix: §4.3.21
> established the existing `-lr⁴` correction is numerically inert (3.8e-15
> against 2.4e-8…2.4e-6), so the sampler has never corrected for gradient noise
> at all. Correcting it properly fixes the defect at its source and is
> insensitive to `n_meas`, `batch_size` and pool size alike — which also means
> it does **not** touch `bt_pool` or `batch_size`, so the cross-family
> comparability constraints (§3.6.2, §4.3.15) never bind.

### 4.3.26 The `B̂` correction already exists — §4.3.21's mechanism was wrong

**Checked before implementing anything, and the check was necessary.** The
codebase's own spec (`docs/fsghmc_algorithm.pdf`, §2.1 Eq. 7) states:

> σ² = max(2ε²ca − ε⁴, 10⁻¹⁶) … "The −ε⁴ term in (7) is the correction for the
> minibatch-noise estimate `B̂ = ½ε V̂`, which cancels against the squared
> preconditioner exactly as in [Springenberg et al.]."

`adaptive_sghmc.py:147` implements exactly that. **The `B̂` correction is
present, documented, and matches the reference.** There is nothing missing to
add, and the "critical path" §4.3.25 named does not exist.

**The algebra shows why it is small — by design, not by omission.** Per step the
gradient noise enters `v` through the drift term:

    Var(ε² a ∇Ũ_noise) = ε⁴ a² Σ

and the injected noise is `2ε²ca − ε⁴`. With `Σ ≈ V̂` (the spec's `B̂`
assumption) and `a = V̂^{-1/2}`, so `a²Σ ≈ 1`:

    ε⁴·a²Σ  +  (2ε²ca − ε⁴)  =  2ε²ca        ← exact cancellation

The gradient-noise term is **O(ε⁴)** while the thermostat is **O(ε²)**. At this
step size their ratio is **1.6e-7 to 1.6e-9**. Gradient noise is not a dominant
uncorrected heat source; it is correctly cancelled *and* negligible.

> **§4.3.21's mechanism is retracted.** It claimed the `-lr⁴` term was
> "numerically inert" and therefore that gradient noise was the uncorrected
> dominant source. The term is small for the same reason the thing it corrects
> is small — both are O(ε⁴) — and reading its magnitude without comparing it to
> the quantity it cancels was the error. **Every measurement in §4.3.16–25
> stands; only the mechanism attributed to them changes.**

**What actually explains the observations, without any uncorrected noise:**

| observation | corrected mechanism |
|---|---|
| `map_amp2` ↓ → less widening | **prior strength** — a tighter prior narrows the posterior |
| `n_meas` ↑ → less widening | **prior coverage** — §4.3.24 already showed coverage dominates |
| `fix_meas_set` → more widening | **coverage loss** (§4.3.24, §4.3.25) |
| `mdecay` ↑ → more widening | **discretisation error** — σ ∝ √mdecay, so larger per-step displacement, larger O(ε) bias |
| `mdecay` ↓ → *no change* | **exactly what theory predicts**: in exact SGHMC the stationary distribution is **independent of C** (fluctuation–dissipation). The null result is correct behaviour, not a saturating noise sum. |

The asymmetry that drove §4.3.21 is therefore explained by theory rather than by
a binding-source argument: lowering friction *cannot* change the target, and
raising it only does so through discretisation error.

**The residual is a mixing problem, and the shared start is the likely source.**
`ess_bulk` is ~14 of 600 draws — **2.4% efficiency**. Chains start from a shared
warm-up point with `chain_init_jitter = 0`, i.e. from a near-*point* mass, and
must expand to the posterior width. That expansion **is** the measured widening,
and it is per-chain, which is precisely the ALIGNMENT 0.4593 signature of
§4.3.18. At 2.4% efficiency 75 draws is nowhere near enough to finish.

This reframes §4.3.8's inconclusive jitter test: `chain_init_jitter = 0.1` is
10% of each tensor's own sd, which is not remotely posterior scale. **The test
was of the right lever at the wrong magnitude.**

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 \
    --OUT_DIR ./exp/stage3_medium_play_jit10 > ../exp/stage3_medium_play_jit10.log 2>&1 &
```

16 chains, so ALIGNMENT is readable (§4.3.7) — and under this account ALIGNMENT
and centred `ratio` should fall *together*, since both are consequences of
chains expanding from a common point. Judge on centred `ratio` against 1.3200
and CVaR CE against 0.3931. If a posterior-scale start removes the widening, the
sampler needs no correction at all and the whole §4.3.21 line closes.

### 4.3.27 A posterior-scale start reduces the widening — and costs the tail

`chain_init_jitter 1.0`, 16 chains, principled amplitude, against
`amp1e4_c16` (identical but jitter 0):

| metric | jitter 0 | **jitter 1.0** | |
|---|---|---|---|
| **centred `ratio`** | 1.3490 | **1.2494** | **resolvably better** |
| centred `scale_z` | 1.7532 | 1.2228 | much better |
| centred `loc_sd` | 0.2499 | 0.1986 | better |
| CVaR CE | 0.4081 | 0.2417 | **not resolvable — see below** |
| CVaR CE jackknife SE | 0.0279 | **0.1911** | **6.8× worse** |
| unresolved | 0.17% | **1.41%** | 8× worse |
| `ess_bulk` median | 28.41 | 23.91 | worse |
| mean CE | 0.2232 | 0.2388 | worse |
| ALIGNMENT | 0.4593 | 0.4075 | — |

**The expansion account is supported.** Centred `ratio` falls 1.3490 → 1.2494,
a gap of 0.0996 against a run-to-run stability of 0.01–0.03 (§4.3.16, §4.3.18),
so it is resolvable several times over. **Starting chains at posterior scale
removes part of the widening**, which is what §4.3.26 predicted and is the first
lever outside prior strength to move it.

**But the apparent CVaR CE win must not be claimed.** 0.4081 → 0.2417 is a gap
of 0.1664 against a combined 2·SE of **0.3863**. The jackknife SE explodes from
0.0279 to **0.1911** — 6.8× — because overdispersed chains genuinely disagree
about the tail, so the pooled CVaR becomes chain-dependent. The point estimate
is the best in the investigation and it is **statistically indistinguishable
from the baseline**.

> **That SE explosion is itself a finding, and an awkward one.** Overdispersion
> is standard practice precisely so that `R̂` is a valid convergence check
> (§2.6 of the spec, and `f_pref_net.py:130`). But it degrades the *estimator*
> of the quantity this paper depends on: with 16 chains, a tail averaged over
> chains that disagree has 6.8× the error. **CVaR CE becomes much harder to
> select on exactly where overdispersion is used**, which is a direct problem
> for the sweep redesign — and it is a property of the deployed CVaR, not only
> of the diagnostic.

**Jitter reaches the same floor as prior strength, by an independent route.**
Correcting for the ~2.2% 8→16-chain offset (§4.3.18), `jit10` sits at ≈1.22 in
8-chain terms, against the prior-strength floor of 1.2297 (`nmeas256`) and
1.2308 (`amp1e3`). Two mechanically unrelated levers — initial dispersion and
prior strength — land within ~1% of each other. **Whether they compose or share
a common limit is the open question**, and it decides whether §4.3.23's ~1.10
projection is reachable at all:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 \
    --OUT_DIR ./exp/stage3_medium_play_jit10n256 > ../exp/stage3_medium_play_jit10n256.log 2>&1 &
```

Landing near ~1.15 means the two compose and the floor is not hard. Landing at
~1.22 again means both are hitting the same limit from different directions,
which would make that limit the real object of study rather than either lever.

**A tension worth exploiting rather than lamenting.** `mdecay` ↑ gave a better
tail and a worse widening (§4.3.20); jitter ↑ gives a better widening and a
worse tail. **Opposite trades acting through different mechanisms** —
discretisation error versus initial dispersion — so a combination is a genuine
candidate rather than a compromise, and worth a run once the composition
question above is settled.

### 4.3.28 The identified component reaches stationarity

`chain_init_jitter 1.0` **and** `n_meas 256`, 16 chains, principled amplitude:

| | `loc_sd` | `ratio` | `loc_z` | `scale_z` | |
|---|---|---|---|---|---|
| raw `f` | 0.0931 | 1.5831 | 0.3241 | 2.2827 | FAIL |
| **centred (shape)** | 0.1539 | **1.0871** | **0.6144** | **0.6469** | **PASS** |
| offset only | 0.1038 | 1.6454 | 0.3567 | 2.4542 | FAIL |

**The identified component is stationary.** A stationary chain's `|N(0,1)|`
median reference is **0.6745**; the centred scores are **0.6144** (0.91×) and
**0.6469** (0.96×) — indistinguishable from a chain that is sampling its target.
Centred `ratio` **1.0871** beats §4.3.23's projected ~1.10 floor, which was
never reachable by prior strength alone.

**And every remaining bit of drift sits in the direction that provably does not
matter.** §4.3.10 established that the BT/CE likelihood is exactly invariant to
`f → f + c`: the offset is unidentified by the data, cancels in every preference
prediction, and a constant reward offset leaves the IQL greedy policy unchanged.
The residual is `ratio` 1.6454 / `scale_z` 2.4542 **on the offset alone**.

**The levers compose superadditively.** Jitter alone removed 0.0996 of the
excess; `n_meas` 256 alone removed 0.0903 (8-chain); together they removed
**0.2619**, against a sum of separate effects of 0.1899. Neither §4.3.19's nor
§4.3.27's floor was a hard limit — each lever was bounded by the other's
mechanism.

**The §4.3.27 estimator problem is cured.** Overdispersed starts inflated the
CVaR CE jackknife SE 6.8× to 0.1911 because chains disagreed about the tail.
With the stronger prior the SE falls to **0.0575** — 3.3× smaller — and the
comparison becomes decisive:

| | CVaR CE | SE |
|---|---|---|
| `amp1e4_c16` | 0.4081 | 0.0279 |
| **`jit10n256`** | **0.2648** | 0.0575 |

Gap **0.1433** against a combined 2·SE of **0.1278** — **the first resolvable
CVaR CE improvement in the investigation.**

**Costs, stated plainly.** `cvar_ess` 76.33 → 25.19, relMCSE median 0.4560,
unresolved 0.17% → 0.81%, mean CE 0.2232 → 0.2912. Tail *efficiency* is worse
even as tail *correctness* improves — the same dissociation §4.3.20 found, and
the reason §4.3.15's rule to judge on CVaR CE rather than on any single
diagnostic matters here.

> **Procedural consequence, and it is not optional. §4.2's gate and §3.6.3's
> eligibility criterion are computed on RAW `fn_drift_*`.** This configuration
> **fails the raw gate** (`scale_z` 2.2827) while being stationary in the
> identified component. Under the criteria as written, **the best-sampling
> configuration produced in this investigation would be rejected as
> ineligible** — and the runs that pass are the ones whose offset happens to be
> pinned by an over-tight prior, which is what §4.3.14 diagnosed as the original
> pathology.
>
> **The gate must move to the centred component**, with the offset reported
> separately rather than gated on. Until that change lands, do not run the sweep
> redesign: it would select against the thing it is meant to find. §7.1 should
> also record that the published stationarity criterion was computed on a
> quantity containing an unidentified direction.

**Next.** Confirmed on large_diverse (§4.3.29) — the hardest variant, and it
reaches centred stationarity too. large_play and medium_diverse remain.

### 4.3.29 The recipe generalises — confirmed on large_diverse

Same recipe (`map_amp2` 16894, `chain_init_jitter 1.0`, `n_meas 256`), 16
chains, on the variant §4.3.2 recorded as the **hardest** — 22 of 40 trials
rejected, the thinnest eligible field of the four. Its `n_meas` went **7 → 256**,
a 37× change against medium_play's 7.3×:

| | `loc_sd` | `ratio` | `loc_z` | `scale_z` | |
|---|---|---|---|---|---|
| raw `f` | 0.0595 | 1.0476 | 0.2015 | 0.2502 | PASS |
| **centred (shape)** | 0.1642 | **1.1278** | **0.7154** | **0.8091** | **PASS** |
| offset only | 0.0561 | 1.0421 | 0.1866 | 0.1934 | PASS |

**The identified component is stationary here too.** Centred `loc_z` 0.7154 is
1.06× the `|N(0,1)|` null median of 0.6745 and `scale_z` 0.8091 is 1.20× —
indistinguishable from a chain sampling its target, as on medium_play (0.6144,
0.6469). **Two of four variants now confirmed**, including the hardest.

**The amendment's logging is live and correct.** wandb's summary carries
`val_fn_drift_centred_loc_z_median` 0.715435 and `centred_scale_z_median`
0.809066, matching the offline decomposition to five digits. **§3.6.3's
criterion is now evaluable directly from wandb**, without saved chains — the gap
that made round 2 un-re-adjudicable is closed for every future trial.

> **And the amendment is independently vindicated by a variant difference.**
> The two variants disagree entirely about the *offset* while agreeing about the
> identified component:
>
> | | medium_play | large_diverse |
> |---|---|---|
> | centred `ratio` | 1.0871 | 1.1278 |
> | **offset `ratio`** | **1.6454** | **1.0421** |
> | **raw `scale_z`** | **2.2827 FAIL** | **0.2502 PASS** |
>
> Raw would **reject medium_play and accept large_diverse** on the strength of a
> direction that cancels in every preference prediction. The centred criterion
> accepts both, correctly, because both are stationary in the part that
> matters. **The amendment makes the criterion consistent across variants**,
> which is a stronger argument for it than the single-run case in §3.6.3.
>
> It also does **not** loosen the gate: large_diverse passes raw as well, so
> centring only changes a verdict when the offset actually drifts. It is not a
> route for admitting badly-sampled runs.

**Tail: the best unresolved fraction in the investigation** — **2 of 6400
(0.03%)**, against the previous best of 0.17%. But the same compression as
§4.3.17: relMCSE median 0.5411 and `cvar_ess` 28.65 are mediocre while the
extremes are excellent. Typical points noisier, unusable points essentially
eliminated.

> **One claim that cannot be made.** CVaR CE is 0.3417 (SE 0.0280), but there is
> **no matched large_diverse baseline** — its `c4` figures come from a sweep
> trial whose chains no longer exist (§4.3), and no unmodified 16-chain run was
> made. So this number cannot be called an improvement. The **stationarity**
> claim stands unaided because it is absolute — measured against the `|N(0,1)|`
> null, not against a baseline — but the CVaR claim is not transferable from
> medium_play and would need a control run to establish here.

**Remaining: large_play and medium_diverse.** Both were selected under the same
sweep and carry the same signature (§4.3.2). With the centred metrics now logged
automatically, each is one run and the verdict reads straight from wandb.

### 4.3.30 The recipe does NOT generalise — two of four fail, by contraction

Same recipe on the remaining two variants. Both fail, and neither fails the way
§4.3.28 was fixing:

| variant | `mdecay` | centred `ratio` | centred `scale_z` | gate | CVaR CE |
|---|---|---|---|---|---|
| large_diverse | 0.3761 | 1.1278 | 0.8091 | PASS | 0.3417 |
| medium_play | 0.1946 | 1.0871 | 0.6469 | PASS | 0.2648 |
| **large_play** | **0.0312** | **0.8578** | 1.0365 | PASS | **3.0359** |
| **medium_diverse** | **0.0072** | **0.5299** | **13.1542** | **FAIL** | 0.6705 |

**Both failures are CONTRACTION, not expansion.** Centred `ratio` < 1 means the
identified component's spread *shrank* between halves — 0.86 for large_play and
**0.53** for medium_diverse, whose posterior width nearly halved. That is the
opposite of the widening every section from §4.3.2 onward chased, so the recipe
has not merely failed to help here: **it has overshot in the other direction.**

**large_play passes both gates while being broken.** CVaR CE **3.0359** against
`log 2` = 0.6931, with CVaR accuracy **0.5741** — barely above chance, and the
worst number in the investigation by a wide margin (SE 0.4702). Raw and centred
`scale_z` are 0.7649 and 1.0365, both comfortably inside the criterion.
**§4.3.15's rule holds and is now demonstrated twice: stationarity is necessary,
not sufficient.** A chain that has collapsed is stationary in the trivial sense,
and no drift diagnostic can tell that apart from a chain sampling correctly.
Only CVaR CE catches it.

**medium_diverse is the amendment working in the protective direction.** Raw
`scale_z` is **0.8187 — a comfortable PASS** — while centred is **13.1542**, a
catastrophic fail. §4.3.29 showed the amendment stopping raw from rejecting a
good configuration; this is the converse, and the more important one: **raw
would have accepted a run whose identified component collapsed by half.** The
two cases together establish that centring is not a loosening or a tightening of
the criterion but a correction of what it measures.

> **Mechanism: friction separates the four exactly.** The two failures carry the
> two lowest `mdecay` (0.0072, 0.0312); the two successes the two highest
> (0.1946, 0.3761). `chain_init_jitter 1.0` displaces each chain by 100% of each
> tensor's own sd, and friction is what dissipates that displacement. **Below
> some `mdecay`, burn-in cannot absorb an overdispersed start**, so chains enter
> the sampling phase still falling inward — which registers as contraction, and
> at low enough friction (medium_diverse, 27× below medium_play) as collapse.
>
> **This is n = 4 with perfect separation, which is suggestive and not
> established.** It is also confounded: `map_amp2` was moved to 16894 from
> per-variant values spanning ÷5.6 to ÷54.8, and large_play's ÷54.8 is by far
> the largest. The warm-up point that the jitter scales off was produced under
> the variant's *original* amplitude, so a large amplitude change means the
> jitter is calibrated to weights of the wrong scale. Friction and amplitude
> ratio are not independent across these four, and one run cannot separate them.

**What this does and does not overturn.** §4.3.28 and §4.3.29 stand as measured:
medium_play and large_diverse do reach centred stationarity under this recipe.
What fails is the claim that **one fixed recipe** transfers to all four.
`chain_init_jitter` is not a variant-independent constant — it must be scaled to
what the chain can dissipate, which depends on `mdecay`, and possibly to the
amplitude change as well.

**The discriminating run.** Re-run the two failures with jitter scaled down
rather than the recipe abandoned. If friction is the mechanism, a jitter small
enough for the available damping should restore expansion-then-stationarity:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_diverse_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 0.05 --n_meas 256 \
    --OUT_DIR ./exp/stage3_medium_diverse_jit005 > ../exp/stage3_medium_diverse_jit005.log 2>&1 &
```

medium_diverse first: it is the cleaner test, since its amplitude change (÷7.1)
sits inside the range where the recipe already works, leaving `mdecay` as the
distinguishing variable. Centred `ratio` returning to ≥1 with `scale_z` near the
0.6745 null confirms friction; still contracting means the amplitude change is
the culprit and the principled value needs deriving per variant after all.

**Do not run the sweep redesign on the current recipe.** Two of four variants
produce either a collapsed posterior or a worse-than-chance CVaR reward, and
`large_play` shows the gate cannot detect it.

### 4.3.31 The friction hypothesis is refuted — and a confound replaces it

`chain_init_jitter` 1.0 → **0.05** on medium_diverse, everything else as
§4.3.30:

| | jitter | centred `ratio` | centred `scale_z` | centred `loc_z` | raw `scale_z` |
|---|---|---|---|---|---|
| `recipe` | 1.0 | 0.5299 | 13.1542 | 2.5544 | 0.8187 PASS |
| `jit005` | **0.05** | **0.4556** | 10.7043 | 2.0173 | 0.7841 PASS |

**A 20× reduction in jitter did not fix the contraction — it deepened it.**
§4.3.30's mechanism was that burn-in cannot dissipate an overdispersed start at
low friction; under that account, near-eliminating the dispersion should have
restored expansion. It did the opposite. **Refuted.**

**A second hypothesis is out as well.** The warm-up that produces
`initial_weights` runs **in the same script under the same config**
(`run_bnn_training_antmaze_eval.py:619`), so it already uses the recipe's
`map_amp2` and `n_meas`. The start is not a point fitted under the variant's
original prior.

**What survives is a clean but confounded signal.** Centred `ratio` is
**monotone-inverse in warm-up accuracy** across all four variants:

| variant | `mdecay` | warm-up acc | centred `ratio` |
|---|---|---|---|
| large_diverse | 0.3761 | 0.5182 | 1.1278 |
| medium_play | 0.1946 | 0.7273 | 1.0871 |
| large_play | 0.0312 | 0.7593 | 0.8578 |
| medium_diverse | 0.0072 | **0.8411** | **0.5299** |

The better the warm-up fits, the harder the chain contracts — perfectly ordered,
no exceptions. **But `mdecay` is monotone in the same order.** The two are
*perfectly confounded* at n = 4 and cannot be separated from these runs. The
jitter test refutes one specific friction mechanism, not friction as such.

> **Do not add a third n = 4 story on top of two refuted ones.** §4.3.30's
> friction account and the warm-up-prior account both looked clean on four
> points and both are now dead. Any explanation resting on ordering four
> variants is worth roughly nothing until a *within-variant* measurement
> distinguishes the candidates.

**The discriminating measurement needs no new sampling.** `--drift-blocks`
(§4.3.21) splits the draws into consecutive blocks and reports the block-to-block
scale change against a noise floor, which separates a **decaying transient** —
a chain relaxing from an atypical start, which is what both refuted hypotheses
predicted — from a **constant per-cycle drive**, which implicates the sampler or
the target rather than the initialisation:

```
python scripts_bnn/diagnose_sampling_tail.py \
    --run-dir exp/stage3_medium_diverse_jit005_0 --drift-blocks 5 \
    --offset-shape-split --device cuda \
    2>&1 | tee exp/stage3_medium_diverse_jit005_0_blocks.txt
```

The `sd/first` column is the one to read: falling monotonically toward a plateau
means a transient and the initialisation family of explanations survives in some
form; falling at a constant rate across all five blocks means the chain is being
driven inward throughout sampling, and initialisation is irrelevant. §4.3.19
warned this test is underpowered at 75 draws for *location*, but the scale
column is far better determined and the effect here is large (a 2× spread change),
so it should resolve.

**Status: the sweep redesign remains blocked**, and the recipe is confirmed for
two variants only (§4.3.28, §4.3.29). Two of four produce a collapsed identified
component, and `large_play` shows the gate cannot detect it (§4.3.30).

### 4.3.32 The contraction is a decaying transient — and a third confound appears

`--drift-blocks 5` on medium_diverse `jit005`, reading the centred column added
for this purpose:

| block | draws | raw `sd/first` | **centred `sd/first`** |
|---|---|---|---|
| 0 | 0–14 | 1.0000 | 1.0000 |
| 1 | 15–29 | 1.3022 | **0.4068** |
| 2 | 30–44 | 1.4508 | 0.3108 |
| 3 | 45–59 | 1.4246 | 0.3421 |
| 4 | 60–74 | 1.4092 | 0.2894 |

**Verdict: a decaying transient.** First per-block change 0.593, last 0.053 —
the collapse is essentially complete within the first block and the trajectory
is flat thereafter. Note raw `sd/first` *widens* to 1.41 across the same blocks:
without the centred column this run reads as the opposite phenomenon.

**This is the first mechanism in §4.3.30–32 resting on a within-variant
measurement rather than an ordering of four points**, which matters given
§4.3.30's friction account and §4.3.31's warm-up-prior account both looked clean
on four points and both are dead.

> **A third confound, and it separates the four exactly like the other two.**
> `cycle_length` is 500/750 for the two failures and 2750 for the two successes,
> so total sampling is **40–60k steps versus 220k**. `mdecay`, warm-up accuracy
> and sampling-window length are now **all three** monotone in the same order
> across n = 4. **These runs cannot distinguish them, and no further
> cross-variant ordering will.** Stop proposing them.

**A simple account that needs none of the three.** Every variant runs the same
**20,000-step burn-in**, and if the absolute relaxation time is similar across
variants, the same transient occupies:

| variant | transient ÷ sampling window |
|---|---|
| medium_play / large_diverse | ~20k / 220k ≈ **9%** |
| large_play | ~20k / 40k ≈ **50%** |
| medium_diverse | ~20k / 60k ≈ **33%** |

A transient covering 9% of the draws barely moves a first-half/second-half
ratio; one covering a third of them **dominates the first half**. On this
account the four variants may share one transient of similar absolute size, and
differ only in how much of it lands inside the sampling window — no appeal to
friction, warm-up quality or prior scale required. It also explains why cutting
jitter 20× did nothing (§4.3.31): the jitter was never the displacement being
relaxed.

**The fix follows from the transient alone, without resolving the confound.**
Lengthening burn-in absorbs it before sampling begins, whichever variant
property amplifies it:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_diverse_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --num_burn_in_steps 100000 \
    --OUT_DIR ./exp/stage3_medium_diverse_burn100k > ../exp/stage3_medium_diverse_burn100k.log 2>&1 &
```

5× the burn-in against a 60,000-step sampling window; the run grows from 80k to
160k total steps, which is still a third of medium_play's. Jitter is back at 1.0
because §4.3.31 established it is not the driver, and holding it at 0.05 would
confound this test with the §4.3.29 recipe.

> **The caveat that could sink this.** Burn-in runs at `lr_min` with the
> cyclical schedule **off**, and cycling begins only afterwards (§4.3.5,
> `f_pref_net.py:724–745`). A longer burn-in therefore equilibrates under
> *different dynamics* than sampling uses. If the transient is specifically the
> chain's response to the first hot phase — `lr_max` steps it never saw during
> burn-in — then **more burn-in at `lr_min` cannot absorb it at all**, and the
> lever is `n_discarded` (discard more post-burn-in draws) or a warm-up that
> runs the schedule. If 100k burn-in leaves centred `ratio` near 0.5, that is
> the answer, and it is worth knowing before spending more runs on burn-in.

### 4.3.33 Burn-in absorbs the transient — medium_diverse passes

`num_burn_in_steps` 20,000 → **100,000** on medium_diverse, jitter back at 1.0:

| | burn-in | centred `ratio` | centred `scale_z` | centred `loc_z` | verdict |
|---|---|---|---|---|---|
| `recipe` | 20,000 | 0.5299 | 13.1542 | 2.5544 | FAIL |
| `jit005` | 20,000 | 0.4556 | 10.7043 | 2.0173 | FAIL |
| **`burn100k`** | **100,000** | **0.9087** | **1.7976** | **0.9787** | **PASS** |

**§4.3.32's transient account is confirmed.** Five times the burn-in moves
centred `scale_z` from 13.15 to 1.80 and `ratio` from 0.53 to 0.91. medium_diverse
now passes the amended §3.6.3 gate, so **three of four variants work**.

**And the caveat that could have sunk it is refuted.** §4.3.32 warned that
burn-in runs at `lr_min` with cycling off, so a longer burn-in might not absorb
a transient that is really the chain's response to the first `lr_max` hot phase.
It absorbed it. **The transient is a relaxation toward the typical set, not a
schedule artifact** — which also retires the `n_discarded` / schedule-aware
warm-up alternatives that caveat proposed.

**It is a marginal pass, not a comfortable one.** `scale_z` 1.7976 sits close to
the 2.0 threshold and is **2.66× the 0.6745 null**, against medium_play's 0.6469
under the same recipe. `ratio` 0.9087 is still contracting slightly. The
transient is largely absorbed, not eliminated — more burn-in may be needed, and
the tail cost is real: `cvar_ess` 97.12 → 51.23, accuracy 0.8744 → 0.8505.

**large_play is the remaining failure** and has the shortest sampling window of
all four (40,000 steps, `cycle_length` 500), so on §4.3.32's account it should
need at least as much burn-in:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_large_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --num_burn_in_steps 100000 \
    --OUT_DIR ./exp/stage3_large_play_burn100k > ../exp/stage3_large_play_burn100k.log 2>&1 &
```

Judge it on centred `ratio`/`scale_z` **and CVaR CE against 3.0359** — §4.3.30
showed large_play passing both gates while its CVaR reward was worse than
chance, so the gate alone cannot clear it.

> **A unification worth testing, and a risk worth stating.**
>
> The two working variants sit slightly **above** 1 (medium_play 1.0871,
> large_diverse 1.1278) while the failures sit **below**. If the transient is a
> contraction that long sampling windows dilute to invisibility, then a uniform
> 100k burn-in might bring all four to ~1.0 — making the recipe genuinely
> variant-independent, which §4.3.30 showed the current one is not. Worth one
> re-run of medium_play at 100k burn-in to check, since a recipe that needs
> per-variant burn-in is much weaker than one that does not.
>
> **The risk:** the recipe is now **four hand-tuned knobs** — `map_amp2`,
> `chain_init_jitter`, `n_meas`, `num_burn_in_steps` — three of which were
> derived from medium_play's diagnostics and then patched when they failed
> elsewhere. That is a lot of post-hoc fitting to four variants. `map_amp2` and
> `n_meas` have principled justifications (§4.3.16, §4.3.24); jitter and burn-in
> do not, and are currently set by what worked. **State them as sampler-hygiene
> settings chosen on diagnostics, not as tuned hyperparameters**, and do not
> present the four-variant agreement as independent confirmation — the knobs
> were adjusted using those same four variants.

### 4.3.34 Burn-in is not a general fix — and the warm-up never converges

> **The second half of this heading is CORRECTED by §4.3.36.** The oscillation
> below was read on warm-up *accuracy*, which is quantized in units of 1/54 to
> 1/107. On **NLL** — continuous, and the same CE the run is selected on —
> medium_diverse converges monotonically and medium_play is flat; only
> large_play genuinely wanders away from its minimum. The burn-in result
> below stands, and §4.3.36 explains its sign.

`num_burn_in_steps` 100,000 on **large_play**, everything else as §4.3.33:

| | burn-in | centred `ratio` | centred `scale_z` | mean CE | acc |
|---|---|---|---|---|---|
| `recipe` | 20,000 | 0.8578 | 1.0365 | 0.2545 | 0.9005 |
| `burn100k` | 100,000 | **0.6194** | **2.2707 FAIL** | **0.7236** | 0.7755 |

**Five times the burn-in produced a worse model.** Mean CE 0.7236 is above
`log 2` = 0.6931 — worse than an uninformative predictor, on the *mean*, not
just the tail. §4.3.33's fix helped medium_diverse and broke large_play, so
burn-in length is not a general lever. That is the fourth mechanism in
§4.3.30–34 to work on some variants and fail on others.

**The burn-in trajectories say why, and the reason is worse than a failed fix.**
`warmup_log_every` was on, so the whole curve is recorded:

| run | burn-in | peak acc | at step | handed off | after the peak |
|---|---|---|---|---|---|
| large_play `recipe` | 20,000 | 0.8333 | 14,000 | 0.7593 | 30% |
| large_play `burn100k` | 100,000 | **0.9815** | 41,500 | **0.8148** | **58%** |
| medium_div `recipe` | 20,000 | 0.8785 | 19,000 | 0.8411 | 5% |
| medium_div `burn100k` | 100,000 | 0.9159 | 71,750 | 0.8224 | 28% |

**The warm-up does not converge — it oscillates, and the point handed to all 16
chains is wherever the oscillation happened to be at the final step.**
large_play's `burn100k` reached 0.9815 at step 41,500 and was handed off at
0.8148 after 58,500 further steps: **16.7 accuracy points worse than a point it
had already visited.** Longer burn-in did not refine the start, it gave the
walk more room to wander away from a good one.

**The measurement is also very noisy.** Warm-up accuracy is quantized in units
of 1/54 for large_play and 1/107 for medium_diverse (every logged value is an
exact multiple), so the eval sets are **54 and 107 pairs** — one pair is 1.85%
and 0.93% respectively. Swings of 10–17 points are 5–9 pairs.

> **This undercuts §4.3.31's warm-up-accuracy correlation.** That was four
> points of a statistic measured on ≤107 pairs and sampled at one arbitrary
> moment of an oscillating trajectory. It should not be carried forward as
> evidence of anything.
>
> **And it raises a harder problem for §4.3.28–33 as a whole.** Every
> configuration in that sequence is a **single run**, and each begins from a
> warm-up endpoint drawn from this oscillation. If the endpoint lottery moves
> centred `ratio` as much as the knobs do, most of those comparisons are
> underdetermined — including §4.3.28's flagship 1.0871 and §4.3.32's transient
> account, which rests on a single medium_diverse run.

**The next run is not another knob. It is a replicate.**

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --sampling_seed 100 --num_chains 16 --chains_per_gpu 4 \
    --map_amp2 16893.982289052463 --chain_init_jitter 1.0 --n_meas 256 \
    --OUT_DIR ./exp/stage3_medium_play_jit10n256_s100 > ../exp/stage3_medium_play_jit10n256_s100.log 2>&1 &
```

Identical to §4.3.28's flagship configuration, with only the sampling RNG
changed. Compare centred `ratio` against **1.0871** and centred `scale_z`
against **0.6469**.

> **`--seed` alone cannot do this.** `config.seed` selects the *data files*
> (`{data_root}/{variant}/eval/seed_{seed}/…`) as well as driving sampling —
> deliberately, so the model seed and the data split always match. `--seed 100`
> therefore looks for a seed-100 dataset that does not exist, and would change
> the data rather than replicate the run. **`sampling_seed`** (added 2026-08-24)
> re-seeds the warm-up, the per-chain RNG streams and the jitter draws while
> holding the data split at `seed`. It defaults to `None` = `config.seed`, so
> every existing run reproduces bit-identically, and it leaves §1's reserved
> seeds 1–10 untouched because the data seed never moves.

**This is the measurement that decides how much of §4.3.28–34 survives.** If the
replicate lands near 1.09, run-to-run variance is small and the sequence stands.
If it lands anywhere near medium_diverse's 0.53 or large_play's 0.62, then the
knob effects and the warm-up lottery are the same size, and **every single-run
comparison in this thread needs redoing with replicates** before anything is
concluded from it. Either answer is worth more than another knob.

> **Regardless of the replicate, the warm-up hand-off is a defect to fix.**
> Handing chains the *last* burn-in state rather than the best — or better, a
> state selected by a running average rather than a single noisy evaluation —
> is a design choice nothing in §3.6.2's provenance or the spec justifies, and
> it injects an uncontrolled random component into every run in this project.

### 4.3.35 The replicate holds — and kills the warm-up-lottery worry

§4.3.28's flagship configuration re-run with the same data (`seed 0`) and an
independent sampling RNG (`sampling_seed 100`):

| metric | original | replicate | |
|---|---|---|---|
| **centred `ratio`** | 1.0871 | **1.1198** | 1.03× |
| **centred `scale_z`** | 0.6469 | **0.6999** | 1.08× |
| centred `loc_z` | 0.6144 | 0.6617 | 1.08× |
| mean CE | 0.2912 | 0.2932 | 1.01× |
| accuracy | 0.8799 | 0.8782 | 1.00× |
| **raw `scale_z`** | **2.2827** | **1.1916** | **0.52×** |
| `cvar_ess` | 25.19 | 47.77 | 1.90× |
| **warm-up acc** | **0.7273** | **0.5195** | **0.71×** |

**§4.3.28 replicates**, and §4.3.34's worry is answered decisively — in the
opposite direction to the one feared. **Warm-up accuracy differed by 21
points** between the two runs, the exact lottery §4.3.34 identified, **and the
centred metrics barely moved** (3–8%). The warm-up endpoint is a lottery, but
it is **not what drives the centred drift statistics**. §4.3.28–33's
single-run comparisons are not undermined by it.

> **§4.3.31's warm-up-accuracy correlation is now definitively dead.** Within a
> single configuration, warm-up accuracy moved 0.7273 → 0.5195 while centred
> `ratio` moved 1.0871 → 1.1198. The n = 4 cross-variant correlation was
> coincidence, and this is a within-configuration refutation rather than
> another ordering argument.

**A third vindication of the amendment, and the sharpest.** Raw `scale_z`
varied **2.2827 → 1.1916 (1.9×)** between replicates of the *same
configuration*, flipping from a clear FAIL to a comfortable PASS, while centred
`scale_z` moved 8%. **The raw criterion is not reproducible; the centred one
is.** §3.6.3's amendment is not only measuring the right quantity — it is
measuring the only one of the two that a run can be held to.

**A resolution floor for everything in §4.3.26–34.** The replicate range on
centred `ratio` is **0.0327**, so:

| comparison | gap | vs replicate range |
|---|---|---|
| §4.3.33 medium_diverse burn-in | 0.3788 | 11.6× — solid |
| §4.3.28 composition | 0.2619 | 8.0× — solid |
| §4.3.34 large_play burn-in | 0.2384 | 7.3× — solid |
| §4.3.27 jitter alone | 0.0996 | 3.0× — probably real |
| §4.3.29 large_diverse vs medium_play | 0.0407 | **1.2× — NOT resolvable** |

The load-bearing conclusions survive at 7–12× the noise floor. **Two do not:**
any *ordering* between medium_play and large_diverse (§4.3.29) is noise — they
both reach stationarity, but neither is better than the other — and §4.3.27's
jitter-alone effect at 3× is suggestive rather than established.

**`cvar_ess` varied 1.9× between replicates**, so CVaR CE differences near
their jackknife SE (§4.3.22's resolution rule) need replication on top of that
SE, not instead of it. The §4.3.28 CVaR CE gap of 0.1433 against a 2·SE of
0.1278 was already marginal; it should be treated as unconfirmed until
replicated.

**Adopt `sampling_seed` replication as standing practice.** One replicate per
load-bearing configuration, at ~2× the compute, is what separates a real effect
from the two independent lotteries now measured — the warm-up endpoint and the
tail estimator. Nothing in §4.3.26–34 had that before this run.

### 4.3.36 Warm-up NLL corrects §4.3.34 — and predicts the burn-in effect

§4.3.34 read the warm-up on **accuracy**, which is quantized in units of 1/54
to 1/107 (§4.3.34). NLL is the same cross-entropy the run is selected on,
continuous, and far better resolved. The trajectories:

| run | burn-in | NLL across burn-in | min | final |
|---|---|---|---|---|
| medium_play `jit10n256` | 20,000 | 0.581 → 0.592 → 0.594 → 0.585 → **0.596** | 0.580 | 0.596 |
| medium_play `s100` | 20,000 | 0.873 → 0.867 → 0.851 → 0.839 → **0.828** | 0.818 | 0.828 |
| medium_diverse `recipe` | 20,000 | 0.996 → 0.585 → 0.481 → 0.436 → **0.352** | 0.328 | 0.352 |
| medium_diverse `burn100k` | 100,000 | 0.996 → 0.329 → 0.318 → 0.280 → **0.327** | 0.272 | 0.327 |
| large_play `recipe` | 20,000 | 0.715 → 0.603 → 0.557 → 0.436 → **0.470** | 0.385 | 0.470 |
| large_play `burn100k` | 100,000 | 0.715 → 0.411 → 0.339 → 0.366 → **0.426** | **0.172** | 0.426 |

**§4.3.34's "oscillates and never converges" was largely an accuracy artifact.**
On NLL, medium_diverse converges monotonically, medium_play is essentially
**flat** (0.581 → 0.596 over 20,000 steps — the warm-up barely moves), and only
large_play genuinely rises after its minimum. The claim holds for one variant of
three, not as a general property.

> **NLL predicts the SIGN of the burn-in effect, within-variant, with no
> confound.** This is the first diagnostic in §4.3.30–36 that does:
>
> | variant | NLL at end of a 20k burn-in | predicts | §4.3.33/34 observed |
> |---|---|---|---|
> | medium_diverse | still **falling** (0.436 → 0.352) | too short — more helps | 0.5299 → 0.9087 ✓ |
> | large_play | already **rising** (0.436 → 0.470, min 0.385) | too long — more hurts | 0.8578 → 0.6194 ✓ |
> | medium_play | **flat** | length barely matters | works at 20k ✓ |
>
> And `large_play burn100k` overshoots badly: min **0.172** at mid-burn-in
> against a final **0.426**. Three variants, three signs, all predicted.

**This replaces §4.3.33's hand-tuned burn-in with a rule.** Set
`num_burn_in_steps` by **early stopping on warm-up NLL** — stop where it stops
improving — rather than fixing a per-variant count. That removes one of the two
knobs §4.3.33 flagged as having no principled justification, and `warmup_log_every`
already records the curve, so nothing new needs building.

> **Warm-up quality does not predict final quality — at all.** The `s100`
> replicate's warm-up ended **at chance**: NLL **0.828**, above `log 2` =
> 0.6931, with accuracy 0.5195. Its final sampled posterior was nonetheless
> indistinguishable from the original's — val CE 0.2932 vs 0.2912, accuracy
> 0.8782 vs 0.8799, centred `ratio` 1.1198 vs 1.0871 (§4.3.35). **Sampling
> recovers completely from a warm-up that failed.**
>
> Across variants the relationship is if anything inverted: medium_play has the
> **worst** warm-up NLL of the three (0.596/0.828 against medium_diverse's 0.352)
> and the **best** final CE (0.2912). Warm-up NLL is a *burn-in-length*
> diagnostic, not a quality signal.
>
> **`early_stop_acc_threshold` is not implicated, and §3.5 got there first.**
> It is **0.0 in every `_antmaze_eval` config and in all 62 round-2 runs**, with
> `early_stopped = 0` on every one — it never fired, and no round-2 trial was
> discarded by it. The 0.98 value survives only in the round-1 configs
> (`scripts_bnn/antmaze_<v>_bnn.yaml`). **§3.5 already removed the gate in round
> 1 on the same grounds this section reaches independently** — "it rejected on
> the wrong quantity… warm-up accuracy, which is not the selection metric" —
> and identified a second failure mode this section did not: it became a hard
> wall in `mdecay`. Nothing needs re-examining; the sweep yaml pins it at 0.0
> with that reasoning recorded inline.

> **One refinement §4.3.35 does add to §3.5.** That section argues warm-up
> outcome "is a deterministic function of `mdecay` alone", explicitly premised
> on `seed` being fixed. The replicate breaks that premise and measures what it
> was holding constant: at **identical `mdecay` and identical data**, changing
> only `sampling_seed` moved warm-up accuracy **0.7273 → 0.5195** and warm-up
> NLL **0.596 → 0.828**. So the determinism is conditional on the seed, and the
> seed-to-seed spread is large — which means §3.5's pass/fail-by-`mdecay`
> thresholds are **not sharp boundaries**, and the perfect separation it
> reports would blur under replication. That does not weaken §3.5's conclusion
> (the gate is still gating on the wrong quantity), but the threshold table
> should not be read as locating a hard `mdecay` boundary.
>
> It also explains §4.3.31's confound rather than leaving it unresolved: warm-up
> accuracy and `mdecay` were never independent variables that happened to align
> across four points — §3.5 establishes warm-up outcome is *downstream* of
> `mdecay`. The n = 4 correlation was structural, not coincidental, and still
> carries no information about centred drift (§4.3.35).

### 4.3.37 The warm-up state is not the channel — the frozen preconditioner is

`warmup_use_best` on large_play, 20,000 burn-in, everything else as §4.3.30:

| | useBest | centred `ratio` | centred `scale_z` | mean CE | acc | warm-up NLL |
|---|---|---|---|---|---|---|
| `recipe` | — | 0.8578 | 1.0365 | 0.2545 | 0.9005 | 0.4701 (final) |
| `bestwu` | **True** | **0.8609** | 1.1368 | 0.2603 | 0.9005 | **0.3852** (best, step 17k) |

**A clean negative.** The flag engaged and handed the chains a state **22%
better in NLL** (0.3852 vs 0.4701). Centred `ratio` moved **0.003**, against a
resolution floor of **0.0327** (§4.3.35). Accuracy is identical to four decimal
places. **The warm-up state is not what makes large_play fail**, and the
best-vs-final gap I proposed as the mechanism explains nothing.

That is the **fifth** refuted mechanism in §4.3.30–37, and it confirms §4.3.36
from the opposite direction: that section showed a *worse* warm-up (the `s100`
replicate, at chance) costs nothing; this shows a *better* one buys nothing.

> **But it forces a reinterpretation of §4.3.33 and §4.3.34.** Burn-in length
> demonstrably matters — medium_diverse 0.5299 → 0.9087 and large_play 0.8578 →
> 0.6194, both ~7–12× the floor. If the **starting point** is not the channel,
> those effects must run through something else that burn-in length changes.
>
> **It does change something else, permanently.** `adaptive_sghmc.py:107` gates
> the preconditioner adaptation on `iteration <= num_burn_in_steps`: `tau`, `g`
> and `v_hat` adapt during burn-in and are **frozen for the entire sampling
> phase**. Burn-in length therefore fixes the preconditioner the sampler uses
> forever, independently of where the chain ends up.
>
> And the averaging window degenerates. `tau ← (1 − ĝ²/(v̂+λ))·tau + 1` grows
> **linearly** when the mean gradient is small relative to the second moment —
> which is exactly the regime near a mode:
>
> | `ĝ²/v̂` | `tau` at 20k | `tau` at 100k | `tau_inv` |
> |---|---|---|---|
> | 0 (at a mode) | 20,001 | 100,001 | 5.0e-5 → 1.0e-5 |
> | 0.01 | 100 | 100 | 1.0e-2 (saturates) |
> | 0.10 | 10 | 10 | 1.0e-1 (saturates) |
>
> So near a mode `v_hat` stops updating long before burn-in ends, and a 100k
> burn-in freezes a preconditioner averaged over a 5× longer and staler window
> than a 20k one. **That is a real, permanent difference between the two runs
> that has nothing to do with the starting point** — and it is the only channel
> left standing.

**This is a hypothesis, not a result**, and §4.3.30–37 is a graveyard of clean
four-point stories. What distinguishes it: it is a *mechanism in the code* with
a documented freeze point, it predicts burn-in effects without reference to the
start (which is now excluded by measurement), and it is checkable **without a
run** — instrument `tau`, `v_hat` and `minv_t` at the freeze point and compare
20k against 100k on the same variant.

**Do that before spending another run on burn-in.** If the frozen preconditioner
differs materially between the two, it explains §4.3.33 and §4.3.34 and the
lever is the adaptation window, not the burn-in length. If it does not, the
channel is something else again and burn-in should be left alone.

**Instrumentation added 2026-08-24.** `AdaptiveSGHMC.preconditioner_snapshot()`
summarises `tau`, `v_hat` and `minv_t`, and `FPrefNet.train` captures it at the
freeze boundary — the last step on which they change. It prints one line per
chain to the run log (`[precond] FROZEN at step …`) and the warm-up's copy is
logged to wandb as `precond_*`. No extra compute; it reads state the sampler
already holds.

The load-bearing number is **`precond_tau_over_burnin`**. `tau` saturates at
`v_hat/ĝ²` when the mean gradient is an appreciable fraction of the second
moment, and grows **linearly** when it is not — the regime near a mode.
Verified on the real optimiser with synthetic gradients:

| gradient | burn-in 20k | burn-in 100k | |
|---|---|---|---|
| zero-mean (at a mode) | `tau` 12,536, ratio **0.627** | `tau` 63,226, ratio **0.632** | never saturates |
| persistent mean | `tau` 1, ratio 0.000 | `tau` 1, ratio 0.000 | saturates immediately |

**A ratio that stays constant as `num_burn_in_steps` grows is the degenerate
case** — `tau` scaling with the burn-in means `v_hat` averages over all of it.
A ratio that falls means the window saturated and burn-in length should not
matter. One run gives the ratio; **two at different burn-in lengths give the
scaling, which is the actual discriminator** — so read this on the existing
large_play 20k and 100k configurations.

> **A note for §7.1.** Freezing the preconditioner after burn-in is Springenberg
> et al.'s design and §3.6.2 records it as faithful. But the consequence — that
> `num_burn_in_steps` silently sets a sampling-phase hyperparameter, and that
> `tau` grows without bound near a mode so the window is effectively the whole
> burn-in — is not something the reference discusses at this scale, and it makes
> burn-in length a **sampler** parameter rather than a warm-up convenience.

### 4.3.38 Audited against Springenberg et al. — the implementation is faithful; the step size is not

Read against the source paper (`NIPS-2016-…-robust-bayesian-neural-networks-Paper.pdf`,
§4.2 and Eqs. 8–10).

**The adaptation is implemented exactly as specified.** Paper Eq. (9)
`Δτ = −g²V̂⁻¹τ + 1` and `Δg = −τ⁻¹g + τ⁻¹∇Ũ`, Eq. (8)
`ΔV̂ = −τ⁻¹V̂ + τ⁻¹(∇Ũ)²` — all three match `adaptive_sghmc.py:114–129`
line for line. The freeze is the paper's own design: *"our estimation/adaptation
of the parameters only changes the HMC procedure during the burn-in phase. After
it, when actual samples are recorded, all parameters stay fixed."*

**And `mdecay` is the right object.** Paper Eq. (10)'s noise is
`2ε³V̂^{-1/2}CV̂^{-1/2} − ε⁴`; the code computes `2ε²·mdecay·minv_t − ε⁴`. These
agree exactly when `mdecay = εV̂^{-1/2}C`, which is precisely what the paper
holds constant: *"we chose C such that we have `εV̂^{-1/2}C = 0.05I`
(intuitively this corresponds to a constant decay in momentum of 0.05 per time
step)"*. **So `mdecay` is the paper's momentum-decay-per-step, and its
recommended value is 0.05.**

> **§4.3.37's framing needs softening.** `τ`'s unbounded growth near a mode is
> inherent to the *published* scheme — Eq. (9) has no upper bound either — so
> burn-in length setting the frozen preconditioner is a property of the
> reference algorithm, not a codebase defect. The instrumentation is still
> worth reading, but it is measuring an intended behaviour, and §7.1 should not
> describe it as a deviation.

**The correctness constraint is satisfied with enormous margin.** The paper
requires `min(V̂⁻¹)C ≥ ε` for unbiased sampling — equivalently, the Eq. (10)
noise variance staying non-negative. Measured across the four variants the
margin is **10⁵–10⁸×**, so the `clamp_(min=1e-16)` never binds and this is not
the problem.

> **What the audit does find: `sghmc_lr` is 40–130× below the paper's value.**
> Springenberg fixes **ε = 10⁻²** as *"a robust choice in our experience"*.
>
> | variant | `sghmc_lr` | × below 1e-2 | rel. diffusion/step | `mdecay` | vs 0.05 |
> |---|---|---|---|---|---|
> | medium_play | 2.49e-4 | 40× | 6.2e-4 | 0.1946 | 3.9× |
> | large_diverse | 7.57e-5 | 132× | 5.7e-5 | 0.3761 | 7.5× |
> | large_play | 1.42e-4 | 70× | 2.0e-4 | 0.0312 | 0.62× |
> | medium_diverse | 1.25e-4 | 80× | 1.6e-4 | 0.0072 | 0.14× |
>
> Per-step diffusion scales as ε², so **these chains explore 1,600–17,000×
> more slowly per step than the reference calibration.** That is a direct,
> quantitative candidate for `ess_bulk` ≈ 2.4% (§4.3.26), for transients that
> outlast a 20,000-step burn-in (§4.3.32), and for chains that never finish
> relaxing inside a 40–220k sampling window.

**And this is §4.3.14's pathology again, on a third parameter.** `sghmc_lr` is
swept and selected on validation CE. A chain that barely moves stays near the
well-fit warm-up point, which *scores well on CE* — so CE drives the step size
**down**, exactly as it drove `map_amp2` **up** (§4.3.16's cap history) and
`n_meas` down (§4.3.24). Three parameters, one mechanism: **mean CE rewards a
sampler that does not sample.**

**`mdecay` straddles the paper's 0.05, and the split is suggestive.** The two
variants that pass under the §4.3.28 recipe sit *above* it (3.9×, 7.5×); the
one that fails at every burn-in tried sits *below* (0.62×). Given §4.3.30–37's
record with four-point orderings this is not evidence — but it is the first such
ordering with an *external* reference point rather than an internal one.

**Next, and it supersedes §10.2 step 1's preconditioner check.** Run large_play
at the paper's calibration — `sghmc_lr` 1e-2, `mdecay` 0.05 — with the §4.3.28
recipe otherwise unchanged:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_large_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --sghmc_lr 0.01 --mdecay 0.05 \
    --warmup_use_best True \
    --OUT_DIR ./exp/stage3_large_play_paper > ../exp/stage3_large_play_paper.log 2>&1 &
```

Watch `param_clamp_sampling_pct` and `gradnorm_sampling_pct_over_clip`: a 40×
larger step size is exactly the regime where `max_param_step` and the gradient
clip start to bind, and §3.6.3 rejects any run whose clamp fires. If they fire,
that is informative rather than fatal — it locates the actual ceiling on ε for
this model, which the sweep never explored because CE was pulling the other way.

### 4.3.39 Audited against Tran et al. — every sampler default is theirs, and the sweep walked away from all of them

`tran.pdf` §A.3 (p. 34–35) is the immediate parent: this repo is a fork of that
codebase.

**The code implements Tran et al.'s Eq. (27), not Springenberg's Eq. (10)
directly.** Tran substitutes `εCV̂^{-1/2} = αI` into Springenberg's Eq. (25) to
get

    Δv = −ε²V̂^{-1/2}∇Ũ − αv + N(0, 2ε²α V̂^{-1/2} − ε⁴I)

which is `adaptive_sghmc.py:147` *verbatim*: `2·lr²·mdecay·minv_t − lr⁴`. So
**`mdecay` is Tran's momentum coefficient α**, and §4.3.38's reconciliation
via Springenberg was the long way round to the same place.

**Every sampler default in the codebase is Tran et al.'s configuration:**

| codebase default | value | Tran et al. §A.3 |
|---|---|---|
| `sghmc_lr` | 0.008 | ε = 0.01 |
| `mdecay` | **0.01** | α = **0.01** (exact) |
| `num_chains` | **4** | **4** (exact) |
| `num_burn_in_steps` | 3000 | 2000–5000 |
| `keep_every` | **2000** | thinning **2000**–10,000 |
| `num_samples` | 50 | 30–200 |

**And the sweep moved `sghmc_lr` 32–106× below it:**

| winner | `sghmc_lr` | × below the 0.008 default | `mdecay` | thinning | vs Tran's 2000 floor |
|---|---|---|---|---|---|
| medium_play | 2.49e-4 | **32×** | 0.1946 | 2750 | 1.38× |
| large_diverse | 7.57e-5 | **106×** | 0.3761 | 2750 | 1.38× |
| large_play | 1.42e-4 | **56×** | 0.0312 | **500** | **0.25×** |
| medium_diverse | 1.25e-4 | **64×** | 0.0072 | **750** | **0.38×** |

> **This is the sharpest form of §4.3.14 yet: the inherited default was already
> the published value, and CE selection walked 32–106× away from it.** Not a
> parameter nobody had calibrated — a parameter calibrated twice, by two papers,
> agreeing with each other, sitting in the config as the default. On `map_amp2`
> and `n_meas` CE chased a boundary; here it walked away from a known-good
> point.

**Two more alignments, both external.** The two variants that fail under the
§4.3.28 recipe are the two that thin **below Tran's 2000 floor** (500 and 750),
which gives §4.3.32's `cycle_length` observation a published reference rather
than an internal four-point ordering. And `mdecay` moved the *other* way in the
variants that pass — 0.195 and 0.376 against the 0.01 default, 20–38× above.

**That opposite movement is mechanically coherent, and it identifies α as a
compensator.** In Eq. (27) the injected noise is `2ε²α V̂^{-1/2}`, so its
standard deviation scales as `ε√α`. An ε that is ~100× too small costs 100× in
noise; raising α by 20–38× recovers only `√20`–`√38` ≈ **4.5–6×** of it.
**CE selected α upward to partially offset an ε it had itself driven down** —
and the two variants where that compensation is largest are the two that pass.

**Temperature is not a confound.** `temperature` defaults to 1.0 and no
production config sets it, so these are untempered posteriors. Tran et al. grid-
searched cold posteriors down to T = 1e-4 (§A.4); this project does not temper
at all, which is a deviation in the opposite direction and a deliberate one.

**Revised next run — restore the inherited defaults rather than mixing papers.**
§4.3.38 proposed Springenberg's α = 0.05; Tran's α = 0.01 is both the codebase
default and the value matching the implemented equation, so it is the more
defensible restoration:

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_large_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --warmup_use_best True \
    --sghmc_lr 0.008 --mdecay 0.01 --cycle_length 2000 \
    --OUT_DIR ./exp/stage3_large_play_trandefaults > ../exp/stage3_large_play_trandefaults.log 2>&1 &
```

Three changes, but one hypothesis: **the sampler settings the fork inherited
were right, and CE selection is what broke them.** `cycle_length` 2000 makes
the run 4× longer than large_play's current 500 (160k vs 40k sampling steps),
which is itself part of the restoration.

**Watch `param_clamp_sampling_pct` and `gradnorm_sampling_pct_over_clip`.** A
56× larger step is exactly where `max_param_step` binds, and §3.6.3 rejects any
run whose clamp fires during sampling. If it fires, that locates the real
ceiling on ε for this model — which the sweep never probed, because CE was
pulling the other way the whole time.

### 4.3.40 The reference step size does not transfer — and a constant `f` passes the gate

large_play at the inherited Tran defaults (`sghmc_lr` 0.008, `mdecay` 0.01,
`cycle_length` 2000):

| | `sghmc_lr` | centred `ratio` | centred `scale_z` | mean CE | acc | `clamp%` | `clip%` |
|---|---|---|---|---|---|---|---|
| `recipe` | 1.42e-4 | 0.8578 | 1.0365 | 0.2545 | 0.9005 | 0.0000 | 0.0003 |
| **`trandefaults`** | **8.0e-3** | **1.0000** | **0.0000** | **0.6932** | **0.4606** | **0.0076** | **1.9933** |

**The chain collapsed to a constant function.** Centred `ratio` is *exactly*
1.0000 and both centred z-scores are *exactly* 0.0000 — the signature of an
identically-zero centred component. Mean CE is **0.6932** against `log 2` =
0.6931, and accuracy 0.4606 is below chance: `Φ₁ = Φ₂` for every pair, so every
prediction is 0.5.

> **And it passed the §3.6.3 gate perfectly.** `scale_z` 0.0000 ≤ 2.0 and
> `loc_z` 0.0000 ≤ 2.0, with `param_clamp_sampling_pct` 0.0076 ≤ 0.01. **The
> most useless run in this entire investigation satisfies every eligibility
> criterion.** This is the third and most extreme instance of §4.3.15's rule —
> a constant `f` is not merely stationary, it is *perfectly* stationary, because
> there is nothing left to drift.
>
> **Guard added 2026-08-24.** `function_space_drift` now returns
> `fn_drift_shape_var_frac` — the fraction of `f`'s variance in the identified
> component — and prints a `*** DEGENERATE ***` banner below 1e-4. Verified:
> a healthy chain gives 0.9963, a constant `f` gives 2.5e-32 while its centred
> `scale_z` reads 0.0001. **This must gate alongside the drift criteria in the
> sweep redesign**, or a collapsed run will be selected.

**Why it collapsed.** The preconditioner caps `minv_t` at `1/√v_hat_min` = 100,
so the per-step displacement is bounded by `ε²·minv_t`: **6.4e-3 at
ε = 0.008 against 2.0e-6 at the selected 1.42e-4 — 3,174× larger.** The
instability markers fire for the first time in the investigation:
`param_clamp_sampling_pct` 0.0076 where every previous run was *exactly* 0.0000,
and `gradnorm_pct_over_clip` 1.9933 against 0.0003–0.0025.

> **This revises §4.3.38 and §4.3.39.** I claimed CE selection "walked away from
> a known-good value". The reference ε **does not transfer to this model**, so
> CE's small step size was at least partly tracking a real stability ceiling,
> not purely pathology. The provenance findings stand — the defaults *are*
> Tran's, and `sghmc_lr` *was* moved 32–106× — but "the inherited settings were
> right and CE broke them" is refuted as stated.

> **My run design was poor and I could not attribute the failure.** I changed
> three parameters at once and called it "one hypothesis". `mdecay` went *down*
> (0.0312 → 0.01, reducing noise) and `cycle_length` only thins further, so ε is
> the prime suspect by elimination — but this run cannot prove it. **Ladder ε
> alone.**

**Next: find the actual ceiling on ε**, holding everything else at the §4.3.28
recipe:

```
for LR in 5e-4 1.5e-3 4e-3; do
  cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
      --config_path scripts_bnn/antmaze_large_play_bnn_antmaze_eval.yaml \
      --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
      --chain_init_jitter 1.0 --n_meas 256 --warmup_use_best True --sghmc_lr $LR \
      --OUT_DIR ./exp/stage3_large_play_lr$LR > ../exp/stage3_large_play_lr$LR.log 2>&1
done
```

Sequential, since each needs 4 GPUs. Read **`fn_drift_shape_var_frac` first** —
below 1e-4 the run is degenerate and nothing else in it means anything — then
`param_clamp_sampling_pct` (§3.6.3 rejects above 0.01), then centred `ratio` and
CVaR CE against large_play's 0.8578 and 3.0359. The ceiling is the largest ε
that keeps `shape_var_frac` healthy and the clamp at zero; whether anything
below that ceiling also fixes the contraction is the actual open question.

### 4.3.41 The ε ladder — the selected step size is already at the ceiling

large_play, §4.3.28 recipe, `sghmc_lr` varied alone:

| `sghmc_lr` | × selected | `shape_var_frac` | `clip%` | centred `ratio` | centred `scale_z` | mean CE | acc |
|---|---|---|---|---|---|---|---|
| **1.42e-4** (selected) | 1.0× | — | 0.00 | **0.8578** | **1.0365** | **0.2545** | **0.9005** |
| 5.0e-4 | 3.5× | 0.6997 | 60.39 | 0.5542 | 2.8312 | **0.7535** | 0.7512 |
| 1.5e-3 | 10.6× | 0.0972 | 100.00 | 0.3920 | 16.6166 | 0.5351 | 0.7176 |
| 4.0e-3 | 28.2× | 0.0805 | 13.65 | 0.1609 | 35.3081 | 0.5788 | 0.7106 |
| 8.0e-3 | 56.3× | — | 1.99 | 1.0000 | 0.0000 | **0.6932** | 0.4606 |

**The ceiling is below 5e-4 — only 3.5× above the CE-selected value — and mean
CE is already worse than `log 2` there.** Every rung is worse than the selected
step size on every axis. The ladder found no better setting; it found that
there is no headroom.

> **§4.3.38 and §4.3.39 are now fully refuted, and §4.3.40's revision is
> confirmed with a ladder rather than a single point.** `sghmc_lr` was **not**
> driven down by CE pathology away from a known-good value. It is pinned near a
> hard stability ceiling, and CE selection was tracking a real constraint. The
> provenance findings survive as provenance — the defaults are Tran's, and the
> selected values are 32–106× below them — but the *interpretation* that the
> inherited settings were right for this model is dead. **The reference
> calibration does not transfer, and this model tolerates roughly 1/50th of
> their step size.**

**The posterior is stiff, and that is the actual finding.** `clip%` — the
fraction of sampling steps whose gradient norm exceeds 100 — goes
**0.0003 → 60.4 → 100.0** over a 10× rise in ε. Gradient norms grow
super-linearly with the step size, which is the signature of a chain leaving a
well-conditioned basin as soon as it takes larger steps. `shape_var_frac` tracks
the resulting collapse of `f` toward a constant: **0.6997 → 0.0972 → 0.0805**.

> **Consequence, and it redirects the whole sampler-repair effort.** `ess_bulk`
> ≈ 2.4% (§4.3.26) **cannot be fixed by raising ε** — ε is at its ceiling. The
> slow mixing is a consequence of a stiff, ill-conditioned posterior, not of a
> mis-set hyperparameter. The remaining levers are of a different kind:
> **better preconditioning** (the diagonal `V̂^{-1/2}` may simply be inadequate
> here, where Springenberg's own §4.2 notes the full Fisher is what you would
> want), **a different sampler**, or **reducing the stiffness** at its source in
> the model or prior. Tuning the existing sampler's scalars is exhausted.

**The degeneracy guard works and earns its place.** `fn_drift_shape_var_frac`
flags every collapsed rung (0.097, 0.081) while the drift statistics on those
same runs look progressively *better* as the model degrades — `cvar_ess` reads
**858 and 1255** at 1.5e-3 and 4e-3, against 58 on the healthy run. **High
`cvar_ess` with low `shape_var_frac` is a degeneracy signature, not a good
tail**: a near-constant `f` has trivially high effective sample size. Read the
guard before any tail metric.

**large_play is unresolved and stays that way.** Its centred `ratio` 0.8578 is
the best available under any ε tried, and it still contracts. §10.2's step 1
should stop treating that as a tuning problem.

### 4.3.42 The stiffness is the prior's Gram conditioning — and two scalars set it

§4.3.41 left "better preconditioning" as a lever. Before building one, the
prior gradient is worth looking at, because it is `K⁻¹(f − m)` and **cond(K) is
the dynamic range that gradient spans**.

`map_informed_prior.py:163` gives the structure:

    K / amp2 = sig_c2·J + sig_g2·K_geo + sig_n2·I

with `sig_c2 = 1.0`, `sig_n2 = 0.001` in every variant. `J` is rank-1 with
eigenvalue exactly `n`, and the nugget floors the spectrum at `sig_n2`, so

    cond(K) ≥ n · sig_c2 / sig_n2

| `n_meas` | cond(K) ≥ |
|---|---|
| 7 (large_diverse selected) | 7.0e3 |
| 29 (large_play selected) | 2.9e4 |
| 256 (§4.3.28 recipe) | **2.6e5** |

**Two consequences, both new.**

**The condition number grows linearly in `n_meas`.** The §4.3.28 recipe makes
the Gram **8.8× more ill-conditioned** on large_play (29 → 256) and **36.6×**
on large_diverse (7 → 256). §4.3.24 attributed `n_meas`'s effect to prior
*coverage*; that stands, but `n_meas` also trades conditioning against
coverage, and only the coverage half was measured.

**`amp2` is irrelevant to it.** The amplitude multiplies the whole matrix,
nugget included, so cond is amplitude-invariant — this is orthogonal to
everything in §4.3.16–23 and could not have been found by any amplitude ladder.

> **The dominant term is the one direction the likelihood cannot see.**
> `sig_c2·J` is the constant-offset component — precisely the offset §4.3.10
> proved is unidentified, cancels in every preference prediction, and leaves the
> IQL greedy policy unchanged. It contributes an eigenvalue of `n·sig_c2`,
> which at `n_meas` 256 is **256 against `K_geo`'s O(1)**. So the single largest
> contributor to the stiffness that pins ε is prior mass on a direction that
> does not affect any prediction.

**Two scalars control it, and neither has ever been varied.** Both are
design-fixed (§3.6.2 lists `map_sig_*` as "design-fixed from geometry"), not
swept:

| lever | effect on cond | modelling cost |
|---|---|---|
| **`map_sig_c2` ↓** (1.0 → 0.01) | **÷72** at n=256 (2.28e5 → 3.16e3) | frees the *unidentified* offset — nothing the likelihood sees |
| `map_sig_n2` ↑ (0.001 → 0.05) | ÷45 at n=256 (2.28e5 → 5.12e3) | weakens the fine-scale prior — a real cost |

Verified numerically against a PSD stand-in for `K_geo`; the structural bound
holds within a factor of ~3.

**`map_sig_c2` is the better lever**, and the argument is the same one that
justified §3.6.3's amendment: the offset is unidentified, so prior mass on it
buys nothing and costs conditioning. The risk is the mirror image — with the
offset less constrained it can wander further (§4.3.28 already measured offset
`ratio` 1.6454 on medium_play), and an unbounded offset could saturate the
network numerically. **That is what the run has to check**, and the centred
gate plus `fn_drift_shape_var_frac` are exactly the instruments for it.

**Instrumentation added.** `_gram_from_idx` now logs `cond(K)` once per process
with λ_min/λ_max — one eigendecomposition, not one per step, since the spectrum
depends on the kernel and the draw rather than on `w`. It turns the bound above
into a measurement.

**The run**, on large_play, which fails at every ε (§4.3.41):

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_large_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --warmup_use_best True --map_sig_c2 0.01 \
    --OUT_DIR ./exp/stage3_large_play_sigc2 > ../exp/stage3_large_play_sigc2.log 2>&1 &
```

Read `[prior] Gram cond(K)` in the log first to confirm the drop, then
`fn_drift_shape_var_frac` (degeneracy), then centred `ratio` against **0.8578**
and CVaR CE against **3.0359**. If a better-conditioned prior lets large_play
mix at the ε it already tolerates, the stiffness diagnosis is right and
**preconditioning does not need rebuilding** — the conditioning was fixable at
source, which §4.3.41 listed as the third lever and which is far cheaper than
the first.

### 4.3.43 `sig_c2` does nothing — the stiffness is at the *bottom* of the spectrum

`map_sig_c2` 1.0 → 0.01 on large_play, everything else as §4.3.28:

| | `sig_c2` | `shape_var_frac` | centred `ratio` | centred `scale_z` | offset `scale_z` | mean CE | acc |
|---|---|---|---|---|---|---|---|
| `recipe` | 1.0 | — | 0.8578 | 1.0365 | 0.6885 | 0.2545 | 0.9005 |
| `bestwu` | 1.0 | — | 0.8609 | 1.1368 | 0.4772 | 0.2603 | 0.9005 |
| `sigc2` | **0.01** | 0.5573 | **0.8539** | 1.2240 | **0.1805** | 0.2578 | 0.8981 |

Centred `ratio` moved **0.004** against a 0.0327 floor. No effect. (The offset
became *more* stationary, 0.6885 → 0.1805, the opposite of the risk I flagged —
but offset metrics vary ~2× between replicates, so that is not readable either.)

**Computing the actual spectrum shows why, and my §4.3.42 analysis was wrong in
its target.** The large maze has **33 free cells**, and the kernel is
*cell-based* — two measurement points in the same cell get identical rows.
Measured on the real prior and pool:

| `n_meas` | distinct cells | λ_max | λ_min | cond(K) | eigenvalues at the nugget |
|---|---|---|---|---|---|
| 7 | 6 | 7.72 | 1e-3 | 7.7e3 | 1 |
| 29 (large_play selected) | 14 | 30.79 | 1e-3 | 3.1e4 | 15 |
| 35 | 16 | 37.42 | 1e-3 | 3.7e4 | 19 |
| **256** (recipe) | **25** | 269.15 | 1e-3 | **2.7e5** | **231** |

**231 of 256 eigenvalues sit exactly at the nugget.** Those are the *within-cell
difference* directions: the prior asserting that `f` is equal at all points
sharing a maze cell, enforced with weight `1/sig_n2 = 1000`. **That** is the
stiffness.

> **`sig_c2` only moves λ_max.** Measured, it cuts cond 14.7× (2.69e5 → 1.84e4
> at n=256) — real, and less than the 72× I projected from a synthetic
> `K_geo` — but it does nothing to the 231 directions at the floor. I aimed at
> the top of the spectrum when the stiffness is at the bottom. The offset
> argument that motivated it was sound and irrelevant.

**Two corrected levers, both measured:**

| lever | effect | cost |
|---|---|---|
| **`map_sig_n2` 0.001 → 0.05** | raises λ_min 50×, cond → ~5.4e3, and directly softens the within-cell constraint | weakens the cell-equality prior — a real modelling choice |
| **`n_meas` ≈ 33** | 35 points already reach 16 cells at cond 3.7e4 vs 256's 25 cells at 2.7e5 — **7.2× better conditioning for 1.6× less coverage** | mild coverage loss |

> **And §4.3.24's coverage account needs a ceiling.** Coverage cannot exceed
> **33 cells**, and `n_meas` 256 reaches only **25** — so the recipe's 7.3×
> increase in measurement points buys 1.6× more cells while making the Gram
> 7.2× more ill-conditioned. `n_meas` is not a free "more is better" knob; it
> should be set relative to the free-cell count, which is a property of the maze
> and computable without any run (`maze_layouts.get_antmaze_layout`).

**Next: the nugget, which is the direction the spectrum actually points.**

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_large_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --warmup_use_best True --map_sig_n2 0.05 \
    --OUT_DIR ./exp/stage3_large_play_nugget > ../exp/stage3_large_play_nugget.log 2>&1 &
```

`[prior] Gram cond(K)` should read ~5e3 against the current 2.7e5. Then
`fn_drift_shape_var_frac`, then centred `ratio` against **0.8578**.

**If this also does nothing, stiffness is not the mechanism** and §4.3.41's
remaining levers reduce to rebuilding the preconditioner or changing sampler —
both large. Two negative conditioning results in a row would be reason to stop
and disclose large_play's contraction under §7.1 rather than keep spending.

### 4.3.44 The nugget works — all four variants now pass

`map_sig_n2` 0.001 → 0.05 on large_play, everything else as §4.3.28:

| metric | `recipe` | **`nugget`** | |
|---|---|---|---|
| centred `ratio` | 0.8578 | **0.9231** | +0.0653 = **2.0× the floor** |
| centred `scale_z` | 1.0365 | 1.0870 | |
| `shape_var_frac` | — | **0.9015** | healthiest of any large_play run |
| mean CE | 0.2545 | **0.2130** | best of any large_play run |
| accuracy | 0.9005 | 0.9028 | |
| `cvar_ess` | 58.04 | **583.3** | 10× |

**The first change to improve large_play on every axis at once**, and the
`ratio` gain is resolvable at 2× the §4.3.35 replicate floor.

> **The conditioning hypothesis is confirmed — but only at λ_min.** §4.3.42's
> `sig_c2` targeted λ_max, **one** eigendirection, and did nothing (§4.3.43).
> `sig_n2` targets λ_min, where **231 of 256** directions sit, and works. That
> distinction is the whole content of the result: the stiffness is the prior's
> within-cell equality constraint, enforced at weight `1/sig_n2`, and softening
> it is what let the chain mix. Two attempts at "better conditioning" that
> differ only in *which end of the spectrum* they move, with opposite outcomes.

**`cvar_ess` 583 is genuine, not degeneracy** — `shape_var_frac` 0.9015, against
the collapsed `lr 1.5e-3` run's 858 at 0.0972. The guard separates them, as it
was built to.

**All four variants now clear the amended §3.6.3 gate:**

| variant | centred `ratio` | centred `scale_z` | configuration |
|---|---|---|---|
| medium_play | 1.0871 | 0.6469 | recipe |
| large_diverse | 1.1278 | 0.8091 | recipe |
| medium_diverse | 0.9087 | 1.7976 | recipe + burn-in 100k |
| large_play | 0.9231 | 1.0870 | recipe + `sig_n2` 0.05 |

> **But they pass under four *different* configurations**, which is §4.3.33's
> concern grown from four hand-tuned knobs to five. **Test `sig_n2` 0.05
> uniformly before adopting any of this.** The argument for it is not merely
> that it worked here: **0.001 was never a modelling choice.** The config
> documents it as "nugget / diagonal jitter (mandatory for invertibility)" —
> chosen to make the Cholesky succeed, not to express a belief about
> within-cell reward variation. A value chosen for numerical convenience has
> been setting the sampler's stiffness, and 0.05 is at least as defensible on
> modelling grounds.

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 16893.982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --warmup_use_best True --map_sig_n2 0.05 \
    --OUT_DIR ./exp/stage3_medium_play_nugget > ../exp/stage3_medium_play_nugget.log 2>&1 &
```

Against medium_play's flagship **1.0871 / 0.6469 / CE 0.2912**, replicated at
1.1198 / 0.6999 (§4.3.35). Neutral-or-better makes `sig_n2` 0.05 a uniform
setting and collapses two of the five per-variant knobs; materially worse means
it is a large_play-specific fix and the per-variant tuning stands as a
disclosure.

**Also worth re-testing once `sig_n2` settles: `n_meas`.** §4.3.43 measured
coverage saturating at 33 cells with `n_meas` 256 reaching only 25, at 7.2×
worse conditioning than `n_meas` 35. If a larger nugget removes the conditioning
penalty, `n_meas` could come back down toward the cell count with no loss —
which would collapse a third knob.

### 4.3.45 Uniform `sig_n2` is refuted — it does two things, not one

`map_sig_n2` 0.05 on medium_play, against its flagship:

| metric | `jit10n256` | `nugget` | |
|---|---|---|---|
| centred `ratio` | 1.0871 | **1.3260** | **worse**, 7× the floor |
| centred `scale_z` | 0.6469 | 1.4930 | worse |
| `shape_var_frac` | — | **0.4510** | vs large_play's 0.9015 |
| mean CE | 0.2912 | **0.2205** | better |
| accuracy | 0.8799 | **0.9115** | better |
| `cvar_ess` | 25.19 | 47.36 | better |

**Opposite to large_play, where it improved every axis.** Uniform adoption is
refuted; `sig_n2` 0.05 is not a global setting.

**And conditioning cannot explain the difference.** The two mazes are
structurally near-identical:

| maze | free cells | `n_meas` 256 → cells | pts/cell | cond(K) | at nugget |
|---|---|---|---|---|---|
| medium | 26 | 21 | 12.2 | **2.69e5** | 235 |
| large | 33 | 26 | 9.8 | **2.69e5** | 230 |

Same condition number to three figures. A conditioning-only account predicts the
same response and we measured opposite ones.

> **The resolution: `sig_n2` changes two things at once.** Raising it adds
> `0.049·I` to `K`, which (a) lifts λ_min 50× — better conditioning — and (b)
> shrinks `K⁻¹` in *every* direction, i.e. **uniformly weakens the prior**.
> Effect (b) is the same trade §4.3.16 measured on `map_amp2`: a weaker prior
> gives worse stationarity and better CE. **medium_play's response is exactly
> that trade** (`ratio` 1.0871 → 1.3260, CE 0.2912 → 0.2205), with
> `shape_var_frac` 0.4510 confirming the mechanism — more than half of `f`'s
> variance has moved into the offset, which is what a weaker prior permits.
>
> On large_play, conditioning dominated; on medium_play, prior strength does.
> **This is mechanistic rather than another four-point ordering** — the two
> effects are separable in the algebra, and each has an independent precedent
> (§4.3.42–44 for conditioning, §4.3.16 for prior strength).

**Which suggests decoupling them, and the levers are already characterised.**
`amp2` multiplies the entire matrix including the nugget, so it changes prior
strength **without touching conditioning** (§4.3.42). So:

    sig_n2 ↑  (better conditioning, weaker prior)
    amp2   ↓  (stronger prior, conditioning unchanged)

gives better conditioning at constant prior strength — the thing neither knob
delivers alone.

```
cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_bnn_training_antmaze_eval.py \
    --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
    --seed 0 --num_chains 16 --chains_per_gpu 4 --map_amp2 1689.3982289052463 \
    --chain_init_jitter 1.0 --n_meas 256 --warmup_use_best True --map_sig_n2 0.05 \
    --OUT_DIR ./exp/stage3_medium_play_decouple > ../exp/stage3_medium_play_decouple.log 2>&1 &
```

`map_amp2` 1.69e3 is not arbitrary — §4.3.23 measured it as the **CVaR CE
optimum** on this variant, one decade below the derived value, and §4.3.23's
open tension was whether a sampler fix would move that optimum back up. This
run tests the decoupling and that tension together.

> **Stopping rule, and I would hold to it.** All four variants already pass the
> amended gate (§4.3.44). This is a run to *collapse knobs*, not to reach a
> passing configuration. **If it does not produce a setting that works on both
> medium_play and large_play, stop optimising**: accept per-variant sampler
> settings, record them as tuned-on-diagnostics rather than selected by the
> pre-registered procedure, and move to §10.2 step 2 (replicate the four final
> configurations) and step 3 (the sweep redesign). Per-problem MCMC tuning is
> normal practice; what would not be defensible is presenting it as though the
> selection procedure produced it.

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
- **The pre-registered stationarity criterion was computed on a quantity
  containing an unidentified direction, and was amended mid-project.** §3.6.3
  gated on raw `fn_drift_*`, but the BT/CE likelihood is exactly invariant to
  `f → f + c`, so raw drift mixes the identified shape with an offset that
  cancels in every preference prediction (§4.3.10). The criterion therefore
  rejected samplers stationary in the part that matters and admitted ones whose
  offset was merely pinned by an over-tight prior — the same pathology that
  drove `map_amp2` to improperness (§4.3.14). Amended 2026-08-24 to gate on the
  centred component (§3.6.3). **Report that round-2's BNN winners were selected
  under the unamended criterion**, that they cannot be re-adjudicated because no
  sweep trial's chains survive (§4.3), and that the amendment governs the
  redesigned sweep rather than a re-scoring of round 2. The generalisable point:
  a stationarity diagnostic must be computed on the identified component, or it
  measures partly a direction the likelihood cannot see.
- **A degenerate reward model scores a PERFECT stationarity gate.** A run at
  the reference step size collapsed `f` to a constant: predictive CE 0.6932
  against `log 2` = 0.6931, accuracy below chance — and centred `loc_z` and
  `scale_z` both *exactly* 0.0000, because a constant function has nothing to
  drift (§4.3.40). It satisfied every §3.6.3 criterion. Report that eligibility
  was therefore made conditional on a **degeneracy check**
  (`fn_drift_shape_var_frac`, the fraction of `f`'s variance in the identified
  component) in addition to the drift criteria. The generalisable point: a
  convergence diagnostic is a ratio, and ratios are undefined-in-the-limit
  rather than merely noisy when the quantity being diagnosed vanishes.
- **A stationarity gate cannot detect a collapsed posterior.** `large_play`
  under the §4.3.28 recipe passes both the raw and centred criteria
  (`scale_z` 0.7649 / 1.0365) while its CVaR reward predicts preferences at
  **0.5741 accuracy and CVaR CE 3.0359** — worse than chance, `log 2` = 0.6931.
  A chain that has contracted onto too small a region is stationary in the
  trivial sense, and no drift statistic distinguishes that from correct
  sampling. Report that stationarity was verified **jointly** with a predictive
  check on the deployed quantity (CVaR CE), never on drift diagnostics alone —
  and that in this project the drift gate alone would have certified a
  configuration whose reward model is uninformative.
- **A gate PASS at the chain count you happened to run is not stationarity.**
  The converse of the bullet below, and it bit in the other direction: the
  final medium_play configuration passes every §4.2 gate at 16 chains while
  its centred `scale_ratio` is 1.3490 — a 35% variance growth between chain
  halves — and projects to a FAIL at 32 chains (§4.3.18). Report the effect
  size, which is chain-count invariant, alongside any gate verdict, and do
  not describe the shipped sampler as stationary. It is not; the drift was
  reduced from 1.6026 and is expected to plateau near 1.19 (§4.3.17).
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
| 3 | BNN | **halted at `c16`, by result** — medium_play `c4`/`c8`/`c16` measured (§4.3.1, §4.3.2), plus the half-split (§4.3.3), the per-chain drift (§4.3.5) and the non-cyclical control (§4.3.6). The ladder's axis is orthogonal to the binding constraint: a drift common to 14/16 chains that shrinks with neither draws nor chains. No budget selected; `c32` is not to be run. The cyclical schedule is cleared (§4.3.6) and the shared start is refuted (§4.3.8). **Closed as a negative result (§4.3.13).** The location drift is largely the likelihood-invariant offset and §4.3.2's headline does not survive correction (§4.3.11). The live defect is a widening of the identified shape that grows as `t^0.4` — scale-free, so no budget fixes it. Both levers are measured and neither works: doubling the draws gave +4% ESS and *lower* CVaR ESS (§4.3.12). **Superseded by §4.3.14: stage 3 cannot be completed until stage 1 is redone.** The paper's claim is CVaR, so the mean-based fallback is unavailable. Root cause is the selection objective, not the sampler: CE improves monotonically as the functional prior flattens, so `map_amp2` chases its cap (99.5% of range for large_play, third round running) and `n_meas` sits at 7–35 of 0–64. The resulting target has an equilibration time ~10²–10³× any feasible budget |
| 3b | BNN | **all four pass (§4.3.44), but under four different configurations.** Uniform `sig_n2` is refuted (§4.3.45): it improves large_play on every axis and *worsens* medium_play's stationarity, because it does two things — better conditioning **and** a uniformly weaker prior. One decoupling run left (`sig_n2` 0.05 + `amp2` 1.69e3); **if it fails, stop optimising** and move to replication + the sweep redesign |
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
remaining hypotheses was inconclusive at 8 chains (§4.3.7) and **refuted the
shared start** when redone at 16 (§4.3.8): ALIGNMENT held at 0.7216 against
0.7564, where the hypothesis predicted a collapse toward 0.25. Weight-space
diffusion could not be tested across chains — ‖w‖ growth is 1.51× in every
chain to ±1%, leaving no leverage (§4.3.9). That test did reveal that most of
the drift is a shift in the *global offset* of `f`, which the BT/CE likelihood
is exactly invariant to. Splitting the gate (§4.3.10, §4.3.11) showed the centred
location PASSES and, redone on centred `loc_sd`, §4.3.2's headline does not
survive — the identified location drift *does* shrink with chains. The
remaining defect is a widening of the identified component, and `d150`
(§4.3.12) settled it: the spread grows as a **scale-free power law `t^0.4`**,
so no horizon fixes it, and doubling the draws bought +4% bulk ESS and a CVaR
ESS that went *down*. **Both of §4.1's levers are exhausted; stage 3 closes
without a budget (§4.3.13).**

**The sampler repair is now the work (§4.3.13's fork, resolved).** CVaR is the
paper's mechanism, no theory makes the posterior *mean* a differentiator under
reduced data or label noise, and a mean-only fallback would be a different and
much weaker paper. So the sampler is fixed rather than worked around.

**Judge every candidate fix on centred `ratio` AND predictive CE together.**
§4.3.15 is the reason: a `bt_pool="sum"` run drove the widening from 1.6026 to
1.0661 and passed every §4.2 gate while *doubling* CE, because ~50× larger
effective steps equilibrate fast onto the wrong measure. Stationarity is
necessary, not sufficient, and centred `ratio` on its own can be bought by
breaking the sampler.

**Do this next, in order.** Rewritten 2026-08-24 against §4.3.26–36; the
previous list was reorganised around a gradient-noise fix that §4.3.26 voided,
and its step 1 had accumulated four layers of correction.

**Where the sampler stands.** The recipe — `map_amp2` **16893.98**,
`chain_init_jitter` **1.0**, `n_meas` **256**, at 16 chains — reaches centred
stationarity on three of four variants:

| variant | burn-in | centred `ratio` | centred `scale_z` | status |
|---|---|---|---|---|
| medium_play | 20,000 | 1.0871 / 1.1198 | 0.6469 / 0.6999 | PASS, **replicated** (§4.3.35) |
| large_diverse | 20,000 | 1.1278 | 0.8091 | PASS, unreplicated |
| medium_diverse | 100,000 | 0.9087 | 1.7976 | PASS, marginal (§4.3.33) |
| **large_play** | — | 0.8578 @20k, 0.6194 @100k | — | **FAILS both ways** (§4.3.30, §4.3.34) |

Resolution floor on centred `ratio` is **0.0327** (§4.3.35): differences below
~0.05 are not readable from single runs.

1. **~~Re-run large_play with `warmup_use_best`.~~ DONE — negative (§4.3.37).**
   Handing over a 22% better warm-up moved centred `ratio` by 0.003 against a
   0.0327 floor. The warm-up state is not the channel, and large_play still
   fails. **The remaining candidate is the frozen preconditioner**: `tau`, `g`
   and `v_hat` adapt only during burn-in and are frozen for all of sampling
   (`adaptive_sghmc.py:107`), so `num_burn_in_steps` silently sets a
   sampling-phase hyperparameter. **Superseded by §4.3.38** — the audit against Springenberg et al. found
   `sghmc_lr` is **40–130× below the paper's ε = 1e-2**, giving 1,600–17,000×
   less diffusion per step, which is a direct candidate for the slow relaxation
   the preconditioner hypothesis was invented to explain. It also found `τ`'s
   unbounded growth is the *published* algorithm's behaviour, not a defect. **Done, and it collapsed (§4.3.40).**
   `sghmc_lr` 0.008 drove `f` to a CONSTANT: mean CE 0.6932 = `log 2`,
   accuracy 0.4606, and a *perfect* stationarity gate (centred `scale_z`
   0.0000) because a constant function cannot drift. The reference ε does not
   transfer to this model. **Ladder done (§4.3.41): the selected ε is
   already at the ceiling.** 5e-4 — only 3.5× up — already gives mean CE worse
   than `log 2`, and every rung is worse than the selected value on every
   axis. `sghmc_lr` was pinned by stability, not by CE pathology. **The
   posterior is stiff and `ess_bulk` ≈ 2.4% cannot be fixed by tuning ε.**
   Remaining levers are of a different kind — better preconditioning, a
   different sampler, or reducing stiffness at source — so **stop tuning the
   existing sampler's scalars**. Gate on `fn_drift_shape_var_frac` regardless. The preconditioner
   instrumentation is in place (§4.3.37): `precond_*` is logged to wandb from the warm-up and printed per
   chain to the run log. Read `precond_tau_over_burnin` on large_play at 20k
   and 100k — a ratio that stays constant means the adaptation window never
   saturated and a longer burn-in freezes a staler preconditioner; a falling
   ratio exonerates burn-in length. **Two runs, no new sampling** — but note the
   existing runs predate the instrumentation, so this needs the two
   configurations re-run, or the snapshot recomputed from saved chains.

   *Superseded rationale, kept for the record:* §4.3.36 showed the warm-up does
   not converge monotonically and the final-state hand-off discards the best
   state visited — worst on large_play, which reached NLL **0.172** and handed
   off **0.426** (2.48× worse). `warmup_use_best: true` (added 2026-08-24, needs
   `warmup_log_every > 0`) hands over the best-by-NLL burn-in state instead.
   **This removes a knob rather than tuning one**: keeping the best state makes
   burn-in length largely irrelevant, where §4.3.36's early-stopping rule still
   requires picking a criterion. Judge on centred `ratio`/`scale_z` **and CVaR
   CE against 3.0359** — §4.3.30 showed large_play passing both gates with a
   worse-than-chance CVaR reward, so the gate alone cannot clear it.

2. **Replicate the unreplicated load-bearing runs** with `sampling_seed`
   (§4.3.35's standing practice): large_diverse `recipe` and medium_diverse
   `burn100k` are each a single run. This also settles §4.3.28's CVaR CE gain,
   currently unconfirmed at 1.12× its 2·SE, and §4.3.27's jitter-alone effect at
   3.0× the floor.

3. **Redesign the sweep, then re-run stage 1 for the BNN.** Blocked on step 1.
   - **Objective: CVaR CE** (`--cvar-ce`, α = 0.05), validated in §4.3.22–23 —
     it ranks configurations opposite to mean CE, has a genuine interior
     optimum, and resolves the differences that matter at 8×75. Mean CE cannot
     be used: §4.3.22 shows it rewards worse stationarity, and at the amplitude
     it selected the CVaR reward is worse than chance.
   - **Fix `map_amp2`** at the value §4.3.23's post-fix curve settles. It has no
     interior optimum under mean CE (§4.3.16's cap history) and is derivable
     from the pooling convention and segment length.
   - **Fix `n_meas` high.** §4.3.24 showed it buys prior *coverage*, not just
     noise reduction, and §4.3.25 closed the fixed-set alternative with a bound.
   - **Gate on `fn_drift_centred_*`** per §3.6.3's 2026-08-24 amendment, now
     logged automatically (§4.3.29). Do **not** gate on raw: it is not even
     reproducible across replicates (§4.3.35).
   - **Do not reinstate `early_stop_acc_threshold`** — §3.5 removed it in round
     1 and §4.3.36 confirms warm-up quality does not predict final quality.
   - Sweep the rest jointly. Re-run MR and PT **only** if `batch_size` or
     `bt_pool` changes (§3.6.2, §4.3.15).

4. **Repeat for the other three variants**, then **transcribe** `num_chains` /
   `chains_per_gpu` into `scripts_bnn/antmaze_<v>_bnn_antmaze_eval.yaml` and
   re-run the field-by-field verification (§10.3).

5. **Then stage 4** (§5) — the 8-way normalization grid, which runs outside this
   repo in the surrounding `iqlpref` pipeline.

**Standing rules for every run above.**
- Judge on **centred `ratio` AND CVaR CE together**. §4.3.15's `bt_pool=sum` run
  and §4.3.30's large_play both passed every gate while the model was broken:
  **stationarity is necessary, not sufficient.**
- Read `ratio` from `--offset-shape-split`, never raw — raw is a mixture that
  cancelled a real effect twice (§4.3.16, §4.3.19) and is not reproducible
  (§4.3.35).
- **Replicate anything load-bearing** with `sampling_seed` (§4.3.35).
- Do not read ALIGNMENT at 8 chains (§4.3.7), and do not build arguments on
  orderings of the four variants — three mutually confounded orderings have
  already been refuted (§4.3.30, §4.3.31, §4.3.32).
- A §4.2 gate PASS is chain-count dependent and is not evidence of stationarity
  (§4.3.18, §7.1).

**What is not reachable, and what is disclosed.** Prior strength alone plateaus
near centred `ratio` ≈ 1.10 (§4.3.23); jitter and `n_meas` compose past it
(§4.3.28). The residual, and the fact that the shipped sampler is not stationary
in the raw sense, are §7.1 disclosures rather than open work.

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
