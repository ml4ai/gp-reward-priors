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
| **Cyclical step size** (Zhang et al. 2020) in place of the paper's decaying ε | Deliberate — standard SGHMC gets trapped in a single basin. Correctly implemented (samples taken only at cool-phase end, momentum resampled at cycle start, structurally close to Alg. 2's outer loop). But the *composition* of cSGMCMC with fSGHMC is analysed by neither paper. `function_space_drift` is the empirical check that early and late cycles are one measure. |
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
that is why, and it does not indicate a discrepancy. Any variant needing
`--worst-k` will likewise need its `c4` re-run first, for the same reason.

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

Record per variant: the chosen `num_chains` / `chains_per_gpu`, the full table
of §4.3 metrics at each candidate, the §4.2 numbers at the chosen budget, and
what the `_max` extremes did (they are sparse, not censored — §4.5). Then transcribe `num_chains` and
`chains_per_gpu` into the production configs (§10.3) and re-run the field-by-field
verification.

`scripts_bnn/diagnose_sampling_tail.py --run-dir <OUT_DIR>` recomputes every tail
statistic from saved chains without re-sampling; `--worst-k` lists the
least-converged points with their torso (x, y) so you can judge whether the
unresolved ones are genuinely multimodal states. `--ce-ladder` and
`--draw-ladder` answer "would fewer draws have done?" from a completed run.

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
location drift. A further 16 trials predate the drift metric and cannot be
classified; none was a winner candidate. In large_diverse
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
that predate the drift metric and are unclassifiable. The criteria are **not**
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
| 3 | BNN | **not started — this is the next action** |
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
why those are sparse rather than censored), and **check `fn_drift_*_z` and
`param_clamp_sampling_pct` first**: a run that is not sampling the target
measure makes every tail number meaningless. The winners all pass those at
4 chains × 75 draws, but the budget change has to be re-checked, not assumed.

**Progress so far.** medium_play's `c4` rung is done (2026-08-16): the winner
reproduced, `fn_drift`/clamp re-verified, and `--worst-k 20` showed only 56 of
6400 points unresolved (0.88%), clustered in three maze regions rather than
diffuse — see §4.5. Nothing else has been run.

**Do this next, in order:**

1. **medium_play `c8`.** One command; only the budget fields need overriding,
   because the production config already carries the winner's nine values:

   ```
   cd scripts_bnn && CUDA_VISIBLE_DEVICES=0,1 nohup python run_bnn_training_antmaze_eval.py \
       --config_path scripts_bnn/antmaze_medium_play_bnn_antmaze_eval.yaml \
       --seed 0 --num_chains 8 --chains_per_gpu 4 \
       --OUT_DIR ./exp/stage3_medium_play_c8 > ../exp/stage3_medium_play_c8.log 2>&1 &
   ```

   Then, from the **repo root** (the diagnostic does not `chdir`, unlike the
   training script, so its `--run-dir` is relative to wherever you launch it):

   ```
   python scripts_bnn/diagnose_sampling_tail.py \
       --run-dir exp/stage3_medium_play_c8_0 --worst-k 20 --device cuda
   ```

   Compare against the `c4` row in §4.3. Expect `cvar_ess_median` and
   `q05_ess_median` to roughly double, `cvar_mcse_rel_median` to fall by about
   √2, and **R-hat to rise** — §4.5 explains why that is correct behaviour and
   not a regression.

2. **medium_play `c16`**, if `c8` still buys a meaningful ESS/MCSE improvement.
   Stop when a doubling buys little (§4.6).

3. **Repeat the ladder for the other three variants.** They start from very
   different places: large_play and medium_diverse already sit at
   `cvar_ess_median` 210 and 272 at `c4` and may need no increase at all, while
   large_diverse (57) looks like medium_play. Do not assume one chain count
   suits all four (§4.3).

4. **Transcribe** the chosen `num_chains` / `chains_per_gpu` per variant into
   `scripts_bnn/antmaze_<v>_bnn_antmaze_eval.yaml`, then re-run the
   field-by-field verification (§10.3).

5. **Then stage 4** (§5) — the 8-way normalization grid, which runs outside this
   repo in the surrounding `iqlpref` pipeline.

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
