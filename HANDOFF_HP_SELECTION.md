# Hand-off: hyperparameter selection procedure (antmaze, all model families)

> Status 2026-08-08. **Stage 1 complete** for all three families, winners written
> into the production configs. **Stage 2 (BNN sampling tier): 3 of 4 sweeps have
> fired**; medium_diverse is still searching, and no stage-2 winner has been
> written into a production config yet. **Stages 3–4 not started.**
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

## 2. The four stages

| stage | what is chosen | selected on | seed |
|---|---|---|---|
| 1 | model architecture + optimiser/warm-up HPs | validation loss | 0 |
| 2 | (BNN only) posterior sampler schedule | validation loss | 0 |
| 3 | (BNN only) chain count / draws per chain | MCMC tail diagnostics | 0 |
| 4 | output normalization function | max mean IQL score over eval points | 0 |

Stages 1–2 are automated wandb sweeps. Stage 3 is a deliberate manual step.
Stage 4 is a small grid search over the downstream RL objective.

---

## 3. Stages 1–2: sweep-based selection

### 3.1 Shared design (identical across all families)

**Search:** wandb `method: bayes`, one sweep per (family × antmaze variant), 16
sweeps total for stage 1 + 4 more for stage 2.

**Trial budget: 130 per model family**, matched deliberately:

| family | stage | swept params (d) | run_cap |
|---|---|---|---|
| MR | 1 | 3 | 130 |
| PT | 1 | 4 | 130 |
| BNN | 1 (warm-up tier) | 6 | 70 |
| BNN | 2 (sampling tier) | 5 | 60 |

The BNN's two tiers sum to 130, so **every family receives the same trial
budget**. This is the single most important fairness property: a reviewer
cannot argue the proposed method simply got tuned harder than the baselines.
130 is generous for MR's 3-dimensional space — that is intentional, since
over-tuning a *baseline* is the safe direction. The `run_cap` is a safety
limit, not the thing that determines the answer; the stopping rule below is.

**Selection metric: minimise validation loss.** Never accuracy. Accuracy is
insensitive to confidence, and the downstream use of these models is a
posterior-predictive quantity, so a calibrated loss is the right target.

| family / stage | metric key | notes |
|---|---|---|
| MR, PT | `eval_loss_best` | requires `criteria_key: loss` in the config |
| BNN stage 1 | `warmup_final_nll` | val-subsample CE logged right after burn-in |
| BNN stage 2 | `val_mean_cross_entropy` | posterior-predictive CE after full chains |

**Stopping rule: stop when the best-so-far has not improved for K = 15
consecutive trials.** Applied uniformly to every sweep in both stages.

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

**BNN stage 1 — warm-up tier**
(`scripts_bnn/sweep_antmaze_<variant>_bnn_warmup_antmaze_eval.yaml`)

| param | distribution | range |
|---|---|---|
| `width` | int_uniform | 6–10 (64–1024) |
| `depth` | int_uniform | 2–6 |
| `n_meas` | int_uniform | 0–64 (0 = functional prior off) |
| `map_amp2` | log_uniform_values | 1 – 1000 |
| `sghmc_lr` | log_uniform_values | 1e-4 – 1e-2 |
| `mdecay` | log_uniform_values | 1e-3 – 1e-1 |

Fixed: `num_burn_in_steps: 5000`, `warmup_log_every: 250`, `seed: 0`, and
`early_stop_acc_threshold: 1.01`. That last value is a trick: accuracy is always
below 1.01, so **every run stops right after warm-up**, skipping the expensive
parallel-chain phase. Stage 1 therefore costs ~7 min/trial and tunes only what
is visible at warm-up: architecture and prior strength.

`n_meas` and `map_amp2` are the two **prior-strength** knobs, and both are
searched, including `n_meas = 0` so the optimiser can reject the prior outright
if it does not help. `map_amp2` scales the whole map kernel (`K → amp2·K`),
changing the prior's reward amplitude while leaving its correlation structure —
the map-informed part — untouched. It exists because the `bt_pool="mean"` fix
tied per-point reward amplitude to Bradley–Terry logit magnitude; without it the
legacy O(1) prior amplitude is incompatible with the likelihood and the sweep
can only escape by driving `n_meas → 0`. See `HANDOFF_CVAR_SAMPLER_2026-08.md`.

**BNN stage 2 — sampling tier**
(`scripts_bnn/sweep_antmaze_<variant>_bnn_sampling_antmaze_eval.yaml`)

| param | distribution | range |
|---|---|---|
| `sghmc_lr` (lr_min) | log_uniform_values | 5e-5 – 5e-4 |
| `sghmc_lr_max` | log_uniform_values | 5e-4 – 5e-3 |
| `cycle_length` | q_uniform | 500 – 3000, q = 250 |
| `mdecay` | log_uniform_values | 1e-3 – 1e-1 |
| `fraction_cool` | uniform | 0.1 – 0.5 |

The `sghmc_lr` ceiling meets the `sghmc_lr_max` floor at 5e-4, so
`lr_max ≥ lr_min` holds by construction. The 5e-3 cap on `lr_max` reflects a
measured divergence cliff under mean pooling (stable at 0.0048, divergent at
0.0064 for medium_play).

Architecture and prior strength are **inherited from that variant's stage-1
winner** (`width`, `depth`, `n_meas`, `map_amp2`), transcribed into the stage-2
config with the source run id, trial number and metric recorded inline for
provenance. The launcher refuses to start stage 2 while any `FILL_ME` remains.

Fixed budget per stage-2 trial: `num_chains: 4`, `chains_per_gpu: 4`,
`num_samples: 35`, `n_discarded: 2`, `num_burn_in_steps: 5000`,
`samples_per_cycle: 1`, `chain_init_jitter: 0.0`, `use_cyclical_lr: true`,
`seed: 0`. This is a deliberately reduced budget (140 draws vs the production
2480) — stage 2 ranks *schedules*, and the draw count is set in stage 3.

Two stage-2 values deserve their own note because they look like tuning and are
not:

- **`burn_in_lr: 0.002`, uniform across variants.** Burn-in inherits `lr_min` by
  default, but stage 2's `lr_min` range sits far below stage 1's productive
  range, so a fixed-length burn-in under-fits and the warm-up gate rejects the
  very architecture stage 1 selected. Decoupling burn-in step size from the
  swept cool-phase `lr_min` is a **design fix** — warm-up quality should not be
  a function of the schedule under test — not a tuned value.
- **`early_stop_acc_threshold: 0.0` — the warm-up gate is DISABLED.** Every
  stage-2 trial runs to completion and is ranked on `val_mean_cross_entropy`;
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
- **`batch_size: 64`, `epochs: 5000`, `num_burn_in_steps: 5000`.**

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

### 3.5 Why stage 2 has no warm-up gate

Stage 2 originally early-stopped any trial whose warm-up accuracy fell below a
static 0.75, to avoid spending ~6 h sampling from a broken starting point. That
gate was **removed and all four stage-2 sweeps restarted**, for two reasons.

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

### 3.6 What a divergent stage-2 trial looks like

With the gate removed, low-friction schedules run to completion and are scored.
Some of them **diverge numerically**. Observed on the restarted sweeps:

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

**`val_mean_cross_entropy` is not monotone in brokenness.** A run that collapses
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

---

## 4. Stage 3 (BNN only): hand-tuning the draw budget

Stage 2 selects a *schedule*; it does not select how many samples to draw. The
production runs use a larger budget than the sweep, set by hand, and this is the
one deliberately manual step in the procedure.

**What is adjusted:** `num_chains` and/or `num_samples` (draws per chain), plus
`chains_per_gpu` for placement. Nothing else. The schedule
(`sghmc_lr`, `sghmc_lr_max`, `cycle_length`, `mdecay`, `fraction_cool`),
architecture, and prior strength stay exactly as selected.

**Why it is not a sweep:** the quantity being improved is Monte-Carlo error on
the downstream CVaR₀.₀₅, not model quality. It is not visible in
`val_mean_cross_entropy` — a chain can have excellent predictive CE and still
resolve the lower 5% tail poorly. More draws essentially always help; the only
question is how many are enough, which is read off convergence diagnostics
rather than searched.

**What it is judged on** (all logged by the eval block; see
`HANDOFF_CVAR_SAMPLER_2026-08.md` §3):

- `val_pred_cvar_ess_min` — effective sample size for the CVaR integrand
- `val_pred_cvar_mcse_rel_max` — relative MC standard error of the CVaR
- `val_pred_cvar_rhat_max`, `val_pred_folded_rhat_*` — between-chain agreement
- `val_pred_q05_ess_*` — ESS at the 5% quantile (the VaR)
- `gradnorm_sampling_pct_over_clip` — should be ~0

Do **not** judge this by `param_*` diagnostics (meaningless for a BNN under
weight-space non-identifiability) or by bulk `pred_rhat`/`pred_ess` (they
certify the median, not the tail).

**Warning: at the stage-2 budget, the `_max` / `_min` extremes above are
censored and carry no ranking information.** Measured 2026-08-08 across all
four stage-2 sweeps: `val_pred_cvar_rhat_max` is **1.2555612045333548 on nearly
every healthy trial** — identical across variants, architectures and schedules,
and identical to 16 digits between the val and test splits of the same run
(winner `k83frxm7`). `val_pred_q05_ess_min` is likewise pinned at 12.5367 in a
large majority of trials, with the remainder in a narrow 11.3–14.5 band.

This is the attainable ceiling (resp. floor) of the rank-normalized estimators
at 4 chains × 33 retained draws, reached as soon as *any one* of the ~640
evaluation points has fully separated chains. It is real information — some
point is badly mixed — but it is saturated, so it cannot rank one schedule
against another, and it cannot get worse.

Two consequences for stage 3:

- **Judge on the distributional statistics, which do vary:**
  `val_pred_cvar_rhat_median`, `val_pred_cvar_rhat_pct_over_1.01`,
  `val_pred_folded_rhat_95th_pct`, `val_pred_cvar_ess_median`,
  `val_pred_q05_ess_median`, `val_pred_cvar_mcse_rel_median`. The `_max`/`_min`
  extremes are worth recording but not steering on.
- **Verify the censoring lifts before trusting the extremes at production
  budget.** 2480 draws is ~19× the stage-2 budget and the saturation may
  disappear, but check it explicitly — otherwise the failure mode is
  "diagnostics didn't improve when I added draws", which is indistinguishable
  from a mixing problem (see the last paragraph of this section) and would send
  you chasing `lr_max` / `chain_init_jitter` for no reason.

Note the *diverged* trials are the ones with unpinned values here (CVaR ESS 25–89,
R-hat 0.997–1.085) — a broken run whose predictive has collapsed to a constant
mixes "perfectly". Do not read a good-looking extreme as evidence of health.

**Reference point:** the production configs before this selection round used
`num_chains: 8`, `num_samples: 310` (2480 draws) with `chains_per_gpu: 2`. The
offline validator `scripts_bnn/diagnose_sampling_tail.py --run-dir <OUT_DIR>`
recomputes all tail diagnostics from saved chains without re-sampling, and
`--worst-k` lists the least-converged points with their torso (x, y) so you can
judge whether unresolved points are genuinely multimodal states.

Increasing draws does not fix everything: between-chain mode separation is
resolved by hot-phase mixing (`lr_max`, `fraction_cool`) and by
`chain_init_jitter`, not by more samples. If R-hat stays high with large ESS,
that is a mixing problem, not a budget problem.

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

Entity `champlin-university-of-arizona`. Status as of 2026-07-30.

### Stage 1 — MR (`MR-training`, metric `eval_loss_best`) — complete

| variant | sweep | winner | trial / trigger | metric | width | depth | lr |
|---|---|---|---|---|---|---|---|
| medium_play | `70742ym5` | `za1bgyme` | 18 / 33 | 0.125673 | 8 | 5 | 6.240e-3 |
| medium_diverse | `vilrah4f` | `68b2hjeh` | 18 / 33 | 0.237777 | 6 | 1 | 4.818e-3 |
| large_play | `qkjet6r3` | `r8vpz8s5` | 15 / 30 | 0.154711 | 9 | 4 | 9.378e-3 |
| large_diverse | `59czpdwf` | `88wg34ln` | 5 / 20 | 0.210400 | 7 | 3 | 5.840e-3 ⚠️ |

Transcribed into `scripts_mr/antmaze_<variant>_mr_antmaze_eval.yaml` with
provenance headers and `criteria_key: loss`.

### Stage 1 — PT (`PT-training`, metric `eval_loss_best`) — complete

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

### Stage 1 — BNN warm-up tier (`BNN-training`, metric `warmup_final_nll`) — complete

| variant | sweep | winner | trial / trigger | nll | width | depth | n_meas | map_amp2 |
|---|---|---|---|---|---|---|---|---|
| medium_play | `kk79h8xf` | `05byzfhm` | 41 / 56 | 0.204278 | 6 | 2 | 14 | 313.204 |
| medium_diverse | `pyrz4qou` | `bk27aibh` | 23 / 38 | 0.316808 | 10 | 6 | 35 | 772.779 |
| large_play | `jhpdsl60` | `3orxv3kl` | 10 / 25 | 0.227336 | 7 | 6 | 10 | 623.485 |
| large_diverse | `in2p7l17` | `st3a5fgh` | 17 / 32 | 0.235876 | 6 | 3 | 11 | 459.295 ⚠️ |

**Result worth stating in the paper:** every variant selected a large
`map_amp2` (313–773) together with a non-trivial `n_meas` (10–35). The
functional prior is retained everywhere, at roughly 18–28× the legacy amplitude.
This replicates an earlier, less clean run and is now established under a fully
pre-registered procedure.

### Stage 2 — BNN sampling tier (metric `val_mean_cross_entropy`) — 3 of 4 fired

Live sweeps: `ld9oi90s` (medium_play), `o9g70yby` (medium_diverse), `u5snid84`
(large_play), `gnlrcb7y` (large_diverse). These are the **second** stage-2
attempt; the first was discarded when the warm-up gate was removed (§3.5).

Counts below re-verified 2026-08-08.

| variant | sweep | winner | trial / trigger | val CE | diverged |
|---|---|---|---|---|---|
| large_play | `u5snid84` | `d1ddc4yg` | 10 / 25 | **0.210992** | 7 / 26 |
| large_diverse | `gnlrcb7y` | `k83frxm7` | 17 / 32 | 0.231372 | 0 / 35 |
| medium_play | `ld9oi90s` | `ge5h8lfd` | 5 / 20 | 0.244184 | 0 / 21 |
| medium_diverse | `o9g70yby` | **still running** | best @10 | 0.284295 | 6 / 17 |

All three fired winners have **rule winner = best-of-all** and are themselves
clean (not divergence-flagged). The winning schedules:

| variant | `sghmc_lr` | `sghmc_lr_max` | `cycle_length` | `mdecay` | `fraction_cool` |
|---|---|---|---|---|---|
| medium_play | 1.759684326975111e-4 | 2.7127155004169033e-3 | 2750 | 3.886566686095279e-2 | 0.26490734354273915 |
| large_play | 4.4675299100389534e-4 | 4.123441940823021e-3 | 1500 | 2.2440542355891248e-2 | 0.4464505988695707 |
| large_diverse | 2.2274405063627215e-4 | 4.820665555950937e-3 | 2500 | 8.916002139869199e-2 | 0.41191474620821766 |
| medium_diverse | *(pending)* | | | | |

medium_diverse's current leader is `nfqz8f11` (val CE 0.284295, clean, `sghmc_lr`
4.958689700913806e-4, `sghmc_lr_max` 2.1452292827590347e-3, `cycle_length` 2250,
`mdecay` 4.286099171785899e-2, `fraction_cool` 0.14154720839003554) — **do not
use it until the rule fires.** Its margin over the clean field is comfortable
(0.2843 vs 0.3182 for the next clean trial, `lzyn872v`, ~12%), but three
divergence-flagged trials sit at 0.300–0.305, i.e. between the two — see §3.6.1,
which is the reason that matters.

**Divergence is confined to the depth-6 networks.** medium_play (depth 2) and
large_diverse (depth 3) produced zero divergent trials in 55 combined; large_play
(depth 6) hit 27% and medium_diverse (depth 6, width 1024) 43%. Note large_play
had the most hostile search yet produced the best and cleanest winner — the
blow-ups were confined to bad regions of schedule space and never touched the
selected configuration.

**Discarded first attempt**, for the record (sweeps `zkkg4kdu`, `jpu2vqce`,
`7kfieu41`, `c05yyh72`): ~234 GPU-hours and 47 trials written off when the gate
was removed. None had fired, so no winner was ever read from them; they inform
§3.5 only.

## 7. Disclosures required in the write-up

**Rule-vs-best disagreements.** In two sweeps the stopping-rule winner is not
the best trial observed, because a better configuration arrived after the
trigger and was discarded per the pre-registered rule:

| sweep | rule winner | best-of-all | gap | affects |
|---|---|---|---|---|
| BNN warm-up / large_diverse | t17, nll 0.235876 | t38, nll 0.226704 | 4.0% | the proposed method |
| MR / large_diverse | t5, 0.210400 | t27, 0.203892 | 3.2% | a baseline |

State the gap convention explicitly: these are quoted as *"the rule winner is X%
worse than best-of-all"* (rule ÷ best − 1). The reverse convention gives 3.9%
and 3.1%; pick one and use it consistently.

Note that one disagreement hurts the proposed method and the other hurts a
baseline, so the rule cannot be characterised as self-serving. Note also that
MR/large_diverse's discarded config sits in a very different regime
(lr 2.579e-4 vs 5.840e-3) — the sweep found a distinct basin late, not a
near-tie.

**Budget.** Report that every family received 130 trials under the same
stopping rule, and that sweeps ran until the rule fired rather than to the cap.

**Boundary winners.** PT/medium_diverse (if its lr-floor leader holds) and any
stage-1 `map_amp2` near 1000 should be flagged as possibly range-limited.

**Numerically divergent stage-2 trial counts** (report these): medium_play 0/21,
large_diverse 0/35, large_play 7/26, medium_diverse 6/17-and-counting. No fired
winner is divergence-flagged. Report alongside these that the flag combines a
NaN/Inf condition with a 1%-over-clip threshold, and that in medium_diverse only
3 of the 6 are outright blow-ups (§3.6.1) — quoting the raw count without that
split overstates the instability.

**Numerically divergent stage-2 trials.** Removing the warm-up gate (§3.5) let
low-friction schedules run to completion, and some diverge outright — Inf
gradients, NaN convergence diagnostics, a reward function collapsed to a
constant scoring exactly `ln 2` (§3.6). They are correctly ranked as bad and so
do not threaten selection, but the count should be reported, and no divergent
configuration may be carried into stage 3.

**Residual burn-in/sampling confound in stage 2.** `mdecay` sets friction during
both burn-in and sampling, so a schedule's warm-up quality is not independent of
the schedule being tested. Removing the warm-up gate (§3.5) stops this from
hard-rejecting configurations, but low-friction schedules are still penalised
through a poor warm-up rather than judged purely on their sampling behaviour.
This is a known limitation of the stage-2 search, most likely to matter for
medium_diverse, whose network is the largest and therefore the most
friction-hungry during burn-in.

---

## 8. Tooling

**`launch_hp_sweeps.sh <phase1|phase2>`** (repo root) — creates and runs the
sweeps on the GPU box. Preflights the seed-0 data splits, tuning sets, env, and
GPU count; refuses phase 2 while any `FILL_ME` value remains; caches sweep ids
in `exp/sweep_ids_<phase>.txt` so re-runs resume rather than duplicate; launches
exactly **one agent per sweep** (serial trials give the Bayes optimiser full
history, and the eval scripts write to deterministic per-seed output paths that
two concurrent runs would clobber).

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
| 1 | MR | 4/4 sweeps fired; winners in `scripts_mr/antmaze_<v>_mr_antmaze_eval.yaml`, `criteria_key: loss` |
| 1 | PT | 4/4 sweeps fired; winners in `scripts_pt/antmaze_<v>_pt_antmaze_eval.yaml`, `criteria_key: loss` |
| 1 | BNN warm-up | 4/4 sweeps fired; winners transcribed into the stage-2 sweep yamls |
| 2 | BNN sampling | **3/4 fired** (medium_play, large_play, large_diverse); medium_diverse still running |
| 3 | BNN | not started |
| 4 | all | not started |

Every fired sweep has provenance (run id, trial, metric, trigger) recorded as a
comment in the config it was written into.

### 10.2 Immediate next action

**Wait for `o9g70yby` (medium_diverse stage 2) to fire.** Check it with
`check_sweep_convergence.py` (§8) against sweep `BNN-training/o9g70yby` at
patience 15. It has fired when the output reads `STOP FIRED : yes`.

Do not read a winner before then — this sweep has repeatedly improved late.

As of 2026-08-08 08:00 UTC it has 17 trials done and trial 18 running, with the
best still at trial 10. If nothing improves, **the rule fires at trial 25** — 8
more trials, roughly **2 days** at the current ~6 h/trial. Any improvement
resets the 15-trial count.

It is now the only sweep on the box and running ~2.8× faster per sampling step
than it did while sharing it (§10.7), so the older "6–13 h per trial, expect
days rather than hours" estimate no longer applies.

### 10.3 Then: write the BNN production configs

This is the step with a **trap**. `scripts_bnn/antmaze_<variant>_bnn_antmaze_eval.yaml`
still carries pre-selection values from an earlier era, **and is missing
`map_amp2` and `burn_in_lr` entirely**. Fields absent from the yaml fall back to
`TrainConfig` defaults, which would silently give:

- `map_amp2 = 1.0` — discarding the stage-1 prior-amplitude result (the winners
  are 313–773), i.e. throwing away the project's main prior finding;
- `burn_in_lr = None` — reinstating the burn-in/sampling confound that §3.5
  exists to describe.

Both lines must be **added**, not merely edited. Each config must end up with:

| field | source |
|---|---|
| `width`, `depth`, `n_meas`, `map_amp2` | stage-1 winner (§6; also present in the stage-2 sweep yamls) |
| `sghmc_lr`, `sghmc_lr_max`, `cycle_length`, `mdecay`, `fraction_cool` | stage-2 winner (§6) |
| `burn_in_lr: 0.002` | fixed, uniform (§3.2) |
| `num_chains`, `num_samples`, `chains_per_gpu` | **stage 3 decides these** — leave until then |
| `seed` | leave at 1; `train_rewards.sh` overrides it per evaluation seed |

Afterwards verify that each parsed config contains `map_amp2`, `burn_in_lr` and
all five schedule fields, and that they match the winning wandb runs
field-by-field. (The same verification caught nothing wrong for MR and PT, but it
is cheap and these are the numbers every reported result depends on.)

### 10.4 Then: stage 3, then stage 4

Stage 3 (§4) raises `num_chains` / `num_samples` until the CVaR tail diagnostics
are acceptable — judged on `val_pred_cvar_ess_min`, `val_pred_cvar_mcse_rel_max`,
`val_pred_folded_rhat_*` and `gradnorm_sampling_pct_over_clip ≈ 0`, **not** on
`val_mean_cross_entropy` and **not** on `param_*`. The pre-selection reference
point was `num_chains: 8`, `num_samples: 310`, `chains_per_gpu: 2`.

**Read the censoring warning in §4 before using any of those numbers.** At the
stage-2 budget the `_max`/`_min` extremes are pinned at estimator ceilings and
rank nothing; steer on the median / 95th-pct / `pct_over_1.01` variants and
confirm the extremes de-saturate at the production draw count.

Stage 4 (§5) is the 8-way normalization grid, selected on max mean IQL score at
seed 0. It runs outside this repo, in the surrounding `iqlpref` pipeline.

### 10.5 Sweep IDs

Entity `champlin-university-of-arizona`.

| family / stage | medium_play | medium_diverse | large_play | large_diverse |
|---|---|---|---|---|
| MR stage 1 | `70742ym5` | `vilrah4f` | `qkjet6r3` | `59czpdwf` |
| PT stage 1 | `z6nrw1vy` | `sridqxoj` | `1z6xo2u0` | `gjphiwvs` |
| BNN stage 1 | `kk79h8xf` | `pyrz4qou` | `jhpdsl60` | `in2p7l17` |
| BNN stage 2 | `ld9oi90s` | `o9g70yby` | `u5snid84` | `gnlrcb7y` |

Superseded, not to be used: the first clean-restart stage-2 attempt (`zkkg4kdu`,
`jpu2vqce`, `7kfieu41`, `c05yyh72`), and everything from the pre-restart round
described in §0.

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

### 10.7 Concurrent sweeps contend for something host-side (~2.8×)

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

**It is not GPU or core starvation.** `leviathan` has 6 A6000s and 255 logical
CPUs; each trial occupies one GPU (`num_chains: 4`, `chains_per_gpu: 4`), so at
four concurrent sweeps neither resource was oversubscribed.

The leading candidate is **thread oversubscription**: torch defaults intra-op
threads to core count, so four processes each spawning ~255-thread BLAS/OpenMP
pools on 255 cores thrash on context switching. Shared-filesystem HDF5 reads and
host memory bandwidth are the other candidates. **Not diagnosed — test
`OMP_NUM_THREADS` before the next phase that runs jobs concurrently** (stage 4
runs 4+ IQL jobs). A 2.8× throughput loss is worth an hour of investigation.

This does not affect any selection result — throughput is not an input to any
stopping rule or metric — but it does affect every schedule estimate in this
document.
