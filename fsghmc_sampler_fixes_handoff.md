# Hand-off: fSGHMC sampler review — momentum, `fraction_cool`, gradient clip

## Purpose

Three discrepancies were identified between the intended fSGHMC algorithm (Wu et al. 2025,
"Functional Stochastic Gradient MCMC for Bayesian Neural Networks", AISTATS; cyclical schedule from
Zhang et al. 2020) and the implementation in this repo. This document specifies each one precisely,
explains why it matters, and states the change to make **or** the investigation to run. Please work
through them in the priority order given at the end.

Some tasks are "fix" (implement a change) and some are "examine first" (instrument, then decide).
Each task is tagged accordingly. Where behavior changes, put it behind a config flag so existing runs
remain reproducible — do not silently change defaults unless the task says to.

## Context you need

The sampler draws posterior weight samples for a preference-learning BNN. Each collected weight
sample `w` is turned into a reward function `f(·; w)`; the **downstream quantity of interest is a CVaR
(mean of the lowest 5%) of the per-point posterior-predictive reward**, used for pessimistic offline
RL. This means the *lower tail* of the sampled predictive distribution, and the *number of usable
samples*, are what ultimately matter — bulk convergence is not enough. The evaluation block already
computes tail-ESS, folded R-hat, and CVaR-specific ESS/MCSE (via the Rockafellar–Uryasev integrand);
use those metrics to validate changes.

### File map

- `optbnn/sgmcmc_bayes_net/f_pref_net.py` — `FPrefNet`, contains the main `train()` loop where all
  three issues live (cyclical-LR block, sample-collection block, gradient clip).
- `optbnn/samplers/adaptive_sghmc.py` — `AdaptiveSGHMC` (Springenberg et al. 2016). The `momentum`
  buffer and its stationary scale are defined here.
- `optbnn/samplers/sghmc.py` — plain `SGHMC` (Chen et al. 2014). Cyclical schedule is normally used
  with `adaptive_sghmc`, but handle both where noted.
- `scripts_bnn/run_bnn_training.py` — `TrainConfig` (config fields) and the driver.

Line numbers below may have drifted; **locate by the code shown, not by line number**, and re-read the
current file before editing.

---

## Issue 1 — Momentum is zeroed, not resampled, at each cyclical cycle boundary

**Type: fix (modest impact) — put behind a config flag.**

### Current behavior

In `FPrefNet.train()`, cyclical-LR block, at the start of each cycle (`_cycle_step == 0`):

```python
if _cycle_step == 0:
    # Zero momentum at the start of each new hot phase
    for _pg in self.sampler.param_groups:
        for _p in _pg["params"]:
            _s = self.sampler.state.get(_p)
            if _s is not None and "momentum" in _s:
                _s["momentum"].zero_()
```

### Why this is wrong

Wu et al. Algorithm 2 **resamples** the momentum from its Gaussian at each cycle start; the code sets
it to zero instead. In high dimension the momentum norm concentrates far from zero, so zeroing drops
the chain into an atypical set of the momentum marginal and it then spends ~`1/mdecay` steps
re-equilibrating. Resampling from the correct distribution puts it straight into the typical set.

Important parametrization detail: in `AdaptiveSGHMC` the `momentum` buffer is **not** the HMC momentum
`z`; it is the position increment `v ≈ ε M⁻¹ z` (the update ends with `parameter.data.add_(momentum)`).
Its stationary law is approximately

```
v ~ N(0, lr² · minv_t),   where   minv_t = 1 / (sqrt(v_hat) + epsilon)
```

(derive: OU stationary variance `sigma²/(2·mdecay)` with `sigma² = 2·lr²·mdecay·minv_t − lr⁴ ≈
lr²·mdecay·minv_t·2`). At a cycle start the cyclical LR is at its peak `lr_max`, so the correct scale
uses `lr = lr_max`.

### The change

Replace the `zero_()` with a per-element resample from `N(0, lr_cycle² · minv_t)`:

```python
if _cycle_step == 0 and resample_momentum:   # new config flag, default True
    for _pg in self.sampler.param_groups:
        _eps = _pg.get("epsilon", 1e-16)
        for _p in _pg["params"]:
            _s = self.sampler.state.get(_p)
            if _s is None:
                continue
            if "v_hat" in _s:                 # adaptive_sghmc
                _minv_t = _s["v_hat"].sqrt().add(_eps).reciprocal()
                _std = _cycle_lr * _minv_t.sqrt()
            elif "momentum" in _s:            # plain sghmc: v ~ N(0, lr) → std sqrt(lr)
                _std = math.sqrt(float(_cycle_lr)) * torch.ones_like(_s["momentum"])
            else:
                continue
            _s["momentum"].normal_().mul_(_std)
```

Notes for the implementer:
- Handle both samplers: `adaptive_sghmc` has `v_hat`; plain `sghmc` does not (its buffer stationary
  std is `sqrt(lr)`, a different formula — see `sghmc.py`).
- `_cycle_lr` at `_cycle_step == 0` equals `lr_max`; you can use the already-computed `_cycle_lr`.
- Add `resample_momentum: bool = True` to `TrainConfig` and thread it through
  `sample_multi_chains_parallel` → `train_kwargs` → `train()`, mirroring how existing flags are passed.

### Calibrated expectations

This is the **lowest-impact** of the three fixes. The sample is collected at the *cold end* of the
cycle, by which point the momentum has re-equilibrated regardless, so the collected sample is not
directly contaminated — you are recovering ~10% of wasted exploration per cycle, not fixing a bias. Do
it because it is strictly more principled and matches the paper, but expect a small effect.

### Do NOT

The comments justify zeroing as a blow-up guard. That is a misdiagnosis: the actual blow-up risk is
`minv_t` reaching `1/sqrt(v_hat_min)` and meeting a large gradient at `lr_max`, which is guarded by
`v_hat_min` and `max_param_step`, **not** by the momentum reset. So resampling does not reintroduce a
safety problem. Do not re-add zeroing as a "safety" measure.

### Verify

Log momentum-buffer norm at cycle boundaries before/after the change; confirm the post-boundary
re-equilibration transient shortens. Confirm `param_ess`/`pred_ess` do not regress.

---

## Issue 2 — `fraction_cool` is dead; only one sample is collected per cycle  ← HIGHEST PRIORITY

**Type: fix — this is the most valuable change in this document.**

### Current behavior

`fraction_cool` is a `TrainConfig` field, threaded through `sample_multi_chains_parallel` →
`train_kwargs` → `train()`, but **never read** in the body. Its own docstring says
`fraction_cool: (unused; kept for signature compatibility)`.

The cyclical sample-collection block collects exactly **one** sample per cycle, at the single coldest
step:

```python
if use_cyclical_lr and step >= num_burn_in_steps:
    _post_burn = step - num_burn_in_steps
    _cycle_step = _post_burn % _cycle_len
    if _cycle_step == _cycle_len - 1:
        n_samples += 1
        if n_samples > n_discarded:
            self.sampled_weights.append(self.network_weights)
            self.num_samples += 1
```

### Why this matters (do the arithmetic)

With the typical config (`num_samples=50`, `num_chains=4`) this yields **200 total draws**. The CVaR
averages the lowest 5% → roughly **10 order statistics**. That is far too thin; it will dominate
`pred_cvar_mcse_rel` no matter what else is tuned. This is the primary reliability bottleneck for the
downstream quantity.

### Intended behavior (Zhang et al. 2020 cSG-MCMC)

Each cycle has a hot exploration phase followed by a **cool sampling phase**; you collect *multiple*
(thinned) samples during the cool fraction. `fraction_cool` is the fraction of each cycle spent in the
cool/sampling phase. Implementing it properly takes the draw count from ~200 to a few thousand, with
the lower-5% tail backed by hundreds of samples instead of ~10.

### The change

Replace the single-sample-per-cycle condition with cool-phase collection. The cool phase is the
low-LR tail of the cosine, i.e. `_cycle_step >= (1 - fraction_cool) * _cycle_len`. Thin within it (reuse
`keep_every` as the within-cool thinning interval, or add a dedicated field):

```python
_cool_start = int((1.0 - fraction_cool) * _cycle_len)
if use_cyclical_lr and step >= num_burn_in_steps:
    _post_burn = step - num_burn_in_steps
    _cycle_step = _post_burn % _cycle_len
    if _cycle_step >= _cool_start and ((_cycle_step - _cool_start) % keep_every == 0):
        n_samples += 1
        if n_samples > n_discarded:
            self.sampled_weights.append(self.network_weights)
            self.num_samples += 1
```

### The part that needs care — recompute the step budget

The current step budget assumes one sample per cycle:

```python
if use_cyclical_lr and num_samples is not None:
    num_steps = (num_samples + n_discarded) * _cycle_len
```

With multiple samples per cycle this over-runs. Decide and document the semantics of `num_samples`
(recommended: **total** samples across the whole run), then compute the number of cycles needed:

```python
_samples_per_cycle = max(1, math.ceil(fraction_cool * _cycle_len / keep_every))
_n_cycles = math.ceil((num_samples + n_discarded) / _samples_per_cycle)
num_steps = _n_cycles * _cycle_len
# (then, as now) num_steps += num_burn_in_steps
```

Stop collecting once `self.num_samples` reaches the target so you don't overshoot on the final cycle.

### Interactions / cautions

- Momentum resample (Issue 1) fires only at `_cycle_step == 0`, i.e. before the cool phase — no
  conflict. Do not resample momentum inside the cool phase.
- Within-cool draws are autocorrelated, so ESS will **not** scale linearly with raw draw count. That
  is expected and fine; `keep_every` controls the trade-off. Use `pred_cvar_ess_*` to tune it.
- Keep `fraction_cool` in a sane range (e.g. 0.1–0.5); `0.25` is the config default and a reasonable
  starting point.

### Verify

`pred_cvar_ess_min`/`pred_cvar_ess_median` and `pred_q05_ess_*` should rise substantially; 
`pred_cvar_mcse_rel_max` should fall. Bulk `pred_rhat`/`pred_ess` should not regress. Sanity-check that
total collected samples ≈ `num_chains * num_samples`.

---

## Issue 3 — Gradient clip modifies the sampled distribution's tail

**Type: examine first, then fix conditionally. Make the threshold configurable.**

### Current behavior

After the likelihood backward and the functional-prior-gradient addition, before the sampler step:

```python
torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100.0)
self.sampler.step()
```

The threshold `100.0` is hard-coded and the clip runs on **every** step, including the sampling phase.

### Why this is a problem for this project specifically

`clip_grad_norm_` is a nonlinear, non-measure-preserving modification of the drift. When it fires it
simulates a *different* SDE. It attenuates the drift in **steep** regions of the potential — precisely
where the log-density falls off fast — so the sampler under-corrects there, lingers in low-density
regions, and **fattens the tails** of the sampled distribution. The downstream CVaR reads the lower
tail, so any clip-induced bias lands directly on the reported quantity, in the direction that makes the
pessimistic bound look *more* conservative than it truly is (uncontrolled conservatism). A 95% lower
bound cannot be claimed while this bias is uncharacterized.

It is also partially redundant: `clip_grad_norm_` bounds the gradient (which then passes through
`minv_t`, up to ~100× amplification), whereas `max_param_step` bounds the *actual parameter change* —
the more direct guard.

### Step 1 — instrument (do this first, it is cheap and decisive)

`clip_grad_norm_` returns the **pre-clip** total norm. Capture and log it (per step during burn-in and
sampling, or aggregated):

```python
_gnorm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_grad_norm_value)
# log _gnorm (e.g. running max/mean, split by burn-in vs sampling phase) to wandb
```

Interpretation:
- If the pre-clip norm is routinely `<< 100` during the **sampling** phase, the clip is effectively
  inert and the dynamics are unmodified where it matters. Record this — it lets the paper state
  plainly that clipping did not activate during sampling, which is a clean, defensible sentence.
- If it **does** fire during sampling, there is a real bias to fix, and the source needs identifying
  (see Step 3).

### Step 2 — scope the clip to burn-in

Bias during burn-in is harmless (you are only locating the typical set). Make the clip configurable and
restrict it to burn-in by default:

- Add `clip_grad_norm_value: Optional[float] = 100.0` and `clip_during_sampling: bool = False` to
  `TrainConfig`; thread through as usual.
- Apply the clip when `clip_grad_norm_value is not None and (step < num_burn_in_steps or
  clip_during_sampling)`.

If the chain then blows up in the sampling phase, that is **diagnostic information**, not a regression:
it means step size, `v_hat_min`, or kernel conditioning is wrong — which the clip was previously
masking.

### Step 3 — address root causes rather than clipping harder

If instrumentation shows the clip firing during sampling, identify which source and fix it directly:

- **Bradley–Terry logit scale.** The per-pair logit is a trajectory sum over `T` timesteps
  (`torch.nansum(... , dim=1)`), and a deep net's output scales like `w^(depth+1)`, so the softmax can
  saturate and produce large likelihood gradients. Fix: divide the trajectory sum by `T`, or introduce
  an explicit Bradley–Terry temperature, so the logit scale is trajectory-length-independent. (This is
  a modeling change — gate it behind a flag and confirm it does not shift accuracy.)
- **Functional-prior solve.** The prior gradient contains `K_{X_M}^{-1}(f(X_M) - m(X_M))`, a Cholesky
  solve with `meas_jitter = 1e-6` on a `256×256` kernel. If the LCF features are near-collinear the
  solve is ill-conditioned and the gradient explodes. Fix: raise `meas_jitter`, and/or log the
  condition number / smallest eigenvalue of `K_{X_M}` to confirm.

### Do NOT

Do not simply remove the clip without instrumenting first, and do not raise the threshold blindly — the
goal is to know whether it fires and why, then remove the need for it, not to hide it.

### Verify

After scoping to burn-in: confirm sampling-phase behavior is stable with the clip off (or explain the
instability via Step 3). Compare `pred_cvar_*` and `pred_q05_*` tail metrics with vs. without
sampling-phase clipping — they should not depend on the clip once root causes are addressed.

---

## Cross-cutting guardrails (apply to all three tasks)

- **Out of scope — do not touch:** the momentum is `g(·;z) = z` by design (no auxiliary momentum
  network); this is intentional and correct. Do not add a function-space momentum network or otherwise
  alter the `(f, g)` structure.
- **Do not alter the prior semantics.** The functional GP prior (`LCFModel` / `MapInformedGPPrior`) is
  a genuine prior, not conditioned on the preference labels. Do not "simplify" it into anything that
  reads the training targets.
- **Config-flagged, reproducible.** Every behavioral change goes behind a `TrainConfig` flag with the
  old behavior available, so prior runs reproduce. New flags: `resample_momentum`,
  `clip_grad_norm_value`, `clip_during_sampling` (and any thinning field you add for the cool phase).
- **Validate with the existing diagnostics**, not ad hoc checks. The evaluation block already logs
  `pred_cvar_ess_*`, `pred_cvar_mcse_rel_*`, `pred_q05_ess_*`, `pred_folded_rhat_*`. Use them as the
  acceptance signal, since they measure the tail quantity that actually matters.
- **Naming trap:** `state["g"]` inside `AdaptiveSGHMC` is Springenberg's smoothed-gradient estimate,
  **not** a momentum function. Do not confuse it with anything algorithmic named `g`.

## Priority order

1. **Issue 2 (`fraction_cool`)** — largest impact on CVaR reliability; ~200 → ~thousands of draws.
   Includes the step-budget recomputation, which is the part most likely to introduce bugs — review it
   carefully.
2. **Issue 3 (gradient clip)** — start with Step 1 instrumentation (cheap, decisive), then scope to
   burn-in. Determines whether the reported tail bounds are biased.
3. **Issue 1 (momentum resample)** — correctness/faithfulness improvement, modest expected effect.

## Suggested deliverables back to me

- A short diff per issue, each behind its config flag.
- The instrumentation readout for Issue 3 (pre-clip gradient-norm distribution, burn-in vs sampling),
  since that decides whether Step 2/3 are even necessary.
- One before/after comparison of `pred_cvar_ess_*` and `pred_cvar_mcse_rel_max` for the `fraction_cool`
  change, to confirm the tail actually improved.
