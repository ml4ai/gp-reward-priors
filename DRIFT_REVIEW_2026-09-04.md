# Independent review: fSGHMC stationarity, 2026-09-04

Scope: `HANDOFF_HP_SELECTION.md` §4.3.6–4.3.73, `optbnn/samplers/adaptive_sghmc.py`,
`optbnn/sgmcmc_bayes_net/f_pref_net.py`, `optbnn/utils/util.py::function_space_drift`,
`scripts_bnn/run_bnn_training_antmaze_eval.py`, the four `*_bnn_antmaze_eval.yaml`
configs and `sweep_antmaze_medium_play_bnn_antmaze_eval.yaml`; Wu et al. (2025)
Alg. 2 / Eq. 9, Springenberg et al. (2016) Alg. 1, Zhang et al. (2020) Eq. 1 and
Alg. 1.

Everything below is either (a) read directly out of the code, (b) computed from
numbers already in the handoff or the configs, or (c) simulated. Nothing here
required a training run. Where a claim is a hypothesis rather than a
verification, it says so.

**Headline.** The sampler's *mathematics* is sound — the Springenberg
discretisation is faithful, the `-ε⁴` term really is the correct `B̂` correction
(§4.3.26 is right and should stay closed), and the functional-prior gradient is
assembled exactly as Wu et al. Eq. 6/9 specifies. The problems are in the
**measurement layer and the experimental design**, and there are five of them.
Two of the five mean that conclusions the investigation currently treats as
settled — "ε is at its stability ceiling" and "step scale is refuted, it is
compute" — are not supported by the experiments that produced them. One means a
swept hyperparameter has been inert in every run ever done. And the "60×
per-variant constant" that §4.3.67 elevated to the project's one durable
invariant is tracked, in rank order, by a quantity that is entirely a product of
two swept sampler scalars — i.e. it may be the sweep's doing, not the
posterior's.

---

## 0. What is correct, so it can stop being re-tested

Read the code against the three papers line by line. These are verifications,
not opinions:

1. **The discretisation is Springenberg Alg. 1.** `τ ← τ − τg²/(v̂+ε) + 1`, then
   `α_t = 1/τ`, then the `g` and `v̂` updates in that order — the ordering fix in
   the comment at `adaptive_sghmc.py:171-176` is the correct one. `minv_t`,
   `epsilon_var = 2ε²·mdecay·minv_t − ε⁴`, the momentum update and the position
   update all match. There are no aliasing bugs in the in-place chain (I checked
   each `mul_`/`add_`/`reciprocal_` for tensor reuse).

2. **`scale_grad = N/T` is correct, and the deviation from pybnn is in your
   favour.** `LikCE` uses `reduction='sum'`, the loop divides by the batch size,
   so `scale_grad = N` recovers ∇U over the full data. pybnn injects noise with
   `lr_scaled = lr/√scale_grad` while driving with `lr`, which targets a
   temperature of 1/N; using `lr` in both, as you do, is the self-consistent
   choice. Do not "fix" this back.

3. **The functional-prior gradient matches Wu et al. Eq. 6 exactly.** The
   likelihood term carries the N/n minibatch correction and the prior term
   carries none — `param.grad.add_(-fg/num_datapoints)` followed by
   `×scale_grad` leaves `+Jᵀ K⁻¹(f_M − m)` unscaled by M or N, which is what
   Eq. 6 prescribes. Resampling `X_M` every step is also what Alg. 2 prescribes.

4. **§4.3.26's `B̂` argument is right, and the gradient-noise line should stay
   closed.** I re-derived it. `v̂` estimates E[gradient²]; near a mode the mean
   gradient is small so `v̂ ≈ Σ_g`; the drift injects `ε⁴·minv²·Σ_g ≈ ε⁴`; the
   code subtracts exactly `ε⁴`. The cancellation is exact to the extent
   `v̂ ≈ Σ_g`, and the residual is O(ε⁴) against an O(ε²) thermostat. This is
   the *one* place where the classic SGLD variance-inflation result — Vollmer,
   Zygalakis & Teh (2016) show the SGLD stationary variance is inflated by
   `h·Var(B)/(2A − A²h)`, i.e. proportional to step size × gradient-noise
   variance — does **not** bite you, because you correct it. §4.3.21's
   retraction was correct and §4.3.25's "critical path" genuinely does not
   exist.

5. **Momentum resampling at cycle onset is dimensionally right.** For this
   parameterisation `Var(v)_stationary = 2η/(2−α)` with `η = ε²·minv`; the code
   draws `ε·√minv`, which is the α→0 limit and is within 1.3% in sd at
   `mdecay = 0.195`. Fine.

6. **`max_param_step` and the burn-in-only grad clip are instrumented and read
   zero on the selected runs.** Both are non-measure-preserving and both are
   correctly gated on. No issue.

So: the sampler is not silently sampling the wrong measure. Five refuted
mechanisms plus these six verifications say the same thing — whatever is wrong
is not in the update rule.

---

## 1. The drift gate has never had a null calibration, and it needs one

`_function_space_drift_core` computes, per diagnostic point,

```
z_loc   = |m2 − m1| / sqrt(mcse1² + mcse2²)
z_scale = |log(sd2/sd1)| / sqrt(1/(2·ess1) + 1/(2·ess2))
```

with `ess` from `arviz_stats.ess(·, method="mean")`, and the docstring asserts
these behave like |N(0,1)| under stationarity — "median ~0.67, 95th ~2". §3.6.3
gates trials on median ≤ 2.0. **That assertion is a claim about the ESS
denominators and has never been checked at the τ these chains run at.** §4.5 did
exactly this null calibration for R-hat, and it changed how every R-hat number in
the project is read; the gate that actually rejects trials has not had the same
treatment.

I ran it (simulation code delivered alongside this note as
`scripts_bnn/calibrate_drift_gate.py`, which imports your real gate function
rather than reimplementing it). Perfectly stationary AR(1) chains started from
the stationary law, 16 chains × 75 draws × 400 points — no drift of any kind by
construction:

| true τ (draws) | τ : half-chain | loc_z med | loc_z 95th | scale_z med | scale_z 95th |
|---|---|---|---|---|---|
| 1.0 | 0.03 | 0.736 | 2.218 | 0.744 | 2.147 |
| **2.1** | 0.06 | **1.490** | **4.233** | **1.163** | **3.349** |
| 8.1 | 0.22 | 0.640 | 2.214 | 0.452 | 1.528 |
| 20.0 | 0.54 | 0.518 | 1.498 | 0.410 | 1.256 |
| 28.0 | 0.76 | 0.509 | 1.464 | 0.455 | 1.223 |
| 34.7 | 0.94 | 0.430 | 1.317 | 0.359 | 1.118 |
| 50.0 | 1.35 | 0.403 | 1.122 | 0.355 | 1.150 |

Two things fall out, and they cut in opposite directions from what I expected:

**(a) The gate is NOT anti-conservative at high τ — it goes numb.** At τ = 34.7
the null median `scale_z` is 0.36, half the documented 0.67. So slow mixing on
its own does not manufacture failures, and the 16-of-26 scale failures in
§4.3.73 are *not* an artefact of medium_play being slow. That is a point in
favour of your reading, and I record it as such — I went looking for the
opposite result and did not find it.

**(b) There is a mis-calibration band at τ ≈ 2–5, and medium_diverse sits in
it.** At τ = 2.1 the null median `loc_z` is 1.49 and the null median `scale_z`
is 1.16 — 2.2× and 1.7× the documented 0.67 — with 95th percentiles of 4.23 and
3.35. This is the Geyer initial-positive-sequence estimator under-estimating τ
by ~2× at moderate ρ (my validation block reads τ = 1.9 against a true 4.0 at
ρ = 0.6), which shrinks the denominator and inflates both z-scores by ~√2.
**medium_diverse's measured τ is 2.1 and its settled `scale_z` is 1.7976 — the
worst of the four finals.** Against a null median of 1.16 at that τ, 1.80 is
about 1.5× the noise floor, not the outlier the table in §4.3.46 makes it look
like. medium_diverse's "worst stationarity of the four" reading is substantially
instrument, and the burn-in-100k deviation adopted for it (§4.3.46) may have
been bought against a mis-calibrated number.

**What to do.** Run `calibrate_drift_gate.py --mode null` in your environment,
against the real `arviz_stats`, at each variant's measured τ, and put the
resulting null median and 95th next to every reported `loc_z`/`scale_z` in §7.
The gate is not wrong, but "≤ 2.0" is a different amount of evidence at τ = 2.1
than at τ = 34.7 and the write-up should say so. This is the same disclosure
§4.5 already makes for R-hat.

---

## 2. The gate's own ESS contradicts §4.3.67's τ by 2.5–3.5×

`_function_space_drift_core` computes `ess1`/`ess2` and then **throws them away**
— they appear only inside the `z_scale` denominator and are never returned. That
is a shame, because inverting the formula on the four settled runs' reported
numbers gives a second, independent estimate of the same chains'
autocorrelation, and it disagrees with §4.3.67:

| variant | reported ratio | reported scale_z | implied ESS/half | implied τ (draws) | τ from §4.3.67 | disagreement |
|---|---|---|---|---|---|---|
| medium_play | 1.0871 | 0.6469 | 60.0 | 9.9 | 34.7 | **3.5×** |
| large_diverse | 1.1278 | 0.8091 | 45.3 | 13.1 | 28.0 | **2.1×** |
| medium_diverse | 0.9087 | 1.7976 | 352.5 | 1.7 | 2.1 | 1.2× |
| large_play | 0.9231 | 1.0870 | 184.5 | 3.2 | 8.1 | **2.5×** |

(Approximate: medians do not commute through the formula, so treat this as a
screen, not a measurement. The direction is robust and the disagreement is
largest for the two slow variants.)

The gate sees systematically *faster* mixing than §4.3.67 does, and the gap
grows with τ. Both cannot be right, and it matters a great deal which is:

- **If the gate's τ is right**, medium_play's τ is ~10 draws, not 34.7, and
  §4.3.72's headline pricing — "medium_play needs ~2,000,000 sampling steps,
  ~10× the current budget" — is roughly 3.5× too pessimistic. The whole
  "either budget the 10× or report and disclose" decision in §4.3.72/§10.2 turns
  on this number.
- **If §4.3.67's τ is right**, the gate's ESS is ~3.5× too large on the slow
  variants, its `scale_z` denominator is ~1.9× too small, and a genuine 1.1
  reads as 2.05. That would flip a substantial share of the 16-of-26 failures.

**The fix is two lines and no compute.** Return the ESS from the drift core:

```python
for name, arr in (("loc_z", z_loc), ("scale_z", z_scale),
                  ("loc_sd", raw_loc), ("scale_ratio", ratio),
                  ("ess1", ess1), ("ess2", ess2)):        # <- add
```

and add the §4.3.67 reliability guard *here* as well as in `tail_diagnostics`.
Note the gate operates on **half**-chains (37 draws against τ ≈ 35), i.e. at
τ:chain ≈ 1:1 — twice as far into the unreliable regime as the 2.2:1 that made
§4.3.66 withdraw its headline. The guard was added to the tool that measures τ;
it was never added to the tool that rejects trials.

This is the single cheapest high-value action available, and I would do it
before running the jitter ladder.

---

## 3. The ε ladder (§4.3.41) is confounded, and it tested 5.7% of the step size

§4.3.41 concludes "the selected step size is already at the ceiling", and
§4.3.72 and §4.3.73 both lean on it. The experiment does not support it, for two
independent reasons.

**(a) `sghmc_lr` is three different things at once.** Because `burn_in_lr` is
deliberately `None` in all four configs, `train()` leaves `lr = lr_min` for the
whole burn-in. So `--sghmc_lr` simultaneously sets:

1. the **20k-step burn-in step size**, in the parent warm-up *and* in each
   chain's own burn-in;
2. therefore the operating point at which **`v̂` is estimated and then frozen for
   the entire sampling phase** (`adaptive_sghmc.py:170`);
3. the cool-phase floor and the step size at the harvest point.

The ladder moved the warm-up optimiser, the frozen preconditioner and the
sampler together and attributed the result to (3).

**(b) The ladder barely moved the sampling step size.** Cycle-mean ε² is what
sets the per-step displacement, and `sghmc_lr_max` was held fixed throughout:

| `sghmc_lr` | × selected | cycle-mean ε² | × selected | `gradnorm_pct_over_clip` |
|---|---|---|---|---|
| 1.42e-4 | 1.0× | 3.52e-7 | 1.00 | 0.0003 |
| 5.0e-4 | 3.5× | 5.20e-7 | **1.48** | **60.39** |
| 1.5e-3 | 10.6× | 1.50e-6 | 4.25 | 100.00 |

For large_play, `lr_min²` is only **5.7%** of the cycle-mean ε² (medium_play:
17.1%). The chain already runs at ε = 9.12e-4 at the top of *every single cycle*
and survives it. A **1.48× change in mean ε²** taking `clip%` from 0.0003 to
60.4 is not marginal integrator instability — nothing about the peak step size
changed. It is something breaking upstream, and the obvious candidate is (a):
the 20k burn-in now runs at 3.5× the step size, lands somewhere else, and
freezes a different `v̂`.

**Test, one run, no new code.** Pin `burn_in_lr` to the current `sghmc_lr`
(the plumbing already exists end-to-end: config → `train` → workers) and repeat
the ladder on `sghmc_lr` alone. The config comment "do not add it back (§3.2)"
is right for *reproducing the selection*; it makes the ladder uninterpretable as
a *diagnostic*, and a diagnostic run is not a selection run. If the ladder now
survives 5e-4, "ε is at its ceiling" is refuted and a large lever reopens.

Second test, equally cheap: **ladder `sghmc_lr_max`**, which carries 83–94% of
the step-size budget and has never been laddered at all.

---

## 4. The `v_hat_min` test (§4.3.72) moved the step scale by 8%, not by orders of magnitude

§4.3.72 concludes "step scale refuted — the preconditioner is closed out; it is
compute." Its own table says:

| `v_hat_min` | `minv_median` | `minv_max` | `at_floor` |
|---|---|---|---|
| 1e-4 | (~13.6 implied) | 100 | ~0 |
| 1e-6 | 13.57 | 1000 | 0.011 |
| 1e-8 | 14.71 | 10000 | 0.0009 |

The intervention raised the **cap**, which bound ~1% of elements. It moved
`minv_median` — the thing that actually multiplies the step — by **8%**. τ moved
34.7 → 38.9 → 42.6, which the section itself flags as inside estimator noise.

That refutes **the clamp**, which is precisely what §4.3.71 predicted and
pre-registered. It does not refute *step scale*, because step scale was never
varied by more than 8%. The conclusion in the section heading, and the
"preconditioner line is closed" that follows from it, is a stronger claim than
the experiment supports. A real step-scale test needs a lever that moves
`ε̄²·minv` by an order of magnitude — `sghmc_lr_max` (§3) or `mdecay` (§5).

---

## 5. `fraction_cool` has been inert in every run in this investigation, and is still being swept

`f_pref_net.py:644` says `fraction_cool: (unused; kept for signature
compatibility)`. That docstring is *almost* right, and the sweep did not get the
memo. The harvest condition at `f_pref_net.py:971-980`:

```python
_cool_len = max(1, int(fraction_cool * _cycle_len))
_thin     = max(1, _cool_len // _spc)
_from_end = (_cycle_len - 1) - _cycle_step
if 0 <= _from_end < _spc * _thin and _from_end % _thin == 0:
```

With `samples_per_cycle == 1`, `_thin == _cool_len` and `_spc*_thin == _cool_len`,
so the condition reduces to `_from_end == 0`. I verified numerically: at
`cycle_length = 2750`, the harvest step is `2749` for `fraction_cool` = 0.1,
0.12, 0.34 and 0.5 alike. **`fraction_cool` has no effect whatsoever unless
`samples_per_cycle > 1`, and `sweep_*.yaml` pins `samples_per_cycle: value: 1`
while sweeping `fraction_cool: uniform(0.1, 0.5)`.**

Consequences:

- Every trial in rounds 1–3 that differs only in `fraction_cool` is a **replicate
  under a different name**, and the four settled configs' `fraction_cool` values
  (0.120 / 0.127 / 0.337 / 0.402) are noise fitted by the Bayesian optimiser.
  There is a silver lining: those quadruplets are free replicate estimates of
  the run-to-run floor.
- The observation in §4.5 that "neither swept schedule parameter correlates with
  any mixing metric" has a trivial explanation for one of the two.
- The sweep is searching a dimension that does nothing, which costs surrogate
  quality on the dimensions that do.
- The `sweep_*.yaml` comment at line 125 ("...for `fraction_cool` to take
  effect") records the belief that it was active.

Fix: either drop `fraction_cool` from the sweep, or set `samples_per_cycle > 1`
so it means what the comment thinks it means. Note the second option has a
statistical cost — with `_spc > 1` the harvested draws within a cycle sit at
*different* step sizes (at `fraction_cool = 0.34`, ε ranges over 1×–2.3× lr_min
across the harvest window), so they carry different discretisation bias. I would
drop it from the sweep and keep one draw per cycle.

---

## 6. The 60× "per-variant constant" tracks the two swept sampler scalars

§4.3.67 elevates "steps per independent sample is a per-variant constant
spanning 60×" to the project's one durable invariant, and §4.3.72 concludes it
is a property of the posterior that no knob moves. Both the free-particle
diffusion coefficient of this integrator and the configs say otherwise.

For the update `Δv = −η∇U − αv + N(0, 2αη)`, `Δθ = v` with `η = ε²·minv`, the
position diffusion coefficient is exactly `D = η/α` per step, while the
stationary momentum variance `2η/(2−α)` is essentially α-independent. So α
(`mdecay`) changes the **rate** and — by fluctuation–dissipation, exactly as
§4.3.26 states — leaves the **target** alone. It is the only scalar in the
sampler with that property; ε changes both.

The four configs:

| variant | `mdecay` | cycle-mean ε² | ε̄²/`mdecay` | steps/indep sample |
|---|---|---|---|---|
| medium_diverse | 0.00716 | 6.11e-6 | 8.53e-4 | 1,600 |
| large_play | 0.0312 | 3.52e-7 | 1.13e-5 | 4,000 |
| medium_play | 0.1946 | 3.62e-7 | 1.86e-6 | 95,500 |
| large_diverse | 0.3761 | 2.85e-7 | 7.58e-7 | 77,000 |

`mdecay` spans **52×**, cycle-mean ε² spans **21×**, their ratio spans **1,126×**,
and it orders the four variants correctly on three of four (medium_play and
large_diverse swap, and they differ by only 1.24× in outcome). The mapping from
`D` in weight space to mixing in function space also involves `minv` (measured
on one variant), ‖J‖ and the parameter count, so this is **not** a quantitative
prediction and I am not offering it as one. What it does establish is:

> **The two sampler scalars span far more range across these four
> configurations than the outcome does.** They cannot be dismissed as
> non-levers, and the "per-variant constant" may be a property of what the
> sweep landed on rather than of the posterior. The sweep selects on
> `val_cvar_ce` with mixing entering only through an eligibility gate, so
> nothing in the procedure was ever pushing `mdecay` toward good mixing —
> medium_diverse got 0.0072 and large_diverse got 0.3761 by accident.

**And the friction result was never re-read after the instrument was fixed.**
§4.3.21 tested `mdecay` 0.1946 → 0.08 — a 2.4× move — and read it on
`ess_bulk` 14.57 → 14.81 and `cvar_ess` 46.78 → 26.00. Those are ESS estimates
at τ:chain ≈ 1.8:1, which is exactly the regime §4.3.67 later declared
untrustworthy, added a guard for, and withdrew a headline claim over. The
`mdecay` measurements were never revisited under the corrected reading.

**This is the recommendation with the largest expected payoff.** If `τ ∝ mdecay`
holds even approximately, `mdecay` 0.1946 → 0.02 on medium_play buys the ~10×
that §4.3.72 priced at 2,000,000 sampling steps per chain — for free, and
without touching the target measure. Test it at 0.1946 / 0.05 / 0.01, and read
it on `scale_ratio` and a directly measured diffusion coefficient (§8), **not**
on ESS at 75 draws.

Two things to watch: lower α is more underdamped, so momentum autocorrelation
grows as 1/α (12.5 steps at 0.08, 100 at 0.01 — still ≪ any cycle length here,
so this should be safe); and §4.3.20's dissociation says friction moves tail
precision and stationarity in opposite directions, so expect `cvar_mcse` to get
worse before the extra effective draws pay it back.

---

## 7. What §4.3.73's hypothesis 6 does and does not predict

I made the jitter hypothesis quantitative. Under a single-timescale model, the
ensemble sd after the jitter is exactly

```
s(t) = sqrt(1 + (j² − 1)·exp(−2t/τ))
```

where `j` is the start-to-posterior sd ratio in function space. `j > 1`
contracts, `j < 1` expands, `j = 1` does nothing at any τ. Simulated at
medium_play's geometry (τ = 34.7 draws, burn-in 20,000/2,750 = 7.27 cycles =
0.21 relaxation times, 16 chains × 75 draws):

| j | `scale_ratio` | `scale_z` | shape | gate |
|---|---|---|---|---|
| 0.00 | 1.0596 | 0.435 | expand | pass |
| 1.00 | 1.0062 | 0.377 | flat | pass |
| 2.00 | 0.8706 | 0.573 | contract | pass |
| 3.00 | 0.7399 | 1.173 | contract | pass |

**What it gets right.** The model reproduces the observed contraction magnitude
(`ratio` 0.871 at j = 2 against the observed failure median of 0.837), the sign
flip between variants under one mechanism, and the fast/slow asymmetry: at
medium_diverse's τ = 2.1 with a 26.7-cycle burn-in, *any* `j` from 0 to 3 is
fully absorbed and reads `ratio` 1.0001. §4.3.73's §"why it explains the variant
ordering" survives contact with the arithmetic.

**What it does not get right, and this is the problem.** It cannot produce
`scale_z > 2`. At medium_play's measured τ, even a 3× over-dispersed start reads
`scale_z` 1.17 — the gate is nearly blind to the transient at that τ, exactly as
§1's null table implies. To make the gate fail you need the *ratio* to be driven
by a much slower timescale than the one setting the MCSE:

| j | τ_fast | τ_slow | `ratio` | `scale_z` | gate |
|---|---|---|---|---|---|
| 2.0 | 3.0 | 100 | 0.814 | **5.58** | FAIL |
| 2.0 | 10.0 | 100 | 0.815 | 1.56 | pass |
| 2.0 | **34.7** | 100 | 0.826 | **0.80** | pass |
| 3.0 | 34.7 | 100 | 0.766 | 1.12 | pass |

So: **`scale_z > 2` at `scale_ratio ≈ 0.84` requires the per-point local motion
to be much faster than τ = 34.7.** That is the same contradiction §2 found from a
different direction, and it makes §2's two-line ESS readout the prerequisite for
interpreting the jitter ladder at all.

**Two corrections to the test as designed in §4.3.73.**

1. **Pre-register `scale_ratio`, not `scale_z`, as the readout.** The model says
   `ratio` moves monotonically in `j` while `scale_z` at medium_play's τ barely
   moves. A ladder read on `scale_z` will very likely come back "no effect" and
   refute hypothesis 6 for the wrong reason — which is how five of the previous
   mechanisms died.
2. **Include `j` values above 1.0, not only below.** The current ladder (1.0 /
   0.1 / 0.0) can only walk toward *expansion*. §4.3.73's own data says
   medium_play contracts (0/8 expand ⇒ j > 1) while large_play expands (5/7 ⇒
   j < 1), so the optimum for large_play is *above* 1.0 and the ladder as
   specified cannot find it.

---

## 8. The instrument that would have saved most of §4.3.6–4.3.73

Every mechanism test in this stretch has cost a 220k-step, 16-chain run and then
been read on an ESS estimator that needs τ:chain ≥ 3:1 — which **none** of these
runs achieve. That is why five mechanisms were refuted on evidence that later
turned out to be unusable, and why §4.3.66's headline had to be withdrawn.

The quantity §4.3.67 correctly identified as the invariant is
`steps/independent sample = σ_f²/(2·D_f)`, and **`D_f` can be measured directly,
cheaply, and with a well-conditioned estimator that has no chain-length
requirement at all**:

> From a chain that has finished burn-in, record `f` at the diagnostic points
> every `k` steps for a few thousand steps within a cycle, and regress the mean
> squared displacement `MSD(Δ) = E‖f(t+Δ) − f(t)‖²` on `Δ`. The slope is `2·D_f`.
> `σ_f` you already measure (within-chain σ). Their ratio predicts
> steps-per-independent-sample **before** committing to a 220k-step run.

Properties that matter here:

- **One chain, ~5k steps** instead of 16 chains × 220k steps. Roughly a
  1000-fold reduction in the cost of a hypothesis test.
- **No ESS estimator anywhere.** MSD is a mean over many overlapping lags; its
  reliability requires `Δ_max ≪ n`, not `τ ≪ n`.
- **It separates the two timescales in §7 directly**, because MSD(Δ) is a curve:
  free diffusion is linear in Δ, confinement shows as a plateau at `2σ_f²`, and
  the knee is the relaxation time. That is the measurement that would settle §2.
- **Validatable the way you validated the other instruments** — on an OU process
  with known D and σ, which is what §4.3.66/§4.3.68/§4.3.69/§4.3.70 all did.

With it, §3's ε ladder, §6's friction ladder, §7's jitter ladder and the
open "posterior or settings?" question all become ~1 GPU-hour each. I would
build this before running anything else on the list.

Hook: `f_pref_net.train()` already has the per-step loop and an eval path
(`_eval_current_weights`); a `--msd-probe` mode that appends `self.net(x_diag)`
to a buffer every `k` steps for a window of steps costs one forward pass per
sample and nothing else.

---

## 9. Deviations from the three papers worth disclosing

Not bugs — but §7 should name them, and two of them are testable variants.

1. **Zhang et al.'s exploration stage is `T = 0`, i.e. pure optimisation with no
   injected noise** (Alg. 1: `if r(k) < β: θ ← θ − α∇Ũ(θ)`, else collect with
   SG-MCMC). Their α also decays to **zero**, not to an `lr_min`. Your
   implementation injects full thermal noise across the entire cycle and uses
   `fraction_cool` only — nominally — to choose harvest points. That is a
   defensible variant (it is just SGHMC with a time-varying ε, whose
   continuous-time invariant measure is ε-independent) and arguably a *more*
   correct MCMC than theirs, but it is not the cited algorithm, and it means
   every kept draw follows ~1,800 steps at up to 3.5–32× `lr_min`. Worth naming;
   also worth one run with a genuine β-gated exploration stage, since that is the
   mechanism Zhang et al. use to stop the hot phase contaminating the samples.
2. **Wu et al. Alg. 2 resamples momentum at every outer iteration in both burn-in
   and sampling, and uses a decaying ε** (×0.9 every 5,000 steps in their
   experiments). You resample once per cycle and use a cyclical ε. The first is
   equivalent given your cycle = sample interval; the second is the Zhang et al.
   substitution and is already disclosed.
3. **`bt_pool="mean"` changes the model, not just the scale.** The
   Bradley–Terry likelihood over trajectories is conventionally on the *return*
   (sum). Mean-pooling divides every per-state likelihood gradient by `T` while
   leaving the functional-prior gradient untouched, so it multiplies the
   effective prior-to-likelihood ratio by ~`T`. That is very likely the root of
   both the tiny sampling gradient norms (0.230 on medium_play) that §4.3.70
   found τ tracks, *and* the pressure that drove `map_amp2` 17× above its
   principled value in §4.3.14. `bt_pool` and `map_amp2` are not independent
   knobs. §3.6.2 retains `"mean"` on comparability grounds, which is a fair
   reason, but the interaction belongs in the disclosure.
4. **Each chain freezes its own preconditioner.** Every chain runs its own fresh
   20k burn-in with `tau/g/v_hat` re-initialised to ones, so each freezes a
   *different* `v̂` and therefore runs the whole sampling phase at a different
   effective step `η_i = ε²·minv_i`. Discretisation bias scales with η, so the
   chains sit at slightly different effective temperatures — a **persistent,
   scale-only, between-chain** difference that no amount of burn-in or extra
   draws removes. With `chain_init_jitter = 1.0` (a full per-tensor sd, a large
   perturbation) their burn-in gradient statistics genuinely differ, so this is
   not far-fetched. **`preconditioner_snapshot()` already runs per chain and
   prints to each chain's log**, so the per-chain spread of `precond_minv_median`
   is readable from any existing run's chain logs at zero cost. If it is a few
   percent, forget it; if it is 2×, it is a direct mechanism for `scale_z`
   failing while `loc_z` passes, and it is not a transient.

---

## 10. A better initialisation, which dissolves the tension §4.3.73 calls the real problem

§4.3.73 states the bind exactly: R-hat wants over-dispersed starts,
the §4.2 drift gate wants chains already at stationarity, and with a burn-in
shorter than a relaxation time no `chain_init_jitter` satisfies both.

That is only true because the jitter is *isotropic Gaussian noise in weight
space at a scale nobody calibrated*. There is a start that is over-dispersed
*and* already at the right scale, and it is nearly free:

> **Give chain `i` a thinned state from the shared warm-up trajectory** — save
> the warm-up weights every `num_burn_in_steps / num_chains` steps and hand chain
> `i` the `i`-th. The warm-up already runs 20k steps in the parent process and
> currently throws all but the last state away.

Those states are (a) genuinely different from one another, so R-hat is not
under-dispersed, (b) already at the sampler's own operating scale, so `j ≈ 1`
and the §7 model predicts **no transient at any τ**, and (c) free — no extra
compute, no new hyperparameter, and it removes `chain_init_jitter` from the
sweep. It also removes the §4.3.46 objection to `warmup_use_best` (validation
leakage), because no state is being *selected*.

The obvious caveat: consecutive warm-up states are correlated, so this
under-disperses relative to independent draws — but 20k steps of separation is
7.3 cycles, which for the three fast variants is many relaxation times and for
medium_play is the same 0.21 relaxation times the jitter has to work with
anyway. It is strictly better than a point start and strictly better calibrated
than an arbitrary `j`.

---

## 11. Suggested order of work

Ordered by (information gained) / (compute spent). Items 1–3 need no training
runs at all.

1. **Return `ess1`/`ess2` from `_function_space_drift_core`** and add the
   §4.3.67 τ:chain guard to it. Two lines. Resolves §2, which gates the
   interpretation of everything else including the jitter ladder.
2. **Run `calibrate_drift_gate.py` in your environment** and put null medians
   and 95ths beside every gate number in §7. Resolves §1 and re-reads
   medium_diverse.
3. **Grep any surviving chain logs for `[precond] FROZEN`** and tabulate the
   per-chain spread of `precond_minv_median` (§9.4).
4. **Drop `fraction_cool` from the sweep** (§5), and re-read any conclusion that
   rested on it.
5. **Build the MSD probe** (§8). One afternoon; it changes the cost of every
   remaining question by ~3 orders of magnitude.
6. **`mdecay` ladder** 0.1946 / 0.05 / 0.01 on medium_play, read on MSD-derived
   `D_f` and `scale_ratio` (§6). This is the one with a real chance of buying the
   10×.
7. **ε ladder with `burn_in_lr` pinned**, plus a first-ever `sghmc_lr_max`
   ladder (§3).
8. **Jitter ladder** with `j` above *and* below 1.0, read on `scale_ratio`
   (§7) — or skip it in favour of thinned warm-up starts (§10), which removes the
   parameter instead of tuning it.

---

## Sources

- Wu, Xuan & Lu (2025), *Functional Stochastic Gradient MCMC for Bayesian Neural
  Networks*, AISTATS — `wu25b (1).pdf`, Eq. 6/9, Table 1, Alg. 1–2, App. C–D.
- Springenberg, Klein, Falkner & Hutter (2016), *Bayesian Optimization with
  Robust Bayesian Neural Networks*, NeurIPS — Alg. 1, scale adaptation.
- Zhang, Li, Zhang, Chen & Wilson (2020), *Cyclical Stochastic Gradient MCMC for
  Bayesian Deep Learning*, ICLR — Eq. 1, §3.1–3.2, Alg. 1.
  <https://openreview.net/pdf/faffe100c21426695486861801522350e9587630.pdf>
- Chen, Fox & Guestrin (2014), *Stochastic Gradient Hamiltonian Monte Carlo*,
  ICML. <https://arxiv.org/abs/1402.4102>
- Vollmer, Zygalakis & Teh (2016), *Exploration of the (Non-)Asymptotic Bias and
  Variance of Stochastic Gradient Langevin Dynamics*, JMLR 17 —
  stationary-variance inflation `h·Var(B)/(2A − A²h)`, and the mSGLD correction.
  <https://jmlr.org/papers/v17/15-494.html>
- Betancourt (2015), *The Fundamental Incompatibility of Scalable Hamiltonian
  Monte Carlo and Naive Data Subsampling*, ICML.
  <https://proceedings.mlr.press/v37/betancourt15.html>
- Franzese, Milios, Filippone & Michiardi (2022), *Revisiting the Effects of
  Stochasticity for Hamiltonian Samplers*, ICML — the O(η²) ergodic-error floor
  under minibatching.
  <https://proceedings.mlr.press/v162/franzese22a/franzese22a.pdf>
