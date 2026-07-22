# Hand-off: CVaR convergence diagnostics + fSGHMC sampler fixes

> Generated August 2026. Covers the diagnostics-and-sampler-fixes phase that followed
> `HANDOFF.md` (2026-06) and the external review `fsghmc_sampler_fixes_handoff.md`.
> Audience: a coding assistant joining this project cold. Line numbers drift —
> **locate by the code shown, not by line number**.
>
> Known erratum in the older `HANDOFF.md`: §7.4 states the antmaze `aux_obs` column
> order incorrectly. The actual order is `[goal_x, goal_y, x, y]`.

---

## 1. Project context

Preference-learning reward models on D4RL antmaze (4 variants: medium/large ×
play/diverse). Three model families that **must share one likelihood** for
comparability:

- **BNN** — the main method: functional SGHMC (`FPrefNet`,
  `optbnn/sgmcmc_bayes_net/f_pref_net.py`) with a functional GP prior
  (map-informed heat-kernel maze prior; see `HANDOFF.md`). Posterior weight
  samples → reward functions.
- **MR** — MLP reward baseline (`MRTrainer` in `optbnn/training/training.py`).
- **PT** — Preference Transformer baseline (`PTTrainer`, same file).

**The downstream quantity is CVaR₀.₀₅ — the mean of the lowest 5% of the
per-point posterior-predictive reward** — used as a pessimistic reward for
offline RL. Everything in this phase serves one question: *is that lower tail
sampled well enough, and without bias, to trust?*

Active training entrypoints (a 2026-07 refactor dropped the old `*_sampling_best`
sweeps): `scripts_bnn/run_bnn_training_antmaze_eval.py` (+ `scripts_mr/`,
`scripts_pt/` analogues) with per-variant configs `antmaze_<variant>_*_antmaze_eval.yaml`.
`seed` selects BOTH the sampler seed AND the per-seed data split
`{data_root}/{variant}/eval/seed_{seed}/{variant}_pref_{train,val,test}_{seed}.hdf5`;
`OUT_DIR` gets `_{seed}` appended in `__post_init__`. Batch launcher:
`train_rewards.sh` (repo root).

## 2. Operational conventions (violating these cost us real debugging time)

- **Two machines.** Analysis Mac: conda env `irl`, invoke as
  `/opt/anaconda3/envs/irl/bin/python` (bare `python` = base env, no torch; a
  local git hook even blocks Bash commands containing the word "python" — write
  commit messages to a file and use `git commit -F`). GPU box: Ubuntu 22.04.5,
  6× RTX A6000, conda env **`pt`** — launchers use `python` on PATH and assume
  the env is activated.
- **Absolute paths on the box.** pyrallis opens `--config_path`, and the configs'
  `data_root` / `measurement_dataset`, relative to the process CWD. Always pass
  absolute `--config_path`, `--data_root`, `--measurement_dataset`
  (`train_rewards.sh` and `scripts_bnn/gradnorm_readout/launch*.sh` show the pattern).
- **`data/` is gitignored** — eval splits and `*_tuning_set.hdf5` must be copied
  to the box manually. Preflight their existence before launching.
- **Sampling chains are `mp.spawn` workers with NO wandb run.** To surface a
  worker-side metric: write a file under `OUT_DIR/sampling_f/chain_<i>/` and
  aggregate in the main-process eval block (see `grad_norm_stats.pt`). Workers
  also build their **own** `FPrefNet` — constructor params must be threaded
  explicitly (bundled as `_fpref_kwargs` in `sample_multi_chains_parallel`);
  forgetting this silently reverts workers to defaults.
- **Git: commit and push directly to `master`.** No branches, no PRs (sole
  contributor). The GPU box syncs by `git pull`, so anything it needs must be
  committed.

## 3. Diagnostics built (and the statistics behind them)

Motivation: the run summaries only had weight-space and bulk predictive
diagnostics. Both are wrong for this project: weight-space R-hat/ESS are
**meaningless** for a BNN (permutation/scale non-identifiability → `param_rhat`
~2–4, `param_ess` ~5 forever; ignore them), and bulk predictive diagnostics
certify the **median**, not the tail the CVaR reads.

Added to the eval block of `run_bnn_training_antmaze_eval.py` (per split, prefix
`{label}_`; also in legacy `scripts_bnn/run_bnn_training.py`), computed on
`pred_chains [chain, draw, point]` over the first 64 pairs × valid (non-padded)
timesteps, `alpha = 0.05`:

| metric family | method | reads as |
|---|---|---|
| `pred_q05_ess_*`, `pred_q05_mcse_*` | `arviz_stats` `ess/mcse(method="quantile", prob=0.05)` | VaR (the 95% lower bound) |
| `pred_folded_rhat_*` | `rhat(method="folded")` | tail/scale-sensitive chain agreement |
| `pred_cvar_ess_*`, `pred_cvar_rhat_*`, `pred_cvar_mcse_rel_*` | Rockafellar–Uryasev: `CVaR = VaR + (1/α)·E[min(X−VaR,0)]` is **exact**, so CVaR's MC error = mean-ESS/MCSE of the integrand `u`; R-hat = folded on `u` | **the downstream quantity** |
| `*_mcse_rel_*` | MCSE ÷ per-point predictive sd | scale-free; reward magnitude spans orders of magnitude, so only the **relative** `_max` is trustworthy |
| `gradnorm_{burnin,sampling}_{max,mean,pct_over_clip}` | pre-clip grad-norm accumulated per phase in `train()`, saved per chain, aggregated in eval | is the grad clip modifying the sampled dynamics? |

`arviz_stats` 1.1.0 quirks (cost a round-trip): `rhat` has NO `"tail"` method
(valid: z_scale/rank/identity/folded/split — use `folded`); `ess(method="tail")`
requires `prob`; prefer `method="quantile", prob=0.05` for the lower tail
specifically.

Offline validator (no re-sampling): `scripts_bnn/diagnose_sampling_tail.py
--run-dir <OUT_DIR> [--worst-k 20]` — recomputes bulk+VaR+CVaR from saved chains
(reproduces logged numbers exactly) and lists the worst-K points **with torso
(x, y) = obs[:, :2]** so you can judge whether unresolved points are real states.

Interpretation heuristics that held throughout: rank-R-hat high with folded-R-hat
low = between-chain **location** offset (chains not merged); high ESS with
elevated R-hat = chains individually mixed but settled in different modes; and
**shared chain starts under-estimate R-hat** (see `chain_init_jitter`, §5).

## 4. Empirical findings timeline (old runs, old sum-pooling reward scale)

Original 4-chain × 155-draw runs → re-runs at 8 × 310 (4× draws), CVaR summary:

| variant | CVaR ESS min (620→2480 draws) | CVaR rel-MCSE max | CVaR R-hat max | note |
|---|---|---|---|---|
| medium_diverse | 185 → 740 | 0.94 → 0.56 | 1.03 → 1.015 | clean; also fixed a *pervasive* bulk location offset (rank-R-hat 96%→3.9% over 1.01) |
| medium_play | 83 → 173 | 2.14 → 1.52 | 1.06 → 1.027 | `pred_within_chain_var` exploded 29 → 943k (upper-side output blow-up; foreshadowed Issue 3) |
| large_play | 38 → 190 | 2.65 → 1.26 | 1.14 → **1.11** | R-hat max barely moved: draws can't merge modes |
| large_diverse | 26 → 106 | 2.23 → 1.053 | 1.10 → 1.064 | nearly resolved |

`--worst-k` on large_play: the unresolved points are **real states, ~14 of the
worst 20 from a single trajectory (pair 54) tracing the maze's left (x≈0) and top
(y≈9) boundary** — where the wall-respecting prior is legitimately multimodal.
Draws don't fix that; between-chain mixing does. Note: these CVaR values are on
the OLD (sum-pooled) reward scale — **not comparable to post-fix runs**; only the
scale-free diagnostics carry over.

## 5. Fixes implemented (all flag-gated; defaults = corrected behaviour)

The external review flagged 3 issues; all are done, plus one addition of ours.
Priority was re-ordered for this project (we had already brute-forced draw count,
so Issue 3's bias question outranked Issue 2's).

| flag (TrainConfig, BNN eval script) | default | legacy value | what it does |
|---|---|---|---|
| `bt_pool` | `"mean"` | `"sum"` | **shared Bradley–Terry pooling across BNN/MR/PT** |
| `clip_grad_norm_value` / `clip_during_sampling` | `100.0` / `False` | `True` | clip scoped to burn-in; norm still measured every step |
| `samples_per_cycle` | `1` | `1` | cool-phase harvesting; **`num_samples` is now the TOTAL per-chain count**, reached in `ceil(num_samples/spc)` cycles |
| `resample_momentum` | `True` | `False` | momentum resampled at cycle start (vs zeroed) |
| `chain_init_jitter` | `0.0` | `0.0` | overdisperse per-chain starts around the shared warm-up point |

Details, with the load-bearing subtleties:

- **Issue 3 (`bt_pool`) — the big one.** BNN and MR summed per-timestep rewards
  over the trajectory (`nansum`); PT averaged (÷T). Sum makes the logit scale
  grow with T and, with a deep net (~`w^(depth+1)`), produces rare enormous
  gradients (instrumentation measured pre-clip norms to **17,417** during
  sampling) that the always-on clip was absorbing — a non-measure-preserving
  modification that **fattens the lower tail, biasing CVaR conservative**.
  Fix: one helper `optbnn/utils/util.py: bt_pool_logit[_np]` — masked mean
  `nansum(r·m)/Σm` — applied at **every** train+eval pooling site in
  `f_pref_net.py` and `training.py` (`MRTrainer`, `MRTrainerF`, `PTTrainer`).
  For full-length trajectories masked-mean equals PT's old ÷T, so PT is
  unchanged there. `"sum"` reproduces legacy BNN/MR (NOT legacy PT).
  **This rescales the reward** — the entire reason a fresh HP sweep is required.
- **Issue 2 (`samples_per_cycle`) — the wall-clock lever.** Previously 1 sample
  per full 1250–2750-step cycle (`fraction_cool` was dead). Now harvests `spc`
  thinned samples from each cycle's cool tail (last `fraction_cool·cycle_len`
  steps, spaced to end at the coldest step). `spc=1` reproduces legacy
  **byte-for-byte** (verified: identical step budget and collected steps).
  `spc=k` ≈ k× fewer cycles for the same sample count. Caveats: cool draws are
  autocorrelated (tail-ESS sublinear in spc) and fewer cycles = fewer hot-phase
  mode jumps — keep enough cycles (~80–100) or the multimodal worst points regress.
- **Issue 1 (`resample_momentum`).** The AdaptiveSGHMC `momentum` buffer is the
  position increment `v ≈ εM⁻¹z`; its OU stationary law is `N(0, lr²·minv_t)`,
  `minv_t = 1/(√v̂+ε)`, `lr = lr_max` at the cycle boundary. Plain SGHMC: `√lr`.
  Verified the std matches and `v_hat` is not mutated. Trap: `state["g"]` is
  Springenberg's smoothed gradient, not anything momentum-like.
- **(Ours) `chain_init_jitter` + worker threading fix.** All chains previously
  launched from the identical warm-up point → R-hat under-estimated. Jitter
  perturbs each chain's start by `jitter · per-tensor std`, per-chain seeded.
  Honest framing: if R-hat *rises* under jitter, that's real non-convergence
  being revealed; the cure is hot-phase mixing (`lr_max`, `fraction_cool`), not
  less jitter. While wiring this we fixed a latent bug: `bt_pool`/clip flags were
  **not reaching the worker's FPrefNet** (workers used defaults) — now threaded
  via `_fpref_kwargs`.

## 6. Verification (post-fix readouts)

Cheap readout kit: `scripts_bnn/gradnorm_readout/` (temporary — delete along
with `exp/_gradnorm_readout/` when the sweep lands). `launch.sh` runs all 4
variants at reduced budget (2 chains × 12 cycles, full 5000-step burn-in,
wandb disabled) on disjoint GPUs; `read_results.py` prints per-phase grad-norm
stats straight from the `.pt` files.

Post-fix (mean pooling, clip off), sampling phase:

| variant | %>clip | max | mean | verdict |
|---|---|---|---|---|
| large_play | 0.02% | 831 | 5.9 | clip inert |
| large_diverse | 0.01% | 680 | 3.4 | clip inert |
| medium_diverse | 0.53% | 1,746 | 17.9 | improved |
| medium_play | 0.83% | **4.9e7** | 4,239 | **diverged** |

The medium_play divergence is the review's predicted Step-2 diagnostic: the
always-on clip had been masking an instability; mean pooling also shrinks
gradients → `v_hat` shrinks → `minv_t` pins at its cap (`1/√v_hat_min` = 100) →
max amplification under the OLD `lr_max`. The `lr_max` mini-sweep
(`launch_lrmax.sh` / `read_lrmax.py`, one variable, everything else post-fix):

| lr_max | max | mean | |
|---|---|---|---|
| 0.006402 (old HP) | 2.2e7 | 1575 | diverges |
| 0.0048 | 5,176 | 8.0 | stable |
| 0.0032 | 2,058 | 8.4 | stable |
| 0.0024 | 446 | 8.0 | stable |
| 0.0016 | 414 | 8.0 | stable |

**Cliff between 0.0048 and 0.0064; no kernel-conditioning problem** (lowest
rungs clean → `meas_jitter` needs no change). Post-fix gradient scale is ~3–18
against a clip threshold of 100 → `clip_during_sampling=False` is safe.

## 7. Current state and what's next

**The HP re-selection sweep is green-lit** and is the next major step. Guidance
agreed with the user (champlin):

1. **Do NOT check accuracy/CE invariance under old HPs** — mean pooling rescales
   the reward so old optima shift; same-HP accuracy differences are expected and
   not a bug signal. Accuracy is judged only at each model's re-swept optimum.
2. Cap medium_play's `lr_max` search ≲0.005 (centre 0.0016–0.0032). Other
   variants' old `lr_max` values were stable. Keep `max_param_step=0.5` (it, not
   the clip, is the real blow-up guard); a divergent draw scores poorly rather
   than crashing.
3. Sweepable levers from this phase: `samples_per_cycle` (wall-clock vs
   tail-ESS), `chain_init_jitter` + `lr_max`/`fraction_cool` (the combination
   that should finally resolve the large_play boundary-trajectory points —
   verify with the worst-K diagnostic). Candidate: raise `v_hat_min` if configs
   keep pinning `minv_t` at 100.
4. Acceptance metrics for everything: `pred_cvar_ess_min`,
   `pred_cvar_mcse_rel_max`, `pred_cvar_rhat_max` / worst-K R-hat, and
   `gradnorm_sampling_pct_over_clip` ~ 0. Post-fix reward values are a NEW
   scale; compare convergence (scale-free) metrics only against old runs.

## 8. Do-NOTs (accumulated the hard way)

- Do not judge this sampler by `param_*` or bulk `pred_rhat`/`pred_ess`.
- Do not re-enable `clip_during_sampling` to "fix" instability — it biases the
  CVaR tail; fix HPs (or, if instrumentation implicates it, kernel conditioning).
- Do not compare post-fix CVaR/reward magnitudes to pre-fix runs.
- Do not assume worker processes inherit main-process FPrefNet settings — thread
  through `_fpref_kwargs` / `train_kwargs`.
- Do not pass relative paths to anything pyrallis/h5py opens on the box.
- Do not expect `bt_pool="sum"` to reproduce legacy **PT** (its legacy was ÷T).
- Do not alter the GP prior semantics or add a momentum network — out of scope
  per the original review, still true.
