# Codebase Handoff Document

> Generated 2026-06-07. Covers work done across two coding-assistant sessions on the `gp_reward-priors` project.

---

## 1. Project Overview

This project trains **reward models** for preference-based offline RL on D4RL AntMaze environments. Two model families are supported:

| Family | Script | Architecture | Trainer class |
|--------|--------|--------------|---------------|
| **MR** (MLP Reward) | `scripts_mr/run_mr_training.py` | `MLP` from `optbnn.bnn.nets.mlp` | `MRTrainer` |
| **PT** (Preference Transformer) | `scripts_pt/run_pt_training.py` | `PT` from `optbnn.bnn.nets.pref_trans` | `PTTrainer` |

Both scripts share an identical training loop structure and config API (pyrallis + wandb). Hyperparameter search is done with **WandB Bayesian optimization sweeps**.

A third trainer class, **`MRTrainerF`**, was implemented in `optbnn/training/training.py` for functional MAP estimation using a GP prior (`LCFModel`). Its scripts were created and then deleted at the user's request — the class itself remains in `training.py`.

> **⚠️ READ THIS FIRST.** §§1–6 below were written by a *parallel session* that worked only on the **MR** and **PT** families. They do **not** cover the third, fully-active model family — the **fSGHMC BNN** pipeline (`scripts_bnn/run_bnn_training.py`) — which a separate session built and which is the subject of most recent work. **§7 is the authoritative record for the BNN/fSGHMC pipeline.** Where §§1–6 conflict with §7, §7 wins. In particular, §6's claim that the functional-MAP runner scripts were "deleted" and its reference to `scripts_bnn/bb_optim_f.py` are **stale**: that script was renamed and is now the live `scripts_bnn/run_bnn_training.py` (see §7).

---

## 2. Key Files and Their Current State

### Training scripts

#### `scripts_mr/run_mr_training.py`
- Trains an MLP reward model on preference pair data.
- **No weight-space prior** — `MRTrainer` uses pure cross-entropy loss.
- Consolidated from a former `run_mr_training_noise.py`; `label_flip` is now a config parameter.

#### `scripts_pt/run_pt_training.py`
- Trains a Preference Transformer on preference pair data.
- Consolidated from a former `run_pt_training_noise.py`; same `label_flip` pattern.

### Trainers: `optbnn/training/training.py`

- **`MRTrainer`**: CE loss only, no prior. Simple gradient step.
- **`MRTrainerF`**: Functional MAP trainer. Uses `LCFModel.functional_prior_grad()` to inject GP prior gradient directly into `.grad` before each Adam step. Logs `training_loss = CE - log_p_GP / N`. Not currently used by any script (those scripts were deleted), but the class is production-ready.
- **`PTTrainer`**: Preference Transformer trainer, unchanged.

### Dataset utility: `optbnn/utils/util.py` — `Pref_H5Dataset`

- `label_flip` parameter was **removed** from `__init__`. Labels are always loaded as-is.
- Label flipping now happens post-split inside the training scripts (see §3 below).

### YAML config files

**MR base configs** (4 files, `scripts_mr/antmaze_*_mr.yaml`):
- `width: 8` — note this is a **log2 exponent** (actual width = 256); set by sweep, not the base config default.
- `depth: 2`, `batch_size: 64`, `lr: 0.0003`, `epochs: 10`, `training_split: 0.8`

**PT base configs** (4 files, `scripts_pt/antmaze_*_pt.yaml`):
- `embd_dim: 8` (log2, actual = 256), `head_dim: 6` (log2, actual = 64 → `num_heads = 4`)
- `batch_size: 64`, `lr: 0.0003`, `epochs: 10`, `training_split: 0.8`, `compile_model: true`
- ⚠️ These formerly had `num_heads: 5` (an unrecognized field after rename, and also non-divisible). Fixed to `head_dim: 6`.

**MR sweep configs** (8 files, 4 base + 4 `_best` variants in `scripts_mr/`):
- Method: `bayes`, `run_cap: 40`, metric: `eval_acc_best` (maximize)
- Sweep params: `width` (`int_uniform`, 6–9), `depth` (`int_uniform`, 1–6), `lr` (`log_uniform_values`, 1e-5–1e-2)

**PT sweep configs** (8 files, 4 base + 4 `_best` variants in `scripts_pt/`):
- Method: `bayes`, `run_cap: 50`, metric: `eval_acc_best` (maximize)
- Sweep params: `embd_dim` (`int_uniform`, 6–8), `head_dim` (`int_uniform`, 5–7), `num_layers` (`int_uniform`, 1–4), `lr` (`log_uniform_values`, 1e-5–1e-2)

---

## 3. Design Decisions and Their Rationale

### 3.1 Post-split label flipping

**Old approach**: `Pref_H5Dataset.__init__` accepted `label_flip` and applied it to the entire dataset at load time. This contaminated test labels.

**New approach**: Labels are always loaded clean. After `random_split`, the training script applies flipping **only to `training_data.indices`** by indexing `dataset.labels` directly. Test data is never touched.

```python
train_indices = np.array(training_data.indices)
flip_positions = np.random.choice(len(train_indices), num_to_flip, replace=False)
dataset.labels[train_indices[flip_positions]] = 1.0 - dataset.labels[train_indices[flip_positions]]
```

In `full_training` mode (split = 1.0), flipping is applied to the whole dataset (there is no test set).

### 3.2 `data_reduction` avoids nested Subsets

`random_split` returns `Subset` objects. Rather than wrapping them in another `Subset` (which would produce `Subset(Subset(dataset, ...))` and require double-indexing), the code remaps through `training_data.indices` to always produce a **flat** `Subset(dataset, flat_indices)`.

```python
if full_training:
    training_data = Subset(dataset, keep_positions)          # positions == indices
else:
    keep_indices = np.array(training_data.indices)[keep_positions]
    training_data = Subset(dataset, keep_indices)            # always flat
```

### 3.3 `full_training` mode (`training_split = 1.0`)

When `training_split == 1.0`:
- No `test_data_loader` is created.
- The eval phase in the epoch loop is skipped entirely.
- Best model is selected by **training metrics** (primary: `training_acc`, secondary: `training_loss`) rather than eval metrics.
- WandB logs `training_acc_best` / `training_loss_best` instead of `eval_acc_best` / `eval_loss_best`.

### 3.4 Log2 exponent encoding for exponential hyperparameters

WandB Bayesian optimization uses a Gaussian process internally. For hyperparameters that should be searched on a log scale (e.g., network width 64/128/256/512), supplying the raw value gives the GP a badly-scaled space.

**Solution**: Store and sweep the **log2 exponent** as an integer, convert in `__post_init__`.

- `width` in `TrainConfig` (MR): sweep `int_uniform` over [6, 9]; `__post_init__` does `self.width = 2 ** self.width`.
- `embd_dim` and `head_dim` in `TrainConfig` (PT): sweep `int_uniform` over [6,8] and [5,7] respectively; `__post_init__` converts both.

The wandb config logged at run start shows the **actual** (converted) values because `asdict(config)` is called after `__post_init__`.

### 3.5 `head_dim` reparametrisation for PT sweeps

**Problem**: `num_heads` must evenly divide `embd_dim`. If a sweep samples them independently, most combinations are invalid and many will cause a `ZeroDivisionError` inside `GPT2SelfAttention`.

**Solution**: Sweep `head_dim` (size per head) instead of `num_heads`. Since both `embd_dim` and `head_dim` are powers of 2, `num_heads = embd_dim // head_dim` is always a valid integer as long as `head_dim ≤ embd_dim`.

**Edge case**: When the sweep samples `head_dim_exp > embd_dim_exp` (e.g., `head_dim=128 > embd_dim=64`), `num_heads` would be 0 → `ZeroDivisionError`. Fixed by clamping in `__post_init__`:

```python
self.embd_dim = 2 ** self.embd_dim
self.head_dim = 2 ** self.head_dim
self.head_dim = min(self.head_dim, self.embd_dim)  # clamp so num_heads >= 1
self.num_heads = self.embd_dim // self.head_dim
```

This makes those runs equivalent to `num_heads = 1`, and the Bayesian optimizer quickly learns the effective boundary.

### 3.6 `MRTrainerF`: Functional MAP

Implements the Wu et al. (AISTATS 2025) functional MAP objective: `U(f) = CE_loss + I₀(f)` where `I₀` is the Onsager-Machlup functional for the GP prior.

Implementation strategy:
1. Call `loss.backward()` to accumulate CE gradients in `param.grad`.
2. Call `self.prior.functional_prior_grad(net, X_M, ...)` to get the prior gradient.
3. **Add** (not subtract, because Adam minimises) the prior gradient divided by `N` into `param.grad`.
4. Call `opt.step()`.

The training loss reported to WandB is `CE - log_p_GP / N` (the full potential energy), not just CE.

---

## 4. User Preferences and Style

- **Consolidate, don't proliferate**: When noise/variant scripts diverge from a base script by only one flag, roll them into the base script as a config parameter. The user deleted standalone `*_noise.py` scripts once the base script absorbed the flag.
- **Post-hoc cleanup**: The user is comfortable manually deleting files (e.g., `run_mr_training_f.py`) after deciding not to use them. Don't assume every file created will stay.
- **Minimal diffs**: Prefer small, targeted edits over rewrites. The user reviews changes closely.
- **No redundant YAML files**: Old sweep/config YAML variants (FG, optim, optim_star, etc.) were deleted; only the `_mr` and `_pt` family is retained. Don't create new YAML variants without being asked.
- **WandB metric name**: The user manually changed `eval_acc` → `eval_acc_best` in sweep configs. Always use the `_best` suffix for the sweep optimization metric.
- **Commit granularity**: One logical change per commit. The user appreciates descriptive commit messages (e.g., `"Clamp head_dim to embd_dim in __post_init__ to prevent ZeroDivisionError"`).
- **No optional priors**: The user removed weight-space prior support from `MRTrainer` entirely. Don't re-introduce optional prior parameters unless explicitly asked.
- **Test data is sacred**: Any data manipulation (label flip, data reduction) must leave test data untouched. This is a hard correctness requirement.
- **Sweep `run_cap`**: 40 for MR (3 hyperparameters), 50 for PT (4 hyperparameters). Scale by number of parameters being swept.

---

## 5. Known Bugs Fixed

| Bug | Cause | Fix | Commit |
|-----|-------|-----|--------|
| `pyrallis.FieldError: fields {'num_heads': 5} do not belong to class` | PT base YAMLs still had `num_heads: 5` after field was renamed to `head_dim` | Replaced `num_heads: 5` → `head_dim: 6` in all 4 PT base configs | `c0c4422` |
| `ZeroDivisionError` in `GPT2SelfAttention.__init__` | Bayesian sweep could sample `head_dim_exp > embd_dim_exp`, giving `num_heads = 0` | Clamp `head_dim = min(head_dim, embd_dim)` before computing `num_heads` | `837d77d` |

---

## 6. Potential Future Work (Not Requested)

- `MRTrainerF` has no runner script. If functional MAP experiments are resumed, a `run_mr_training_f.py` would need to be recreated. The key reference for how to load the `LCFModel` prior and the measurement dataloader is `scripts_bnn/bb_optim_f.py`. The source function (e.g., `antmaze_task_reward_prior`) should be loaded dynamically from `optbnn/gp/reward_functions.py` via a string config argument.
- The `.ipynb_checkpoints/` directories inside `scripts_mr/` and `scripts_pt/` contain many stale YAML snapshots. These are gitignored by Jupyter but may be worth cleaning up.
- The `_best` sweep variants (e.g., `sweep_antmaze_medium_play_mr_best.yaml`) are present but not described in this session. They appear to be multi-seed re-evaluation sweeps (commit `fafc1bf`). Clarify their intended use before modifying them.

---

## 7. The fSGHMC BNN Pipeline (authoritative — separate session)

This section documents the **third model family**, built in its own session. It is the
most actively developed pipeline and supersedes any conflicting note above.

### 7.1 What it is

A Bayesian neural-network reward model sampled with **functional SGHMC (fSGHMC)**
following Wu et al. (2025). Instead of a weight-space Gaussian prior, the SGHMC
update uses a **functional GP prior gradient** computed from a `LCFModel`
(linear-combination-of-features GP). This replaces the older, now-deleted
"OptimGaussianPrior / star" tuning pipeline entirely.

| Item | Value |
|------|-------|
| Entry-point script | `scripts_bnn/run_bnn_training.py` |
| Sampler class | `FPrefNet` in `optbnn/sgmcmc_bayes_net/f_pref_net.py` (standalone — no `BayesNet`/`PrefNet`/`OptimGaussianPrior` inheritance) |
| GP prior | `LCFModel` in `optbnn/gp/models/model.py` |
| Source reward functions | `optbnn/gp/reward_functions.py` (chosen at runtime by string name) |
| Config API | pyrallis `TrainConfig` + `--config_path` YAML + wandb |
| BB base config | `scripts_bnn/bb_bnn.yaml` (+ `sweep_bb_bnn.yaml`) |
| AntMaze base configs | `scripts_bnn/antmaze_{medium,large}_{play,diverse}_bnn.yaml` |

### 7.2 The model

`LCFModel` defines `f(x) = Φ(x)ᵀ w`, `w ~ N(p_mean, p_covariance)`, inducing a
**degenerate (finite-rank) GP** with mean `m(x) = Φ(x)ᵀ p_mean` and kernel
`K(x,x') = Φ(x)ᵀ Σ Φ(x')`. `Φ` is the feature map produced by a source function
in `reward_functions.py` (signature `f(X, device)` or `f(X, aux_X, device)`).

Key methods added to `LCFModel`: `compute_mean`, `compute_covariance`,
`sample_functions`, `solve_prior`, `log_prior`, `functional_prior_grad`, plus
private helpers `_feature_matrix`, `_alpha_from_phi`, `_woodbury_solve`.

`functional_prior_grad(net, X_M, aux_X, jitter)` returns
`∇_w log p_GP = -J_w(X_M)ᵀ K⁻¹(f(X_M;w) - m(X_M))` via a single VJP
(`torch.autograd.grad`, which does NOT touch `.grad`).

### 7.3 The training loop (`FPrefNet.train`)

Per step:
1. Twin-network preference forward pass → `fx_batch` (trajectory-summed reward
   pairs; `obs_dim = d_dim - 1`, last column is the attention mask).
2. `lik_loss = LikCE(fx_batch, y) / batch_size`; `lik_loss.backward()`.
3. If `n_meas > 0`: sample `n_meas` measurement points, call
   `functional_prior_grad`, and add `-fg / num_datapoints` into each
   `param.grad`. **Scale adaptation**: AdaptiveSGHMC later multiplies by
   `scale_grad = num_datapoints`, so the prior's effective contribution is
   `O(1)` — same order as the weight-space prior it replaces.
4. `clip_grad_norm_` + `sampler.step()`.

Cyclical SGHMC: alternates hot (exploration) / cool (sampling) phases; one
sample collected at the end of each cool phase. AdaptiveSGHMC adapts the
preconditioner during burn-in, then freezes it.

`warmup_log_every > 0` + `eval_data` ⇒ logs `warmup/nll`, `warmup/acc`,
`warmup/step` during burn-in (single-weight forward pass, since no posterior
samples exist yet).

Parallel chains run via `sample_multi_chains_parallel` → `mp.spawn`. The worker
`_fpref_chain_worker` is module-level and picklable; `gp_prior_args` carries
numpy arrays + a module-level function reference; the `LCFModel` is
reconstructed inside each worker.

### 7.4 Measurement (tuning) data

The GP prior gradient is evaluated on a **separate** measurement HDF5, loaded by
`util.load_measurement_data(meas_path)`:
- `"obs"` — (N, obs_dim) **required** → passed as `X`.
- `"aux_obs"` — (N, K) **optional** → passed as `aux_X`; returns `None` when absent
  (then `LCFModel` calls `function_vect(X, device)` with no aux arg).

For AntMaze these are the `data/antmaze/<env>_tuning_set.hdf5` files produced by
the `scripts_bnn/create_tuning_set_antmaze_*.py` generators (≈999k rows,
`aux_obs` = 4-d `[x, y, goal_x, goal_y]`).

### 7.5 The degenerate-GP numerics fix (important)

Because `K = ΦΣΦᵀ` has rank ≤ `n_concepts`, a direct `(n_M, n_M)` Cholesky is
singular up to the jitter and **fails** ("leading minor of order n_concepts+1
not positive-definite") once a feature's scale grows. Fixed via:
- **`_woodbury_solve`**: solves `α = (K+εI)⁻¹ r` and `log|K+εI|` in the `(d, d)`
  weight space (Woodbury identity + matrix-determinant lemma). Exact,
  well-conditioned, cheaper when `n_M ≫ d`.
- **Relative nugget**: `ε = jitter · max(mean(diag K), 1)` — stays effective at any
  feature scale without retuning `meas_jitter`. Equivalent to the valid GP
  `k_eff = ΦᵀΣΦ + ε·δ` (finite-rank GP + observation noise).
- **`n_M = 0` guard**: `n_meas = 0` (prior off) short-circuits to empty `α`,
  `logdet = 0` — both in `_woodbury_solve` and in `FPrefNet.train` (which skips
  the whole prior block). Use `n_meas=0` to get a **pure-likelihood baseline**.

### 7.6 Reward functions (`optbnn/gp/reward_functions.py`)

All module-level (picklable), signature `(X, aux_X, device)`, returning an
`(n, n_concepts)` double tensor. Available: `pen_task_reward_prior` (6 feats),
`pen_task_reward_prior_no_intercept` (5), `bb_reward_prior`,
`antmaze_task_reward_prior` (3 feats: `[intercept, -goal_distance,
reached_goal]`).

**Convention learned: the intercept is always feature index 0 when present, and
its `p_mean` coefficient must be 0** (we are uncertain of its sign). Non-intercept
coefficients default to 1. So `p_mean = [0, 1, 1, ...]`, set in the script after
`n_concepts` is known. Any new threshold/indicator feature should use `<=`
(e.g. "1.3 or less" → `goal_distance <= 1.3`) and guard any division denominator
(`.clamp(min=...)`) so the feature can never produce inf/nan (which makes the GP
covariance undefined).

### 7.7 `run_bnn_training.py` config & behavior

- **`reward_function: str`** — loaded via `getattr(reward_functions, name)`; validated at startup.
- **`input_dim` inferred** from data: `X_train.shape[-1] - 1`.
- **`n_concepts`** inferred by probing the source function on a 1-row dummy input
  (or set explicitly via the `n_concepts` config field).
- **`p_mean = ones(n_concepts)` then `p_mean[0] = 0`** (intercept).
- **`width` is a log2 exponent** (`self.width = 2 ** self.width` in `__post_init__`),
  matching the MR/PT convention (§3.4). When filling sweep results into configs,
  paste the **exponent**, not the actual width.
- **`training_split`**: `0<split<1` holds out a test set (metrics logged `test_*`);
  **`training_split == 1.0` trains on the FULL dataset** (in-sample, metrics
  `train_*`). This was consolidated from a former standalone
  `run_bnn_full_training_f.py`, which was then deleted.
- **`label_flip` / `data_reduction`** (0..1): apply to **training data only**; test
  data untouched (the project's hard "test data is sacred" rule). One-hot labels
  ⇒ a flip is `1.0 - y`. `data_reduction == 1.0` ⇒ `wandb.finish()` + clean return
  (no exception — sweep-safe).
- **Warm-up early stop** — `early_stop_acc_threshold` (Optional[float], default 0.6):
  after warm-up, if `warmup_final_acc < threshold`, skip the parallel-chain phase
  and finish cleanly (`wandb.finish()` + `return`, **never an exception**, so a
  wandb sweep records a completed run). `None` disables. **Originally NLL-based
  (`ln 2`); switched to accuracy** because the trajectory-sum Bradley-Terry logit
  saturates the softmax, inflating NLL well above `ln 2` even for accurate models
  (high accuracy + high NLL = overconfident errors). Accuracy is the meaningful
  signal. `early_stopped` (0/1) is logged for sweep filtering.

### 7.8 MAP removed

All MAP-estimate code was removed from `FPrefNet`: no `find_map`, no `self.map`,
no `use_map`/`map_only`. **All metrics use the posterior predictive mean** over
sampled weights (`predict`, `eval_test_data`). `_eval_current_weights` is the
single-weight warm-up monitor (used before any samples exist). Do not reintroduce
MAP options.

### 7.9 AntMaze configs and sweeps (this session's deliverables)

**Base configs** `scripts_bnn/antmaze_<size>_<type>_bnn.yaml` (4): copies of
`bb_bnn.yaml` adapted per env — `dataset`/`dataset_id` from the matching
`scripts_mr/*_mr.yaml`, `OUT_DIR` mirroring the mr `checkpoints_path` scheme
(`./exp/reward_learning/antmaze_<size>_<type>_bnn`), `measurement_dataset =
data/antmaze/<env>_tuning_set.hdf5`, `reward_function: antmaze_task_reward_prior`.
(The user later set `project/group/name` to `BNN-training`/`BNN`/`bnn`.)

**Bayesian sweeps** `sweep_antmaze_*_bnn.yaml` (4): `method: bayes`,
`run_cap: 70` (user raised from 60), maximize `warmup_final_acc`. Data-driven
design from inspecting the HDF5s:
- pref sets are **small** (363–734 pairs) → fixed `batch_size: 64`,
  `num_burn_in_steps: 5000`.
- `antmaze_task_reward_prior` features are **scale-imbalanced** (`-goal_dist`
  spans 0..−42; `reached_goal` fires only ~7–10%), so `K` is distance-dominated
  → **`gp_cov_scale` is the dominant knob**, swept wide log `1e-2..1e3`.
- `n_meas` swept `int_uniform 0..64` (**0 = prior off**, so the optimizer can
  learn whether the prior helps). Also swept: `width` (exp 6..10), `depth` (2..6),
  `sghmc_lr` (log 1e-4..1e-2), `mdecay` (log 1e-3..1e-1).
- **Efficiency trick**: `warmup_final_acc` is logged *before* chain sampling, so
  these sweeps fix `early_stop_acc_threshold: 1.01` to skip the expensive chain
  phase entirely — every sweep run is warm-up-only, same metric, far cheaper.

**Multi-seed "best" re-eval sweeps** `sweep_antmaze_*_{bnn,mr,pt}_best.yaml` (12,
across all three pipelines): `method: grid`, all hyperparameters as literal
`PLACEHOLDER` (user fills in the bayes winners), plus `seed: [0..9]` (→ 10 runs
for mean ± std), and **`training_split: 1.0`** (full-dataset final eval).
`config_path` is kept real (a placeholder there would break the run). The user has
filled in several mr/pt `_best` files with bayes results. **For the bnn `_best`
files, the user will want `early_stop_acc_threshold` set to a real value (or the
config default), NOT `1.01`, so the full chain phase actually runs.**

### 7.10 Repo hygiene done this session

- Renames (history-preserving): `bb_optim_f.py → run_bnn_training.py`,
  `bb_optim_f.yaml → bb_bnn.yaml`, `sweep_bb_optim_f.yaml → sweep_bb_bnn.yaml`
  (all internal refs, wandb run-name suffix, and `OUT_DIR` defaults updated).
- Deleted 62 obsolete YAMLs and 16 obsolete entry-point scripts from
  `scripts_bnn/` (the antmaze/pen/bb gaussian-prior / optim-star / FG / tuning
  pipeline). **`create_*` tuning-set generators were explicitly preserved** —
  do not delete anything prefixed `create`.
- `batch_size` default set to `64` across all 12 antmaze configs (bnn/mr/pt;
  pt was 256 → 64, a real behavioral change for PT — revisit its lr/epochs if
  needed).

### 7.11 Additional preferences learned (this session)

- **Verify, then commit — and run with the right interpreter.** The user expects
  changes validated before committing. Use `/opt/anaconda3/envs/irl/bin/python`
  (or set `PYTHONPATH` to the repo root) so `import optbnn` works — a bare
  `python /tmp/script.py` puts `/tmp` on `sys.path` and silently fails to import
  the package. Be honest when a verification step didn't actually run.
- **Commit AND push each logical change**; descriptive messages; the user often
  says "commit and push". Commit trailer in use: `Co-Authored-By: Claude ...`.
- **Data-grounded recommendations**: when asked for sweep ranges etc., the user
  wants them tailored to the actual data — inspect the HDF5s first.
- **Single source of truth for config values**: when a value becomes the base-config
  default, remove now-redundant sweep overrides of it.
- The user is comfortable with the assistant making reasonable structural choices
  (sectioning, naming) but flags when defaults differ from theirs (e.g. they reset
  `project/group/name`). Surface such choices explicitly.
- **macOS Stop-hook notification**: at the user's request a user-level
  `~/.claude/settings.json` Stop hook was added that fires an `osascript` desktop
  notification when a turn ends (env is macOS, paths under `/Users/champlin`).
