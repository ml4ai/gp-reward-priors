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
