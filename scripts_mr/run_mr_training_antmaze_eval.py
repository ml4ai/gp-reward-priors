import os
import os.path as osp
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

from optbnn.bnn.nets.mlp import MLP
from optbnn.training.checkpoint_selection import CheckpointSelector
from optbnn.training.training import MRTrainer
from optbnn.utils import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    # wandb params
    project: str = "MR-training"
    group: str = "MR"
    name: str = "mr"
    # model params
    width: int = 6  # log2 exponent; actual width = 2**width (e.g. 6 → 64)
    depth: int = 3
    activations: str = "relu"
    # BT trajectory pooling, shared with BNN/PT: "mean" (masked mean over valid
    # timesteps) or "sum" (legacy).  Must match across all three for comparability.
    bt_pool: str = "mean"
    # training params
    dataset_id: str = "D4RL_antmaze-medium-play-v2"
    # Antmaze evaluation data.  Train / validation / test sets are loaded from
    # the per-seed eval directory:
    #   {data_root}/{antmaze_variant}/eval/seed_{seed}/{antmaze_variant}_pref_{train,val,test}_{seed}.hdf5
    # The same seed drives training and file selection, so the model seed and the
    # loaded data splits always match.
    #
    # Split roles (handoff 4.3.107, adopted 2026-09-15): the SELECTION split
    # (select_split, default "test") picks the checkpoint saved as best_model.pt;
    # the VALIDATION split scores that checkpoint, and eval_loss_at_selected is
    # the stage-1 sweep objective.  select_split="val" reproduces the old rule,
    # where validation did both and a held-out test score was logged at the end.
    antmaze_variant: str = "antmaze-medium-play-v2"
    data_root: str = "data/antmaze"
    # Derived from antmaze_variant + seed in __post_init__ when left unset.  Set
    # explicitly only to override — e.g. a reduction/ or noise/ subdirectory file.
    train_dataset: Optional[str] = None
    val_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    epochs: int = 10
    batch_size: int = 256  # Batch size for all networks
    lr: float = 3e-4
    eval_every: int = 1  # How often (time steps) we evaluate
    criteria_key: str = "acc"
    select_split: str = "test"  # split that picks best_model.pt: "test" (current) or "val" (pre-2026-09-15)
    num_workers: int = 4  # DataLoader worker processes
    prefetch_factor: int = 2  # Batches pre-loaded per worker (ignored when num_workers=0)
    compile_model: bool = False  # Wrap net with torch.compile for kernel fusion
    pin_memory: bool = True
    # general params
    seed: int = 1  # antmaze eval data seeds are 1..10; also selects the data files
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path

    def __post_init__(self):
        if self.select_split not in ("test", "val"):
            raise ValueError(f"select_split must be 'test' or 'val', got {self.select_split!r}")
        self.width = 2 ** self.width
        # Derive the pre-split train / validation paths from the antmaze variant
        # and seed so the loaded data files always match the run's seed.  An
        # explicit train_dataset / val_dataset (if given) takes precedence.
        eval_seed_dir = os.path.join(
            self.data_root, self.antmaze_variant, "eval", f"seed_{self.seed}"
        )
        prefix = f"{self.antmaze_variant}_pref"
        if self.train_dataset is None:
            self.train_dataset = os.path.join(
                eval_seed_dir, f"{prefix}_train_{self.seed}.hdf5"
            )
        if self.val_dataset is None:
            self.val_dataset = os.path.join(
                eval_seed_dir, f"{prefix}_val_{self.seed}.hdf5"
            )
        if self.test_dataset is None:
            self.test_dataset = os.path.join(
                eval_seed_dir, f"{prefix}_test_{self.seed}.hdf5"
            )
        # The wandb run name keeps a uuid for uniqueness across launches, but the
        # on-disk checkpoints directory is deterministic: {checkpoints_path}_{seed},
        # with no uuid leaf.  This lets the IQL eval stage (iql_eval.py) derive the
        # exact reward-model directory from the seed alone.  Exactly one training
        # run per (variant, seed) writes here; a re-run overwrites in place.
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = f"{osp.expanduser(self.checkpoints_path)}_{self.seed}"


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    util.set_seed(config.seed)
    # Load the pre-split train and validation sets from separate files.
    train_data = util.Pref_H5Dataset(osp.expanduser(config.train_dataset), -1)
    val_data = util.Pref_H5Dataset(osp.expanduser(config.val_dataset), -1)
    state_shape, action_shape = train_data.shapes()
    state_dim = state_shape[2]
    action_dim = action_shape[2]

    persistent = config.num_workers > 0
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
    training_data_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_data_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_data = util.Pref_H5Dataset(osp.expanduser(config.test_dataset), -1)
    test_data_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
    print(f"[split roles] checkpoint selection on {config.select_split}; "
          f"sweep objective scored on val")

    net = MLP(
        state_dim + action_dim, 1, [config.width] * config.depth, config.activations
    ).to(device)
    if config.compile_model:
        net = torch.compile(net)
    net_optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    model = MRTrainer(
        net,
        opt=net_optimizer,
        num_datapoints=len(train_data),
        device=device,
        bt_pool=config.bt_pool,
    )
    selector = CheckpointSelector(config.criteria_key)

    for epoch in range(config.epochs + 1):
        metrics = {
            "training_loss": [],
            "training_acc": [],
            **selector.log_dict(),
        }

        if epoch:
            for train_batch in training_data_loader:
                train_batch = [b.to(device, non_blocking=True) for b in train_batch]
                for key, val in model.train(train_batch).items():
                    metrics[key].append(val)
        else:
            metrics["training_loss"] = np.nan

        # eval phase — the validation split is scored every eval epoch (it is the
        # sweep objective); the selection split picks the checkpoint.
        if epoch % config.eval_every == 0:
            val_loss, val_acc = _evaluate(model, val_data_loader)
            if config.select_split == "test":
                sel_loss, sel_acc = _evaluate(model, test_data_loader)
            else:
                sel_loss, sel_acc = val_loss, val_acc
            metrics.update(eval_loss=val_loss, eval_acc=val_acc,
                           select_loss=sel_loss, select_acc=sel_acc)

            if config.checkpoints_path is not None:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{epoch}.pt"),
                )

            if selector.update(epoch, sel_loss, sel_acc, val_loss, val_acc):
                if config.checkpoints_path is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(config.checkpoints_path, "best_model.pt"),
                    )
            metrics.update(selector.log_dict())

        # Drop metrics that weren't computed this epoch (empty lists) instead of
        # logging them as NaN — otherwise training_acc at epoch 0 logs NaN.
        metrics = {
            key: (np.mean(val) if isinstance(val, list) else val)
            for key, val in metrics.items()
            if not (isinstance(val, list) and not len(val))
        }
        wandb.log(metrics, step=epoch)

    _final_check(config, model, selector, val_data_loader, test_data_loader)
    sys.exit(0)


def _evaluate(model, loader):
    """Mean over batches of the trainer's eval loss / accuracy (as logged before)."""
    loss, acc = [], []
    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        out = model.evaluation(batch)
        loss.append(out["eval_loss"])
        acc.append(out["eval_acc"])
    return float(np.mean(loss)), float(np.mean(acc))


def _final_check(config, model, selector, val_data_loader, test_data_loader):
    """Reload best_model.pt and re-score it.

    select_split="test": the test split chose the checkpoint, so a test score
    would be in-sample.  Instead re-score VAL and check it equals the logged
    eval_loss_at_selected -- this verifies best_model.pt is the selected
    checkpoint.  select_split="val": the old behaviour, a held-out test score.
    """
    if selector.best_epoch is None:
        print("[final] no evaluation produced a finite selection metric; nothing to reload")
        wandb.log({"selection_failed": 1})
        return
    if config.checkpoints_path is None:
        print("[final] no checkpoints_path -- best model was not saved; skipping reload check")
        return
    best_path = os.path.join(config.checkpoints_path, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"[final] reloaded {best_path} (selected epoch {selector.best_epoch}, "
          f"on {config.select_split})")
    if config.select_split == "test":
        loss, acc = _evaluate(model, val_data_loader)
        diff = abs(loss - selector.val_loss)
        ok = diff <= 1e-5 * max(1.0, abs(selector.val_loss))
        print(f"[final] val loss reloaded {loss:.6f} vs logged at selection "
              f"{selector.val_loss:.6f} -> {'MATCH' if ok else 'MISMATCH'}")
        wandb.log({"eval_loss_reloaded": loss, "eval_acc_reloaded": acc,
                   "reload_check_abs_diff": diff, "reload_check_ok": int(ok)})
    else:
        loss, acc = _evaluate(model, test_data_loader)
        print(f"[final] held-out test_acc = {acc:.4f}, test_loss = {loss:.4f}")
        wandb.log({"test_loss": loss, "test_acc": acc})


if __name__ == "__main__":
    train()
