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

from optbnn.utils import util
from optbnn.bnn.nets.pref_trans import PT
from optbnn.training.training import PTTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    # wandb params
    project: str = "PT-training"
    group: str = "PT"
    name: str = "pt"
    # model params
    embd_dim: int = 8  # log2 exponent; actual embd_dim = 2**embd_dim (e.g. 8 → 256)
    pref_attn_embd_dim: Optional[int] = None
    head_dim: int = 6  # log2 exponent; actual head_dim = 2**head_dim (e.g. 6 → 64); num_heads = embd_dim // head_dim
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    intermediate_dim: Optional[int] = None
    num_layers: int = 1
    embd_dropout: float = 0.1
    model_eps: float = 0.1
    max_ep_length: Optional[int] = None
    default_max_pos: int = 2048
    # training params
    dataset_id: str = "D4RL/pen-v2"
    # Antmaze evaluation data.  Train / validation / test sets are loaded from
    # the per-seed eval directory:
    #   {data_root}/{antmaze_variant}/eval/seed_{seed}/{antmaze_variant}_pref_{train,val,test}_{seed}.hdf5
    # The same seed drives training and file selection, so the model seed and the
    # loaded data splits always match.  Training uses the train set with the
    # validation set for best-model selection; after training the best model is
    # reloaded and evaluated once on the held-out test set (test_acc / test_loss).
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
    num_workers: int = 4  # DataLoader worker processes
    prefetch_factor: int = 2  # Batches pre-loaded per worker (ignored when num_workers=0)
    compile_model: bool = False  # Wrap net with torch.compile for kernel fusion
    pin_memory: bool = True
    # general params
    seed: int = 1  # antmaze eval data seeds are 1..10; also selects the data files
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path

    def __post_init__(self):
        self.embd_dim = 2 ** self.embd_dim
        self.head_dim = 2 ** self.head_dim
        self.head_dim = min(self.head_dim, self.embd_dim)  # clamp so num_heads >= 1
        self.num_heads = self.embd_dim // self.head_dim
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
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                osp.expanduser(self.checkpoints_path), self.name
            )
        if self.pref_attn_embd_dim is None:
            self.pref_attn_embd_dim = self.embd_dim
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.embd_dim


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
    train_data = util.Pref_H5Dataset(osp.expanduser(config.train_dataset))
    val_data = util.Pref_H5Dataset(osp.expanduser(config.val_dataset))
    state_shape, action_shape = train_data.shapes()
    _, query_len, state_dim = state_shape
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

    max_pos = config.default_max_pos
    while query_len > max_pos:
        max_pos *= 2

    net = PT(
        state_dim,
        action_dim,
        train_data.max_episode_length(),
        config.embd_dim,
        config.pref_attn_embd_dim,
        config.num_heads,
        config.attn_dropout,
        config.resid_dropout,
        config.intermediate_dim,
        config.num_layers,
        config.embd_dropout,
        max_pos,
        config.model_eps,
    ).to(device)
    if config.compile_model:
        net = torch.compile(net)

    net_optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    model = PTTrainer(
        net,
        opt=net_optimizer,
        device=device,
    )
    c_best_epoch = 0

    best_acc = -np.inf
    best_loss = np.inf

    for epoch in range(config.epochs + 1):
        metrics = {
            "training_loss": [],
            "training_acc": [],
            "best_epoch": c_best_epoch,
            "eval_loss": [],
            "eval_acc": [],
            f"eval_{config.criteria_key}_best": (
                best_acc if config.criteria_key == "acc" else best_loss
            ),
        }

        if epoch:
            for train_batch in training_data_loader:
                train_batch = [b.to(device, non_blocking=True) for b in train_batch]
                for key, val in model.train(train_batch).items():
                    metrics[key].append(val)
        else:
            metrics["training_loss"] = np.nan

        # eval phase — evaluate on the held-out validation set
        if epoch % config.eval_every == 0:
            for val_batch in val_data_loader:
                val_batch = [b.to(device, non_blocking=True) for b in val_batch]
                for key, val in model.evaluation(val_batch).items():
                    metrics[key].append(val)

            loss = np.mean(metrics["eval_loss"])
            acc = np.mean(metrics["eval_acc"])

            if config.criteria_key == "acc":
                if acc > best_acc:
                    c_best_epoch = epoch
                    best_acc = acc
                    metrics["best_epoch"] = c_best_epoch
                    metrics["eval_acc_best"] = best_acc
                    if config.checkpoints_path is not None:
                        torch.save(
                            model.state_dict(),
                            os.path.join(config.checkpoints_path, "best_model.pt"),
                        )
                    if loss < best_loss:
                        best_loss = loss
                elif acc == best_acc:
                    if loss < best_loss:
                        c_best_epoch = epoch
                        best_loss = loss
                        metrics["best_epoch"] = c_best_epoch
                        metrics["eval_acc_best"] = best_acc
                        if config.checkpoints_path is not None:
                            torch.save(
                                model.state_dict(),
                                os.path.join(config.checkpoints_path, "best_model.pt"),
                            )
                else:
                    if loss < best_loss:
                        best_loss = loss
            else:
                if loss < best_loss:
                    c_best_epoch = epoch
                    best_loss = loss
                    metrics["best_epoch"] = c_best_epoch
                    metrics["eval_loss_best"] = best_loss
                    if config.checkpoints_path is not None:
                        torch.save(
                            model.state_dict(),
                            os.path.join(config.checkpoints_path, "best_model.pt"),
                        )
                    if acc > best_acc:
                        best_acc = acc
                elif loss == best_loss:
                    if acc > best_acc:
                        c_best_epoch = epoch
                        best_acc = acc
                        metrics["best_epoch"] = c_best_epoch
                        metrics["eval_loss_best"] = best_loss
                        if config.checkpoints_path is not None:
                            torch.save(
                                model.state_dict(),
                                os.path.join(config.checkpoints_path, "best_model.pt"),
                            )
                else:
                    if acc > best_acc:
                        best_acc = acc

        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val):
                    metrics[key] = np.mean(val)
                else:
                    metrics[key] = np.nan
        wandb.log(metrics, step=epoch)

    # ------------------------------------------------------------------ #
    # Final test-set evaluation — reload the best (validation-selected) model
    # and evaluate it once on the held-out test set, logging preference
    # accuracy and loss as test_acc / test_loss.
    # ------------------------------------------------------------------ #
    if config.checkpoints_path is not None:
        best_path = os.path.join(config.checkpoints_path, "best_model.pt")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[test] reloaded best model from {best_path} (best epoch {c_best_epoch})")
    else:
        print("[test] no checkpoints_path — evaluating the final-epoch model on test")

    test_data = util.Pref_H5Dataset(osp.expanduser(config.test_dataset))
    test_data_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
    test_loss, test_acc = [], []
    for test_batch in test_data_loader:
        test_batch = [b.to(device, non_blocking=True) for b in test_batch]
        eval_out = model.evaluation(test_batch)
        test_loss.append(eval_out["eval_loss"])
        test_acc.append(eval_out["eval_acc"])
    test_metrics = {
        "test_loss": float(np.mean(test_loss)),
        "test_acc": float(np.mean(test_acc)),
    }
    print(
        f"[test] test_acc = {test_metrics['test_acc']:.4f}, "
        f"test_loss = {test_metrics['test_loss']:.4f}"
    )
    wandb.log(test_metrics)
    sys.exit(0)


if __name__ == "__main__":
    train()
