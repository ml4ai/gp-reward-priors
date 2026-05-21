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

from optbnn.bnn.nets.pref_trans import PT
from optbnn.training.training import PTTrainer
from optbnn.utils import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    # wandb params
    project: str = "PT-training"
    group: str = "PT"
    name: str = "pt"
    # model params
    embd_dim: int = 256
    pref_attn_embd_dim: Optional[int] = None
    num_heads: int = 4
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
    dataset: str = "~/busy-beeway/transformers/pen_labels/AdroitHandPen-v1_pref.hdf5"
    label_flip: float = 0.0
    epochs: int = 10
    batch_size: int = 256  # Batch size for all networks
    lr: float = 3e-4
    criteria_key: str = "acc"
    num_workers: int = 4  # DataLoader worker processes
    prefetch_factor: int = 2  # Batches pre-loaded per worker (ignored when num_workers=0)
    compile_model: bool = False  # Wrap net with torch.compile for kernel fusion
    pin_memory: bool = True
    # general params
    seed: int = 0
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path

    def __post_init__(self):
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
    dataset = osp.expanduser(config.dataset)
    dataset = util.Pref_H5Dataset(dataset, label_flip=config.label_flip)
    state_shape, action_shape = dataset.shapes()
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
    training_data_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    max_pos = config.default_max_pos
    while query_len > max_pos:
        max_pos *= 2

    net = PT(
        state_dim,
        action_dim,
        dataset.max_episode_length(),
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
            f"training_{config.criteria_key}_best": (
                best_acc if config.criteria_key == "acc" else best_loss
            ),
        }
        for train_batch in training_data_loader:
            train_batch = [b.to(device, non_blocking=True) for b in train_batch]
            for key, val in model.train(train_batch).items():
                metrics[key].append(val)

        loss = np.mean(metrics["training_loss"])
        acc = np.mean(metrics["training_acc"])

        if config.criteria_key == "acc":
            if acc > best_acc:
                c_best_epoch = epoch
                best_acc = acc
                metrics["best_epoch"] = c_best_epoch
                metrics[f"training_acc_best"] = best_acc
                if config.checkpoints_path is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(config.checkpoints_path, f"best_model.pt"),
                    )
                if loss < best_loss:
                    best_loss = loss
            elif acc == best_acc:
                if loss < best_loss:
                    c_best_epoch = epoch
                    best_loss = loss
                    metrics["best_epoch"] = c_best_epoch
                    metrics[f"training_acc_best"] = best_acc
                    if config.checkpoints_path is not None:
                        torch.save(
                            model.state_dict(),
                            os.path.join(config.checkpoints_path, f"best_model.pt"),
                        )
            else:
                if loss < best_loss:
                    best_loss = loss
        else:
            if loss < best_loss:
                c_best_epoch = epoch
                best_loss = loss
                metrics["best_epoch"] = c_best_epoch
                metrics[f"training_loss_best"] = best_loss
                if config.checkpoints_path is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(config.checkpoints_path, f"best_model.pt"),
                    )
                if acc > best_acc:
                    best_acc = acc
            elif loss == best_loss:
                if acc > best_acc:
                    c_best_epoch = epoch
                    best_acc = acc
                    metrics["best_epoch"] = c_best_epoch
                    metrics[f"training_loss_best"] = best_loss
                    if config.checkpoints_path is not None:
                        torch.save(
                            model.state_dict(),
                            os.path.join(config.checkpoints_path, f"best_model.pt"),
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
    sys.exit(0)


if __name__ == "__main__":
    train()
