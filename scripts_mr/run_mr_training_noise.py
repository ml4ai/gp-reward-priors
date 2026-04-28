import copy
import os
import os.path as osp
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pyrallis
import torch
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
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
    width: int = 64
    depth: int = 3
    activations: str = "relu"
    # training params
    dataset_id: str = "D4RL_antmaze-medium-play-v2"
    dataset: str = "~/busy-beeway/transformers/pen_labels/AdroitHandPen-v1_pref.hdf5"
    label_flip: float = 0.0
    epochs: int = 10
    batch_size: int = 256  # Batch size for all networks
    lr: float = 3e-4
    criteria_key: str = "acc"
    pin_memory: bool = True
    # general params
    seed: int = 0
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path
    prior: Optional[str] = None
    prior_ckpt: int = 1000

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                osp.expanduser(self.checkpoints_path), self.name
            )
        if self.prior:
            if self.prior != "FG":
                self.prior = os.path.join(
                    osp.expanduser(self.prior),
                    f"br-{self.dataset_id}-{self.width}-{self.depth}",
                )


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
    dataset = util.Pref_H5Dataset(dataset, -1, config.label_flip)
    state_shape, action_shape = dataset.shapes()
    state_dim = state_shape[2]
    action_dim = action_shape[2]

    training_data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.pin_memory,
    )

    interval = len(dataset) / config.batch_size
    if int(interval) < interval:
        interval = int(interval + 1)
    else:
        interval = int(interval)

    net = MLP(
        state_dim + action_dim, 1, [config.width] * config.depth, config.activations
    ).to(device)
    if config.prior:
        if config.prior == "FG":
            prior = FixedGaussianPrior(std=1.0).to(device)
        else:
            ckpt_path = os.path.join(
                config.prior, "ckpts", "it-{}.ckpt".format(config.prior_ckpt)
            )
            prior = OptimGaussianPrior(ckpt_path).to(device)
    else:
        prior = None
    net_optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    model = MRTrainer(
        net,
        opt=net_optimizer,
        num_datapoints=len(training_data),
        prior=prior,
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
            train_batch = [b.to(torch.float32).to(device) for b in train_batch]
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