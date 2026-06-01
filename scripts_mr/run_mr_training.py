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
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

from optbnn.bnn.nets.mlp import MLP
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
    training_split: float = 0.7
    label_flip: float = 0.0  # fraction of training labels to flip (0 = none, 1 = all)
    data_reduction: float = 0.0  # fraction of training data to remove after split (0 = none, 1 = all)
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
    seed: int = 0
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                osp.expanduser(self.checkpoints_path), self.name
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
    dataset = util.Pref_H5Dataset(dataset, -1)
    state_shape, action_shape = dataset.shapes()
    state_dim = state_shape[2]
    action_dim = action_shape[2]
    full_training = config.training_split == 1.0

    if full_training:
        training_data = dataset
        if config.label_flip > 0.0:
            if config.label_flip == 1.0:
                dataset.labels = 1.0 - dataset.labels
            else:
                num_to_flip = int(len(dataset) * config.label_flip)
                indices_to_flip = np.random.choice(len(dataset), num_to_flip, replace=False)
                dataset.labels[indices_to_flip] = 1.0 - dataset.labels[indices_to_flip]
    else:
        training_data, test_data = random_split(
            dataset, [config.training_split, 1 - config.training_split]
        )
        # Apply label flipping to the training split only, so test labels stay clean.
        if config.label_flip > 0.0:
            train_indices = np.array(training_data.indices)
            if config.label_flip == 1.0:
                dataset.labels[train_indices] = 1.0 - dataset.labels[train_indices]
            else:
                num_to_flip = int(len(train_indices) * config.label_flip)
                flip_positions = np.random.choice(len(train_indices), num_to_flip, replace=False)
                indices_to_flip = train_indices[flip_positions]
                dataset.labels[indices_to_flip] = 1.0 - dataset.labels[indices_to_flip]

    # Randomly discard data_reduction fraction of training points (test data unaffected).
    if config.data_reduction > 0.0:
        if config.data_reduction == 1.0:
            print("data_reduction=1.0: no training data remains. Exiting.")
            sys.exit(0)
        n_train = len(training_data)
        n_keep = int(n_train * (1.0 - config.data_reduction))
        keep_positions = np.random.choice(n_train, n_keep, replace=False)
        if full_training:
            training_data = Subset(dataset, keep_positions)
        else:
            keep_indices = np.array(training_data.indices)[keep_positions]
            training_data = Subset(dataset, keep_indices)

    persistent = config.num_workers > 0
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
    training_data_loader = DataLoader(training_data, shuffle=True, **loader_kwargs)
    if not full_training:
        test_data_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    net = MLP(
        state_dim + action_dim, 1, [config.width] * config.depth, config.activations
    ).to(device)
    if config.compile_model:
        net = torch.compile(net)
    net_optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    model = MRTrainer(
        net,
        opt=net_optimizer,
        num_datapoints=len(training_data),
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
        }
        if full_training:
            metrics[f"training_{config.criteria_key}_best"] = (
                best_acc if config.criteria_key == "acc" else best_loss
            )
        else:
            metrics["eval_loss"] = []
            metrics["eval_acc"] = []
            metrics[f"eval_{config.criteria_key}_best"] = (
                best_acc if config.criteria_key == "acc" else best_loss
            )

        if epoch:
            for train_batch in training_data_loader:
                train_batch = [b.to(device, non_blocking=True) for b in train_batch]
                for key, val in model.train(train_batch).items():
                    metrics[key].append(val)
        else:
            metrics["training_loss"] = np.nan

        if full_training:
            # Best model tracked by training metrics; skip epoch 0 (no training yet).
            if epoch:
                loss = np.mean(metrics["training_loss"])
                acc = np.mean(metrics["training_acc"])
                if config.criteria_key == "acc":
                    if acc > best_acc:
                        c_best_epoch = epoch
                        best_acc = acc
                        metrics["best_epoch"] = c_best_epoch
                        metrics["training_acc_best"] = best_acc
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
                            metrics["training_acc_best"] = best_acc
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
                        metrics["training_loss_best"] = best_loss
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
                            metrics["training_loss_best"] = best_loss
                            if config.checkpoints_path is not None:
                                torch.save(
                                    model.state_dict(),
                                    os.path.join(config.checkpoints_path, "best_model.pt"),
                                )
                    else:
                        if acc > best_acc:
                            best_acc = acc
        else:
            # eval phase
            if epoch % config.eval_every == 0:
                for test_batch in test_data_loader:
                    test_batch = [b.to(device, non_blocking=True) for b in test_batch]
                    for key, val in model.evaluation(test_batch).items():
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
    sys.exit(0)


if __name__ == "__main__":
    train()
