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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

import optbnn.gp.reward_functions as reward_functions
from optbnn.bnn.nets.mlp import MLP
from optbnn.gp.models.model import LCFModel
from optbnn.training.training import MRTrainerF
from optbnn.utils import util
from optbnn.utils.util import load_measurement_data

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
    batch_size: int = 256  # Batch size for training data loader
    lr: float = 3e-4
    criteria_key: str = "acc"
    num_workers: int = 4  # DataLoader worker processes
    prefetch_factor: int = 2  # Batches pre-loaded per worker (ignored when num_workers=0)
    compile_model: bool = False  # Wrap net with torch.compile for kernel fusion
    pin_memory: bool = True
    # functional GP prior params
    source_fn: str = "antmaze_task_reward_prior"  # name of function in optbnn/gp/reward_functions.py
    measurement_dataset: str = "data/bb/bbway_tuning_set.hdf5"  # HDF5 readable by load_measurement_data()
    meas_batch_size: int = 256  # Batch size for the measurement DataLoader
    gp_cov_scale: float = 1.0  # GP prior covariance = gp_cov_scale * I_{n_concepts}
    meas_jitter: float = 1e-6  # Diagonal jitter added to K_{X_M} before Cholesky solve
    n_concepts: Optional[int] = None  # Output dim of source_fn; auto-detected if None
    # general params
    seed: int = 0
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path

    def __post_init__(self):
        self.name = (
            f"{self.name}-{self.label_flip}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        )
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

    # ------------------------------------------------------------------ #
    # Load preference training data (full dataset, no split)
    # ------------------------------------------------------------------ #
    dataset_path = osp.expanduser(config.dataset)
    dataset = util.Pref_H5Dataset(dataset_path, -1, config.label_flip)
    state_shape, action_shape = dataset.shapes()
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
    training_data_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    # ------------------------------------------------------------------ #
    # Load measurement dataset and build its DataLoader
    # ------------------------------------------------------------------ #
    print(f"[fMAP] Loading measurement dataset: {config.measurement_dataset}")
    x_meas, aux_meas = load_measurement_data(config.measurement_dataset)
    print(
        f"[fMAP] Measurement pool: {x_meas.shape[0]} observations "
        f"(obs_dim={x_meas.shape[1]}, "
        f"aux_dim={'none' if aux_meas is None else aux_meas.shape[1]})"
    )

    meas_tensors = [torch.from_numpy(x_meas)]
    if aux_meas is not None:
        meas_tensors.append(torch.from_numpy(aux_meas))
    meas_dataset = TensorDataset(*meas_tensors)
    meas_loader_kwargs = dict(
        batch_size=config.meas_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )
    if config.num_workers > 0:
        meas_loader_kwargs["prefetch_factor"] = config.prefetch_factor
    meas_loader = DataLoader(meas_dataset, **meas_loader_kwargs)
    # Infinite iterator that reshuffles each time the pool is exhausted.
    meas_iter = util.inf_loop(meas_loader)

    # ------------------------------------------------------------------ #
    # Build functional GP prior (LCFModel)
    # ------------------------------------------------------------------ #
    source_fn = getattr(reward_functions, config.source_fn)

    if config.n_concepts is None:
        # Auto-detect output dimension via a single dummy forward pass.
        _dummy_X = torch.zeros(1, x_meas.shape[1], dtype=torch.float64).to(device)
        _dummy_aux = (
            torch.zeros(1, aux_meas.shape[1], dtype=torch.float64).to(device)
            if aux_meas is not None
            else None
        )
        with torch.no_grad():
            if _dummy_aux is not None:
                _out = source_fn(_dummy_X, _dummy_aux, device)
            else:
                _out = source_fn(_dummy_X, device)
        n_concepts = _out.shape[1]
        print(f"[fMAP] Auto-detected n_concepts = {n_concepts} from '{config.source_fn}'")
    else:
        n_concepts = config.n_concepts

    p_covariance = np.eye(n_concepts, dtype=np.float32) * config.gp_cov_scale
    p_mean = np.zeros(n_concepts, dtype=np.float32)
    gp_prior = LCFModel(
        p_covariance=p_covariance,
        function_vect=source_fn,
        device=device,
        p_mean=p_mean,
    ).to(device)

    # ------------------------------------------------------------------ #
    # Build MLP and functional MAP trainer
    # ------------------------------------------------------------------ #
    net = MLP(
        state_dim + action_dim, 1, [config.width] * config.depth, config.activations
    ).to(device)
    if config.compile_model:
        net = torch.compile(net)
    net_optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    model = MRTrainerF(
        net,
        opt=net_optimizer,
        num_datapoints=len(dataset),
        prior=gp_prior,
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
            meas_batch = next(meas_iter)
            x_meas_b = meas_batch[0]
            aux_meas_b = meas_batch[1] if len(meas_batch) > 1 else None
            train_batch = [b.to(device, non_blocking=True) for b in train_batch]
            for key, val in model.train(
                train_batch, x_meas_b, aux_meas_b, config.meas_jitter
            ).items():
                metrics[key].append(val)

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
