#!/usr/bin/env python
# coding: utf-8


import math
import os
import os.path as osp
import sys
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import pyrallis

mpl.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import torch
import wandb

import h5py

warnings.simplefilter("ignore", UserWarning)


# In[3]:

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

# In[4]:


from optbnn.bnn.likelihoods import LikCE
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import FixedGaussianPrior
from optbnn.metrics.sampling import compute_rhat_regression
from optbnn.sgmcmc_bayes_net.pref_net import PrefNet
from optbnn.utils import util
from optbnn.utils.rand_generators import DataSetSampler

# In[5]:

mpl.rcParams["figure.dpi"] = 100


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    # wandb params
    project: str = "BR-training"
    group: str = "BR"
    name: str = "br"
    # model params
    width: int = 64
    depth: int = 3
    # SGHMC Hyper-parameters
    batch_size: int = 256
    num_samples: int = 50
    n_discarded: int = 10
    num_burn_in_steps: int = 3000
    keep_every: int = 2000
    sghmc_lr: float = 0.008
    num_chains: int = 4
    mdecay: float = 0.01
    print_every_n_samples: int = 5
    dataset: str = "data/bb/t0012_pref.hdf5"
    dataset_id: str = "bb_t0012"
    training_split: float = 0.8
    # general params
    seed: int = 1
    OUT_DIR: Optional[str] = "./exp/reward_learning_gp/bb_FG"  # Save path

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.OUT_DIR is not None:
            self.OUT_DIR = os.path.join(osp.expanduser(self.OUT_DIR), self.name)
            util.ensure_dir(self.OUT_DIR)


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=f"{config.name}_FG_training",
        id=str(uuid.uuid4()),
        save_code=True,
    )
    util.set_seed(config.seed)
    # Initialize BNN Priors
    width = config.width  # Number of units in each hidden layer
    depth = config.depth  # Number of hidden layers
    transfer_fn = "relu"  # Activation function

    # SGHMC Hyper-parameters
    sampling_configs = {
        "batch_size": config.batch_size,  # Mini-batch size
        "num_samples": config.num_samples,  # Total number of samples for each chain
        "n_discarded": config.n_discarded,  # Number of the first samples to be discared for each chain
        "num_burn_in_steps": config.num_burn_in_steps,  # Number of burn-in steps
        "keep_every": config.keep_every,  # Thinning interval
        "lr": config.sghmc_lr,  # Step size
        "num_chains": config.num_chains,  # Number of chains
        "mdecay": config.mdecay,  # Momentum coefficient
        "print_every_n_samples": config.print_every_n_samples,
    }

    # In[18]:

    X_train, y_train, X_test, _ = util.load_pref_data(
        config.dataset, config.training_split
    )
    X_test = X_test[:, :, :, :28].reshape(-1, 28)

    # In[19]:

    # Initialize the prior
    util.set_seed(config.seed)
    prior = FixedGaussianPrior(std=1.0)

    # Setup likelihood
    net = MLP(28, 1, [width] * depth, transfer_fn)
    likelihood = LikCE()

    # Initialize the sampler
    saved_dir = os.path.join(config.OUT_DIR, "sampling_std")
    util.ensure_dir(saved_dir)
    bayes_net_std = PrefNet(net, likelihood, prior, saved_dir, n_gpu=1, name="FG")
    # Start sampling
    bayes_net_std.sample_multi_chains(X_train, y_train, **sampling_configs)

    # In[20]:

    # Make predictions
    util.set_seed(config.seed)
    _, _, bnn_std_preds = bayes_net_std.predict(X_test, True)
    # Convergence diagnostics using the R-hat statistic
    r_hat = compute_rhat_regression(bnn_std_preds, sampling_configs["num_chains"])
    wandb.log(
        {"FG_mean_R_hat": float(r_hat.mean()), "FG_std_R_hat": float(r_hat.std())}
    )
    print(
        r"R-hat: mean {:.4f} std {:.4f}".format(
            float(r_hat.mean()), float(r_hat.std())
        )
    )
    bnn_std_preds = bnn_std_preds.squeeze().T

    # Save the predictions
    posterior_std_path = os.path.join(config.OUT_DIR, "posterior_std.npz")
    np.savez(posterior_std_path, bnn_samples=bnn_std_preds)

# In[ ]:
if __name__ == "__main__":
    train()
