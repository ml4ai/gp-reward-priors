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
import arviz_stats as azs
import h5py
import matplotlib.pylab as plt
import numpy as np
import torch
import wandb

warnings.simplefilter("ignore", UserWarning)


# In[3]:

sys.path.insert(0, os.path.abspath(".."))
os.chdir("..")

# In[4]:


from optbnn.bnn.likelihoods import LikCE
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import FixedGaussianPrior
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
    OUT_DIR: Optional[str] = "./exp/reward_learning/bb_FG"  # Save path

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

    X_train, y_train, X_test, y_test = util.load_pref_data(
        config.dataset, config.training_split
    )

    # In[19]:

    # Initialize the prior
    util.set_seed(config.seed)
    prior = FixedGaussianPrior(std=1.0)

    # Setup likelihood
    net = MLP(24, 1, [width] * depth, transfer_fn)
    likelihood = LikCE()

    # Initialize the sampler
    saved_dir = os.path.join(config.OUT_DIR, "sampling_std")
    util.ensure_dir(saved_dir)
    bayes_net_std = PrefNet(net, likelihood, prior, saved_dir, n_gpu=4, name="FG")
    # Start sampling
    bayes_net_std.sample_multi_chains(X_train, y_train, **sampling_configs)
    mean_ce = []
    mean_acc = []
    params_chains = []
    for i in range(config.num_chains):
        bayes_net_std.sampled_weights = bayes_net_std._load_sampled_weights(
            os.path.join(
                saved_dir, "sampled_weights", "sampled_weights_{0:07d}".format(i)
            )
        )
        ce, acc = bayes_net_std.eval_test_data(X_test, y_test, X_train, y_train)
        mean_ce.append(ce)
        mean_acc.append(acc)

        params_chains.append(
            np.stack(
                [
                    np.hstack([arr.ravel() for arr in arrays])
                    for arrays in bayes_net_std.sampled_weights
                ]
            )
        )
    params_chains = np.stack(params_chains)
    rhats = azs.rhat(params_chains)
    summary = {
        "test_mean_cross_entropy": np.mean(mean_ce),
        "test_mean_accuracy": np.mean(mean_acc),
        "max": np.max(rhats),
        "95th_pct": np.percentile(rhats, 95),
        "median": np.median(rhats),
        "mean": np.mean(rhats),
        "pct_over_1.01": np.mean(rhats > 1.01) * 100,
    }
    wandb.log(summary)


# In[ ]:
if __name__ == "__main__":
    train()