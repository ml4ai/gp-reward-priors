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
from optbnn.bnn.priors import OptimGaussianPrior
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
    project: str = "BR-training-optim"
    group: str = "BR"
    name: str = "br_optim"
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
    dataset: str = "data/antmaze/antmaze-medium-play-v2_pref.hdf5"
    dataset_id: str = "antmaze_medium_play"
    training_reduce: float = (
        0.0  # artificially reduce training data by the percentage given (default is no reduction)
    )
    # general params
    seed: int = 1
    OUT_DIR: str = "./exp/reward_learning/antmaze_medium_play_optim"  # Save path
    prior_dir: str = "./exp/reward_learning/antmaze_tuning"
    prior_ckpt: int = 1000

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}-{self.training_reduce}"
        self.OUT_DIR = os.path.join(osp.expanduser(self.OUT_DIR), self.name)
        util.ensure_dir(self.OUT_DIR)

        self.prior_dir = os.path.join(
            osp.expanduser(self.prior_dir),
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
        "eval_map": True,
    }

    # In[18]:

    if config.training_reduce:
        X_train, y_train, _, _ = util.load_pref_data(
            config.dataset, 1.0 - config.training_reduce
        )
    else:
        X_train, y_train = util.load_pref_data(
            config.dataset, 1.0 - config.training_reduce
        )

    # Load the optimized prior
    ckpt_path = os.path.join(
        config.prior_dir, "ckpts", "it-{}.ckpt".format(config.prior_ckpt)
    )
    prior = OptimGaussianPrior(ckpt_path)

    # Setup likelihood
    net = MLP(37, 1, [width] * depth, transfer_fn)
    likelihood = LikCE()

    # Initialize the sampler
    saved_dir = os.path.join(config.OUT_DIR, "sampling_optim")
    util.ensure_dir(saved_dir)
    bayes_net_std = PrefNet(net, likelihood, prior, saved_dir, n_gpu=1, name="optim")
    # Start sampling
    bayes_net_std.sample_multi_chains(X_train, y_train, **sampling_configs)

# In[ ]:
if __name__ == "__main__":
    train()
