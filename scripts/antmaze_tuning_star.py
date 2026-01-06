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

from optbnn.bnn.priors import OptimGaussianPrior
from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.gp.models.model import LCFModel
from optbnn.gp.reward_functions import antmaze_task_reward_prior
from optbnn.prior_mappers.wasserstein_gp_mapper import MapperWassersteinGP
from optbnn.utils import util
from optbnn.utils.rand_generators import DataSetSampler

# In[5]:

mpl.rcParams["figure.dpi"] = 100


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    # wandb params
    project: str = "BR-tuning-star"
    group: str = "BR"
    name: str = "br"
    # model params
    width: int = 64
    depth: int = 3
    # prior tuning params
    tuning_set: str = "data/antmaze/antmaze-medium-play-v2_tuning_set.hdf5"
    dataset_id: str = "antmaze_medium_play"
    mapper_num_iters: int = 1000
    n_data: int = 512
    batches: int = 10
    lr: float = 0.08
    save_ckpt_every: int = 50
    print_every: int = 20
    seed: int = 1
    OUT_DIR: str = "./exp/reward_learning/antmaze_tuning_star"  # Save path

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}-{self.width}_{self.depth}"
        self.OUT_DIR = os.path.join(osp.expanduser(self.OUT_DIR), self.name)
        self.FIG_DIR = os.path.join(self.OUT_DIR, "figures")
        util.ensure_dir(self.OUT_DIR)
        util.ensure_dir(self.FIG_DIR)


# In[8]:


def plot_samples(
    X, samples, var=None, n_keep=12, color="xkcd:bluish", smooth_q=False, ax=None
):
    if ax is None:
        ax = plt.gca()
    if samples.ndim > 2:
        samples = samples.squeeze()
    n_keep = int(samples.shape[1] / 10) if n_keep is None else n_keep
    keep_idx = np.random.permutation(samples.shape[1])[:n_keep]
    mu = samples.mean(1)
    if var is None:
        q = 97.5  ## corresponds to 2 stdevs in Gaussian
        # q = 99.99  ## corresponds to 3 std
        Q = np.percentile(samples, [100 - q, q], axis=1)
        ub, lb = Q[1, :], Q[0, :]
        # ub, lb = mu + 2 * samples.std(1), mu - 2 * samples.std(1)
        if smooth_q:
            lb = moving_average(lb)
            ub = moving_average(ub)
    else:
        ub = mu + 3 * np.sqrt(var)
        lb = mu - 3 * np.sqrt(var)
    ####
    ax.fill_between(X.flatten(), ub, lb, color=color, alpha=0.25, lw=0)
    ax.plot(X, samples[:, keep_idx], color=color, alpha=0.8)
    ax.plot(X, mu, color="xkcd:red")


def posterior_sampler(preds, n_samps):
    samples = []
    for row in preds:
        samples.append(np.random.choice(row, n_samps))
    return np.stack(samples)


# In[9]:


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=f"{config.name}_tuning",
        id=str(uuid.uuid4()),
        save_code=True,
    )
    util.set_seed(config.seed)

    # p_mean = np.array([0.0, -1.0, 1.0, 10.0, 50.0, -5.0])
    p_covariance = np.identity(3)
    antmaze_prior = LCFModel(p_covariance, antmaze_task_reward_prior, device=device)
    antmaze_prior = antmaze_prior.to(device)

    # In[10]:

    util.set_seed(config.seed)
    # Initialize BNN Priors
    width = config.width  # Number of units in each hidden layer
    depth = config.depth  # Number of hidden layers
    transfer_fn = "relu"  # Activation function

    # Prior to be optimized
    opt_bnn = GaussianMLPReparameterization(
        input_dim=37,
        output_dim=1,
        activation_fn=transfer_fn,
        hidden_dims=[width] * depth,
    )

    opt_bnn = opt_bnn.to(device)

    # In[11]:

    util.set_seed(config.seed)
    with h5py.File(config.tuning_set) as f:
        data_generator = DataSetSampler(f["obs"][:], f["aux_obs"][:])

    # In[12]:

    mapper_num_iters = config.mapper_num_iters

    # In[13]:

    # Initiialize the Wasserstein optimizer
    util.set_seed(config.seed)
    mapper = MapperWassersteinGP(
        antmaze_prior,
        opt_bnn,
        data_generator,
        out_dir=config.OUT_DIR,
        input_dim=37,
        n_data=config.n_data,
        n_gpu=1,
        gpu_gp=True,
    )

    # Start optimizing the prior
    w_hist = mapper.optimize(
        num_iters=mapper_num_iters,
        batches=config.batches,
        lr=config.lr,
        save_ckpt_every=config.save_ckpt_every,
        print_every=config.print_every,
    )
    path = os.path.join(config.OUT_DIR, "wsr_values.log")
    np.savetxt(path, w_hist, fmt="%.6e")
    wandb.finish()

    # In[14]:

    # Visualize progression of the prior optimization
    wdist_file = os.path.join(config.OUT_DIR, "wsr_values.log")
    wdist_vals = np.loadtxt(wdist_file)

    fig = plt.figure(figsize=(6, 3.5))
    indices = np.arange(mapper_num_iters)[::5]
    plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
    plt.ylabel(r"$W_2(p_{r}, p_{nn})$")
    plt.xlabel("Iteration")
    wdist_fig = os.path.join(config.FIG_DIR, "antmaze_wsr_plot.png")
    plt.savefig(wdist_fig)
    plt.close(fig)

    # # In[15]:

    # # Load the optimize prior
    # util.set_seed(config.seed)
    # ckpt_path = os.path.join(
    #     config.OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters)
    # )
    # opt_bnn.load_state_dict(torch.load(ckpt_path, weights_only=False))

    # # In[16]:

    # # Draw functions from the priors
    # n_plot = 4000
    # util.set_seed(config.seed + 7)
    # X, aux_X = data_generator.get(100)
    # X = X.to(device)
    # aux_X = aux_X.to(device)
    # gp_samples = (
    #     antmaze_prior.sample_functions(X, n_plot, aux_X)
    #     .detach()
    #     .cpu()
    #     .numpy()
    #     .squeeze()
    # )

    # nngp_samples = (
    #     opt_bnn.sample_nngp(X, n_plot, device).detach().cpu().numpy().squeeze()
    # )

    # opt_bnn_samples = (
    #     opt_bnn.sample_functions(X.float(), n_plot).detach().cpu().numpy().squeeze()
    # )

    # seq = np.arange(100)

    # fig, axs = plt.subplots(1, 3, figsize=(14, 3))
    # plot_samples(seq, gp_samples, ax=axs[0], n_keep=5)
    # axs[0].set_title("GP Prior")
    # axs[0].set_ylim([int(np.min(gp_samples)) - 1, int(np.max(gp_samples)) + 1])

    # plot_samples(seq, nngp_samples, ax=axs[1], color="xkcd:grass", n_keep=5)
    # axs[1].set_title("NNGP Prior")
    # axs[1].set_ylim([int(np.min(nngp_samples)) - 1, int(np.max(nngp_samples)) + 1])

    # plot_samples(
    #     seq, opt_bnn_samples, ax=axs[2], color="xkcd:yellowish orange", n_keep=5
    # )
    # axs[2].set_title("BNN Prior (NNGP-induced)")
    # axs[2].set_ylim(
    #     [int(np.min(opt_bnn_samples)) - 1, int(np.max(opt_bnn_samples)) + 1]
    # )

    # plt.tight_layout()
    # prior_fig = os.path.join(config.FIG_DIR, "antmaze_priors_plot.png")
    # plt.savefig(prior_fig)
    # plt.close(fig)


# In[ ]:
if __name__ == "__main__":
    train()
