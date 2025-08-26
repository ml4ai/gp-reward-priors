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


from optbnn.bnn.likelihoods import LikCE, LikGaussian
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.gp.models.model import LCFModel
from optbnn.gp.reward_functions import pen_task_reward_prior
from optbnn.metrics.sampling import compute_rhat_regression
from optbnn.prior_mappers.wasserstein_mapper import (
    MapperWasserstein,
    WassersteinDistance,
)
from optbnn.sgmcmc_bayes_net.pref_net import PrefNet
from optbnn.utils import util
from optbnn.utils.rand_generators import DataSetSampler


# In[5]:


mpl.rcParams["figure.dpi"] = 100


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


@dataclass
class TrainConfig:
    # wandb params
    project: str = "BR-training"
    group: str = "BR"
    name: str = "br"
    # model params
    width: int = 64
    depth: int = 3
    # prior tuning params
    tuning_set: str = "data/adroit_pen/adroit_pen_tuning_set.hdf5"
    mapper_num_iters: int = 1000
    n_data: int = 512
    wasserstein_steps_0: int = 0
    wasserstein_steps_1: int = 1200
    n_samples: int = 1024
    wasserstein_lr: float = 0.001
    lr: float = 0.08
    save_ckpt_every: int = 50
    print_every: int = 20
    # SGHMC Hyper-parameters
    batch_size: int = 256
    num_samples: int = 40
    n_discarded: int = 10
    num_burn_in_steps: int = 2000
    keep_every: int = 2000
    sghmc_lr: float = 0.01
    num_chains: int = 4
    mdecay: float = 0.01
    print_every_n_samples: int = 5
    dataset: str = "data/adroit_pen/AdroitHandPen-v1_pref_b.hdf5"
    dataset_id: str = "pen"
    training_split: float = 0.8
    # general params
    run_prior_tuning: bool = True
    run_training: bool = True
    seed: int = 1
    OUT_DIR: Optional[str] = "./exp/reward_learning/pen"  # Save path
    preload_prior_dir: Optional[str] = None

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.OUT_DIR is not None:
            self.OUT_DIR = os.path.join(osp.expanduser(self.OUT_DIR), self.name)
            self.FIG_DIR = os.path.join(self.OUT_DIR, "figures")
            util.ensure_dir(self.OUT_DIR)
            util.ensure_dir(self.FIG_DIR)
        if self.preload_prior_dir is not None:
            self.preload_prior_dir = osp.expanduser(self.preload_prior_dir)


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
    util.set_seed(config.seed)

    # p_mean = np.array([0.0, -1.0, 1.0, 10.0, 50.0, -5.0])
    p_covariance = np.identity(6)
    pen_prior = LCFModel(p_covariance, pen_task_reward_prior, device=device)
    pen_prior = pen_prior.to(device)

    # In[10]:

    util.set_seed(config.seed)
    # Initialize BNN Priors
    width = config.width  # Number of units in each hidden layer
    depth = config.depth  # Number of hidden layers
    transfer_fn = "relu"  # Activation function

    # Initialize Gaussian prior.
    # Fixed Prior
    std_bnn = GaussianMLPReparameterization(
        input_dim=69,
        output_dim=1,
        activation_fn=transfer_fn,
        hidden_dims=[width] * depth,
    )

    # Prior to be optimized
    opt_bnn = GaussianMLPReparameterization(
        input_dim=69,
        output_dim=1,
        activation_fn=transfer_fn,
        hidden_dims=[width] * depth,
    )

    std_bnn = std_bnn.to(device)
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
    if config.run_prior_tuning:
        wandb.init(
            config=asdict(config),
            project=config.project,
            group=config.group,
            name=f"{config.name}_tuning",
            id=str(uuid.uuid4()),
            save_code=True,
        )
        mapper = MapperWasserstein(
            pen_prior,
            opt_bnn,
            data_generator,
            out_dir=config.OUT_DIR,
            input_dim=69,
            wasserstein_steps=(config.wasserstein_steps_0, config.wasserstein_steps_1),
            wasserstein_lr=config.wasserstein_lr,
            n_data=config.n_data,
            n_gpu=1,
            gpu_gp=True,
        )

        # Start optimizing the prior
        w_hist = mapper.optimize(
            num_iters=mapper_num_iters,
            n_samples=config.n_samples,
            lr=config.lr,
            save_ckpt_every=config.save_ckpt_every,
            print_every=config.print_every,
            debug=True,
        )
        path = os.path.join(config.OUT_DIR, "wsr_values.log")
        np.savetxt(path, w_hist, fmt="%.6e")
        wandb.finish()

    # In[ ]:

    # Visualize progression of the prior optimization
    if config.preload_prior_dir is None:
        wdist_file = os.path.join(config.OUT_DIR, "wsr_values.log")
    else:
        wdist_file = os.path.join(config.preload_prior_dir, "wsr_values.log")
    wdist_vals = np.loadtxt(wdist_file)

    fig = plt.figure(figsize=(6, 3.5))
    indices = np.arange(mapper_num_iters)[::5]
    plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
    plt.ylabel(r"$W_1(p_{gp}, p_{nn})$")
    plt.xlabel("Iteration")
    wdist_fig = os.path.join(config.FIG_DIR, "pen_wsr_plot.png")
    plt.savefig(wdist_fig)
    plt.show()

    # In[ ]:

    # Load the optimize prior
    util.set_seed(config.seed)
    if config.preload_prior_dir is None:
        ckpt_path = os.path.join(
            config.OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters)
        )
    else:
        ckpt_path = os.path.join(
            config.preload_prior_dir, "ckpts", "it-{}.ckpt".format(mapper_num_iters)
        )
    opt_bnn.load_state_dict(torch.load(ckpt_path, weights_only=False))

    # In[ ]:

    # Draw functions from the priors
    n_plot = 4000
    util.set_seed(8)
    X, aux_X = data_generator.get(100)
    X = X.to(device)
    aux_X = aux_X.to(device)
    gp_samples = (
        pen_prior.sample_functions(X, n_plot, aux_X).detach().cpu().numpy().squeeze()
    )

    std_bnn_samples = (
        std_bnn.sample_functions(X.float(), n_plot).detach().cpu().numpy().squeeze()
    )

    opt_bnn_samples = (
        opt_bnn.sample_functions(X.float(), n_plot).detach().cpu().numpy().squeeze()
    )

    seq = np.arange(100)

    fig, axs = plt.subplots(1, 3, figsize=(14, 3))
    plot_samples(seq, gp_samples, ax=axs[0], n_keep=5)
    axs[0].set_title("GP Prior")
    axs[0].set_ylim([int(np.min(gp_samples)) - 1, int(np.max(gp_samples)) + 1])

    plot_samples(seq, std_bnn_samples, ax=axs[1], color="xkcd:grass", n_keep=5)
    axs[1].set_title("BNN Prior (Fixed)")
    axs[1].set_ylim(
        [int(np.min(std_bnn_samples)) - 1, int(np.max(std_bnn_samples)) + 1]
    )

    plot_samples(
        seq, opt_bnn_samples, ax=axs[2], color="xkcd:yellowish orange", n_keep=5
    )
    axs[2].set_title("BNN Prior (GP-induced)")
    axs[2].set_ylim(
        [int(np.min(opt_bnn_samples)) - 1, int(np.max(opt_bnn_samples)) + 1]
    )

    plt.tight_layout()
    prior_fig = os.path.join(config.FIG_DIR, "pen_priors_plot.png")
    plt.savefig(prior_fig)
    plt.show()

    # In[ ]:

    # In[17]:
    if config.run_training:
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
        X_test = X_test[:, :, :, :69].reshape(-1, 69)

        # In[19]:

        # Initialize the prior
        util.set_seed(config.seed)
        prior = FixedGaussianPrior(std=1.0)

        # Setup likelihood
        net = MLP(69, 1, [width] * depth, transfer_fn)
        likelihood = LikCE()

        # Initialize the sampler
        saved_dir = os.path.join(config.OUT_DIR, "sampling_std")
        util.ensure_dir(saved_dir)
        bayes_net_std = PrefNet(net, likelihood, prior, saved_dir, n_gpu=1, name="FG")
        wandb.init(
            config=asdict(config),
            project=config.project,
            group=config.group,
            name=f"{config.name}_FG_training",
            id=str(uuid.uuid4()),
            save_code=True,
        )
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
        wandb.finish()
        bnn_std_preds = bnn_std_preds.squeeze().T

        # Save the predictions
        posterior_std_path = os.path.join(config.OUT_DIR, "posterior_std.npz")
        np.savez(posterior_std_path, bnn_samples=bnn_std_preds)

        # In[21]:

        # Load the optimized prior
        if config.preload_prior_dir is None:
            ckpt_path = os.path.join(
                config.OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters)
            )
        else:
            ckpt_path = os.path.join(
                config.preload_prior_dir, "ckpts", "it-{}.ckpt".format(mapper_num_iters)
            )
        prior = OptimGaussianPrior(ckpt_path)

        # Setup likelihood
        net = MLP(69, 1, [width] * depth, transfer_fn)
        likelihood = LikCE()

        # Initialize the sampler
        saved_dir = os.path.join(config.OUT_DIR, "sampling_optim")
        util.ensure_dir(saved_dir)
        bayes_net_optim = PrefNet(
            net, likelihood, prior, saved_dir, n_gpu=1, name="GPi"
        )
        wandb.init(
            config=asdict(config),
            project=config.project,
            group=config.group,
            name=f"{config.name}_GPi_training",
            id=str(uuid.uuid4()),
            save_code=True,
        )
        # Start sampling
        bayes_net_optim.sample_multi_chains(X_train, y_train, **sampling_configs)

        # In[22]:

        # Make predictions
        util.set_seed(config.seed)
        _, _, bnn_optim_preds = bayes_net_optim.predict(X_test, True)

        # Convergence diagnostics using the R-hat statistic
        r_hat = compute_rhat_regression(bnn_optim_preds, sampling_configs["num_chains"])
        wandb.log(
            {"GPi_mean_R_hat": float(r_hat.mean()), "GPi_std_R_hat": float(r_hat.std())}
        )
        print(
            r"R-hat: mean {:.4f} std {:.4f}".format(
                float(r_hat.mean()), float(r_hat.std())
            )
        )
        wandb.finish()
        bnn_optim_preds = bnn_optim_preds.squeeze().T

        # Save the predictions
        posterior_optim_path = os.path.join(config.OUT_DIR, "posterior_optim.npz")
        np.savez(posterior_optim_path, bnn_samples=bnn_optim_preds)

        # In[33]:

        util.set_seed(config.seed + 7)
        p_bnn_std_preds = bnn_std_preds[:100]
        p_bnn_std_preds = posterior_sampler(p_bnn_std_preds, 4000)
        fig, axs = plt.subplots(1, 2, figsize=(14, 3))
        plot_samples(seq, p_bnn_std_preds, ax=axs[0], color="xkcd:grass", n_keep=5)
        axs[0].set_title("BNN Posterior (Fixed)")
        axs[0].set_ylim(
            [int(np.min(p_bnn_std_preds)) - 1, int(np.max(p_bnn_std_preds)) + 1]
        )
        p_bnn_optim_preds = bnn_optim_preds[:100]
        p_bnn_optim_preds = posterior_sampler(p_bnn_optim_preds, 4000)
        plot_samples(
            seq, p_bnn_optim_preds, ax=axs[1], color="xkcd:yellowish orange", n_keep=5
        )
        axs[1].set_title("BNN Posterior (GP-induced)")
        axs[1].set_ylim(
            [int(np.min(p_bnn_optim_preds)) - 1, int(np.max(p_bnn_optim_preds)) + 1]
        )

        plt.tight_layout()
        post_fig = os.path.join(config.FIG_DIR, "bb_gp_posteriors_plot.png")
        plt.savefig(post_fig)
        plt.close(fig)


if __name__ == "__main__":
    train()
