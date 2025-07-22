#!/usr/bin/env python
# coding: utf-8


import math
import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import torch

import h5py

warnings.simplefilter("ignore", UserWarning)


# In[3]:


os.chdir("..")


# In[4]:


from optbnn.bnn.likelihoods import LikCE
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.gp.models.model import LCFModel
from optbnn.gp.reward_functions import bb_reward_prior
from optbnn.metrics.sampling import compute_rhat_regression
from optbnn.prior_mappers.wasserstein_gp_mapper import MapperWassersteinGP
from optbnn.sgmcmc_bayes_net.pref_net import PrefNet
from optbnn.utils import util
from optbnn.utils.rand_generators import DataSetSampler


# In[5]:


mpl.rcParams["figure.dpi"] = 100


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


OUT_DIR = "./exp/reward_learning_gp/bb"
FIG_DIR = os.path.join(OUT_DIR, "figures")
util.ensure_dir(OUT_DIR)
util.ensure_dir(FIG_DIR)


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


# In[9]:


util.set_seed(1)

# p_mean = np.array([0.0, -1.0, 1.0, 10.0, 50.0, -5.0])
p_covariance = np.identity(3)
bb_prior = LCFModel(p_covariance, bb_reward_prior)
bb_prior = bb_prior.to(device)


# In[10]:


util.set_seed(1)
# Initialize BNN Priors
width = 64  # Number of units in each hidden layer
depth = 3  # Number of hidden layers
transfer_fn = "relu"  # Activation function

# Prior to be optimized
opt_bnn = GaussianMLPReparameterization(
    input_dim=28, output_dim=1, activation_fn=transfer_fn, hidden_dims=[width] * depth
)

opt_bnn = opt_bnn.to(device)


# In[11]:


util.set_seed(1)
with h5py.File("data/bb/bb_tuning_set.hdf5") as f:
    data_generator = DataSetSampler(f["obs"][:], f["aux_obs"][:])


# In[12]:


mapper_num_iters = 1000


# In[13]:


# Initiialize the Wasserstein optimizer
util.set_seed(1)
mapper = MapperWassersteinGP(
    bb_prior,
    opt_bnn,
    data_generator,
    out_dir=OUT_DIR,
    input_dim=28,
    n_data=512,
    n_gpu=1,
    gpu_gp=True,
)

# Start optimizing the prior
w_hist = mapper.optimize(
    num_iters=mapper_num_iters,
    batches=10,
    lr=0.08,
    save_ckpt_every=50,
    print_every=20,
)
path = os.path.join(OUT_DIR, "wsr_values.log")
np.savetxt(path, w_hist, fmt="%.6e")


# In[14]:


# Visualize progression of the prior optimization
wdist_file = os.path.join(OUT_DIR, "wsr_values.log")
wdist_vals = np.loadtxt(wdist_file)

fig = plt.figure(figsize=(6, 3.5))
indices = np.arange(mapper_num_iters)[::5]
plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
plt.ylabel(r"$W_1(p_{gp}, p_{nn})$")
plt.xlabel("Iteration")
wdist_fig = os.path.join(FIG_DIR, "bb_gp_wsr_plot.png")
plt.savefig(wdist_fig)
plt.show()


# In[15]:


# Load the optimize prior
util.set_seed(1)
ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters))
opt_bnn.load_state_dict(torch.load(ckpt_path))


# In[16]:


# Draw functions from the priors
n_plot = 4000
util.set_seed(8)
X, aux_X = data_generator.get(100)

gp_samples = (
    bb_prior.sample_functions(X, n_plot, aux_X).detach().cpu().numpy().squeeze()
)

nngp_samples = opt_bnn.sample_nngp(X, n_plot).detach().cpu().numpy().squeeze()

opt_bnn_samples = (
    opt_bnn.sample_functions(X.float(), n_plot).detach().cpu().numpy().squeeze()
)

seq = np.arange(100)

fig, axs = plt.subplots(1, 3, figsize=(14, 3))
plot_samples(seq, gp_samples, ax=axs[0], n_keep=5)
axs[0].set_title("GP Prior")
axs[0].set_ylim([-55, 55])

plot_samples(seq, nngp_samples, ax=axs[1], color="xkcd:grass", n_keep=5)
axs[1].set_title("NNGP Prior")
axs[1].set_ylim([-55, 55])

plot_samples(seq, opt_bnn_samples, ax=axs[2], color="xkcd:yellowish orange", n_keep=5)
axs[2].set_title("BNN Prior (NNGP-induced)")
axs[2].set_ylim([-55, 55])

plt.tight_layout()
prior_fig = os.path.join(FIG_DIR, "bb_gp_priors_plot.png")
plt.savefig(prior_fig)
plt.show()


# In[17]:


# SGHMC Hyper-parameters
sampling_configs = {
    "batch_size": 256,  # Mini-batch size
    "num_samples": 40,  # Total number of samples for each chain
    "n_discarded": 10,  # Number of the first samples to be discared for each chain
    "num_burn_in_steps": 2000,  # Number of burn-in steps
    "keep_every": 2000,  # Thinning interval
    "lr": 0.008,  # Step size
    "num_chains": 4,  # Number of chains
    "mdecay": 0.01,  # Momentum coefficient
    "print_every_n_samples": 5,
}


# In[18]:


X_train, y_train, X_test, _ = util.load_pref_data("data/bb/t0012_pref.hdf5", 0.8)
X_test = X_test[:, :, :, :28].reshape(-1, 28)


# In[19]:


# Initialize the prior
util.set_seed(1)
prior = FixedGaussianPrior(std=1.0)

# Setup likelihood
net = MLP(28, 1, [width] * depth, transfer_fn)
likelihood = LikCE()

# Initialize the sampler
saved_dir = os.path.join(OUT_DIR, "sampling_std")
util.ensure_dir(saved_dir)
bayes_net_std = PrefNet(net, likelihood, prior, saved_dir, n_gpu=0)

# Start sampling
bayes_net_std.sample_multi_chains(X_train, y_train, **sampling_configs)


# In[20]:


# Make predictions
util.set_seed(1)
_, _, bnn_std_preds = bayes_net_std.predict(X_test, True)
# Convergence diagnostics using the R-hat statistic
r_hat = compute_rhat_regression(bnn_std_preds, sampling_configs["num_chains"])
print(r"R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))
bnn_std_preds = bnn_std_preds.squeeze().T

# Save the predictions
posterior_std_path = os.path.join(OUT_DIR, "posterior_std.npz")
np.savez(posterior_std_path, bnn_samples=bnn_std_preds)


# In[21]:


# Load the optimized prior
ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters))
prior = OptimGaussianPrior(ckpt_path)

# Setup likelihood
net = MLP(28, 1, [width] * depth, transfer_fn)
likelihood = LikCE()

# Initialize the sampler
saved_dir = os.path.join(OUT_DIR, "sampling_optim")
util.ensure_dir(saved_dir)
bayes_net_optim = PrefNet(net, likelihood, prior, saved_dir, n_gpu=0)

# Start sampling
bayes_net_optim.sample_multi_chains(X_train, y_train, **sampling_configs)


# In[22]:


# Make predictions
util.set_seed(1)
_, _, bnn_optim_preds = bayes_net_optim.predict(X_test, True)

# Convergence diagnostics using the R-hat statistic
r_hat = compute_rhat_regression(bnn_optim_preds, sampling_configs["num_chains"])
print(r"R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))
bnn_optim_preds = bnn_optim_preds.squeeze().T

# Save the predictions
posterior_optim_path = os.path.join(OUT_DIR, "posterior_optim.npz")
np.savez(posterior_optim_path, bnn_samples=bnn_optim_preds)


# In[27]:


def posterior_sampler(preds, n_samps):
    samples = []
    for row in preds:
        samples.append(np.random.choice(row, n_samps))
    return np.stack(samples)


# In[33]:


util.set_seed(8)
rng = np.random.default_rng(8)
p_bnn_std_preds = bnn_std_preds[50:150]
p_bnn_std_preds = posterior_sampler(p_bnn_std_preds, 4000)
fig, axs = plt.subplots(1, 2, figsize=(14, 3))
plot_samples(seq, p_bnn_std_preds, ax=axs[0], color="xkcd:grass", n_keep=5)
axs[0].set_title("BNN Posterior (Fixed)")
axs[0].set_ylim([-5, 3])
p_bnn_optim_preds = bnn_optim_preds[50:150]
p_bnn_optim_preds = posterior_sampler(p_bnn_optim_preds, 4000)
plot_samples(seq, p_bnn_optim_preds, ax=axs[1], color="xkcd:yellowish orange", n_keep=5)
axs[1].set_title("BNN Posterior (GP-induced)")
axs[1].set_ylim([-5, 2])

plt.tight_layout()
post_fig = os.path.join(FIG_DIR, "bb_gp_posteriors_plot.png")
plt.savefig(post_fig)
plt.show()


# In[ ]:
