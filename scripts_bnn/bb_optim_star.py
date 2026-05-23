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
from optbnn.bnn.priors import OptimGaussianPrior
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
    project: str = "BB-training"
    group: str = "BB"
    name: str = "bb"
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
    # Cyclical step-size schedule (Zhang et al. 2020)
    # Each cycle alternates between a hot phase (lr_max, exploration) and a
    # cool phase (sghmc_lr, sampling).  One sample is collected at the end of
    # each cool phase.  Set use_cyclical_lr=False to revert to fixed-lr mode.
    use_cyclical_lr: bool = True
    sghmc_lr_max: float = 0.03      # hot-phase lr; ~10× sghmc_lr
    cycle_length: int = 1000        # total steps per cycle (hot + cool)
    fraction_cool: float = 0.25     # fraction of cycle spent in cool/sampling phase
    dataset: str = "data/bb/t0012_pref.hdf5"
    dataset_id: str = "bb_t0012"
    training_split: float = 0.8
    # general params
    seed: int = 1
    OUT_DIR: Optional[str] = "./exp/reward_learning/bb_optim_star"  # Save path
    prior_dir: str = "./exp/reward_learning/bb_tuning_star"

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.OUT_DIR is not None:
            self.OUT_DIR = os.path.join(osp.expanduser(self.OUT_DIR), self.name)
            util.ensure_dir(self.OUT_DIR)
        self.prior_dir = os.path.join(
            osp.expanduser(self.prior_dir),
            f"bb-{self.width}_{self.depth}",
        )


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=f"{config.name}_optim_star_training",
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
    # Resolve to an absolute path so the worker processes (spawned with a fresh
    # interpreter that may have a different CWD) can always locate the file.
    ckpt_path = os.path.abspath(os.path.join(config.prior_dir, "ckpts", "best.ckpt"))
    prior = OptimGaussianPrior(ckpt_path)

    # net_args is kept as a dict so it can be forwarded to _pref_chain_worker
    # (which must reconstruct the net inside each spawned process).
    net_args = dict(
        input_dim=24,
        output_dim=1,
        hidden_dims=[width] * depth,
        activation_fn=transfer_fn,
    )
    net = MLP(**net_args)
    likelihood = LikCE()

    # Initialize the sampler
    # Absolute path for the same reason as ckpt_path above.
    saved_dir = os.path.abspath(os.path.join(config.OUT_DIR, "sampling_std"))
    util.ensure_dir(saved_dir)
    # n_gpu=1: this instance is used only for orchestration and post-hoc
    # evaluation; the actual training runs in per-chain worker processes,
    # each owning one GPU.
    bayes_net_std = PrefNet(
        net, likelihood, prior, saved_dir, n_gpu=1, name="optim_star"
    )

    # Compute a shared starting point for all chains.
    #
    # Without this, each worker creates a fresh MLP with a different random
    # seed, which drives it into a different basin during burn-in — the primary
    # cause of high prediction R-hat.  Running a warm-up burn-in here (in the
    # parent process, on one GPU) moves the network from its random
    # initialization into a single low-loss region.  All chains then start
    # from that point and diverge only due to their per-chain SGHMC noise.
    util.set_seed(config.seed)
    bayes_net_std.train(
        X_train,
        y_train,
        num_samples=None,               # burn-in only; no weights collected
        num_burn_in_steps=config.num_burn_in_steps,
        lr=config.sghmc_lr,
        mdecay=config.mdecay,
        batch_size=config.batch_size,
    )
    # network_weights returns a tuple of CPU numpy arrays — picklable and safe
    # to pass across the mp.spawn process boundary.
    initial_weights = bayes_net_std.network_weights

    # Run chains in parallel — one process per GPU, batched if num_chains > GPUs.
    bayes_net_std.sample_multi_chains_parallel(
        X_train,
        y_train,
        net_args=net_args,
        ckpt_path=ckpt_path,
        num_chains=config.num_chains,
        seed=config.seed,
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        n_discarded=config.n_discarded,
        num_burn_in_steps=config.num_burn_in_steps,
        keep_every=config.keep_every,
        lr=config.sghmc_lr,
        mdecay=config.mdecay,
        print_every_n_samples=config.print_every_n_samples,
        initial_weights=initial_weights,
        use_cyclical_lr=config.use_cyclical_lr,
        lr_max=config.sghmc_lr_max,
        cycle_length=config.cycle_length,
        fraction_cool=config.fraction_cool,
    )
    # Fixed observation set used for prediction-based R-hat.
    # We pull raw observations out of the first arm of up to 64 test pairs.
    # Shape after reshape: (min(64, N_test) * T, obs_dim).
    _B_rhat = min(64, X_test.shape[0])
    _obs_dim = X_test.shape[-1] - 1  # last column is the attention mask
    x_rhat = (
        X_test[:_B_rhat, 0, :, :_obs_dim]
        .reshape(-1, _obs_dim)
        .astype(np.float32)
    )
    x_rhat_t = torch.from_numpy(x_rhat).to(bayes_net_std.device)

    mean_ce = []
    mean_acc = []
    pred_chains = []
    params_chains = []
    for i in range(config.num_chains):
        # Each chain wrote to its own subdirectory as chain_<i>.
        chain_dir = os.path.join(saved_dir, f"chain_{i}")
        bayes_net_std.sampled_weights = bayes_net_std._load_sampled_weights(
            os.path.join(chain_dir, "sampled_weights", "sampled_weights_0000000")
        )
        ce, acc = bayes_net_std.eval_test_data(X_test, y_test, X_train, y_train)
        mean_ce.append(ce)
        mean_acc.append(acc)

        # Per-sample predicted rewards on the fixed observation set.
        # Shape: (num_samples, n_obs) where n_obs = _B_rhat * T.
        bayes_net_std.net.eval()
        with torch.no_grad():
            chain_preds = []
            for weights in bayes_net_std.sampled_weights:
                bayes_net_std.network_weights = weights
                pred = bayes_net_std.net(x_rhat_t).detach().cpu().numpy().ravel()
                chain_preds.append(pred)
        pred_chains.append(np.stack(chain_preds))

        params_chains.append(
            np.stack(
                [
                    np.hstack([arr.ravel() for arr in arrays])
                    for arrays in bayes_net_std.sampled_weights
                ]
            )
        )

    # pred_chains: (num_chains, num_samples, n_obs)
    # params_chains: (num_chains, num_samples, n_params)
    pred_chains = np.stack(pred_chains)
    params_chains = np.stack(params_chains)

    # Prediction R-hat is the primary convergence diagnostic: it is immune to
    # weight-space symmetries (permutation of hidden units, sign flips) that
    # inflate parameter R-hat even when all chains sample the same function.
    rhats_pred = azs.rhat(pred_chains)
    # Parameter R-hat is retained for reference.
    rhats_param = azs.rhat(params_chains)

    summary = {
        "test_mean_cross_entropy": np.mean(mean_ce),
        "test_mean_accuracy": np.mean(mean_acc),
        # --- prediction R-hat (primary diagnostic) ---
        "pred_rhat_max": float(np.max(rhats_pred)),
        "pred_rhat_95th_pct": float(np.percentile(rhats_pred, 95)),
        "pred_rhat_median": float(np.median(rhats_pred)),
        "pred_rhat_mean": float(np.mean(rhats_pred)),
        "pred_rhat_pct_over_1.01": float(np.mean(rhats_pred > 1.01) * 100),
        # --- parameter R-hat (reference; inflated by weight symmetries) ---
        "param_rhat_max": float(np.max(rhats_param)),
        "param_rhat_95th_pct": float(np.percentile(rhats_param, 95)),
        "param_rhat_median": float(np.median(rhats_param)),
        "param_rhat_mean": float(np.mean(rhats_param)),
        "param_rhat_pct_over_1.01": float(np.mean(rhats_param > 1.01) * 100),
    }
    wandb.log(summary)


# In[ ]:
if __name__ == "__main__":
    train()