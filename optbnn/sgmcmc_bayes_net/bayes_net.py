"""Define a base class of Bayesian Neural Network."""

import glob
import math
import os
from itertools import islice

import numpy as np
import torch
import torch.utils.data as data_utils

from ..samplers.adaptive_sghmc import AdaptiveSGHMC
from ..samplers.sghmc import SGHMC
from ..utils.util import ensure_dir, inf_loop, prepare_device


class BayesNet:
    def __init__(
        self,
        net,
        likelihood,
        prior,
        ckpt_dir,
        temperature=1.0,
        sampling_method="adaptive_sghmc",
        weights_format="state_dict",
        task="regression",
        logger=None,
        n_gpu=0,
    ):
        """
        Bayesian Neural Networks use uses stochastic gradient MCMC methods
            to sample from the posterior distribution.

        Args:
            net: instance of nn.Module, the base neural net.
            likelihood: instance of LikelihoodModule, the module of likelihood.
            temperature: float, temperature in the posterior.
            prior: instance of PriorModule, the module of prior.
            ckpt_dir: str, path to the directory of checkpoints.
            sampling_method: specifies the sampling strategy.
            weights_format: str, the format of sampled weights; possible
                values: state_dict, tuple.
            task: str, the type of task, which is `classification`,`regression`,
                or `pref` (twin network call)
            logger: instance of logging.Logger.
        """
        self.net = net
        self.lik_module = likelihood
        self.prior_module = prior

        self.ckpt_dir = ckpt_dir
        self.sampling_method = sampling_method
        self.weights_format = weights_format
        self.task = task
        self.n_gpu = n_gpu

        self.temperature = temperature

        if logger is None:
            self.print_info = print
        else:
            self.print_info = logger.info

        self.step = 0
        self.sampler = None
        self.sampled_weights = []
        self.num_samples = 0
        self.num_saved_sets_weights = 0
        self.sampled_weights_dir = os.path.join(self.ckpt_dir, "sampled_weights")
        ensure_dir(self.sampled_weights_dir)

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = prepare_device(self.n_gpu)
        self.net = self.net.to(self.device)
        if len(device_ids) > 1:
            self.net = torch.nn.DataParallel(net, device_ids=device_ids)
        self.prior_module = self.prior_module.to(self.device)

        self.map = None

    def reset(self):
        self.step = 0
        self.sampler = None
        self.sampled_weights = []
        self.num_samples = 0
        self.num_saved_sets_weights = 0
        self.net.reset_parameters()
        self.map = None

    @property
    def network_weights(self):
        """Extract current network weight values."""
        if self.weights_format == "tuple":
            return tuple(
                np.asarray(parameter.data.clone().detach().cpu().numpy())
                for parameter in self.net.parameters()
            )

        elif self.weights_format == "state_dict":
            return self.net.state_dict()

    @network_weights.setter
    def network_weights(self, weights):
        """Assign new weights to our neural networks parameters."""
        if self.weights_format == "tuple":
            for parameter, sample in zip(self.net.parameters(), weights):
                parameter.copy_(torch.from_numpy(sample))

        elif self.weights_format == "state_dict":
            self.net.load_state_dict(weights)

    @property
    def _bare_net(self):
        """Return the underlying module, unwrapping DataParallel if present."""
        return (
            self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        )

    def _neg_log_joint(self, fx_batch, y_batch, num_datapoints):
        """Calculate model's negative log joint density.

            Note that the gradient is computed by: g_prior + N/n sum_i grad_theta_xi.
            Because of that we divide here by N=num of datapoints
            since in the sample we will rescale the gradient by N again.

        Args:
            fx_batch: torch tensor, the predictions.
            y_batch: torch tensor, the corresponding targets.
            num_datapoints: int, the number of data points in the entire
                training set.

        Return:
            The negative log joint density.
        """
        return (self.lik_module(fx_batch, y_batch)) / y_batch.shape[
            0
        ] + self.prior_module(self._bare_net) / num_datapoints

    def _initialize_sampler(
        self,
        num_datapoints,
        lr=1e-2,
        mdecay=0.05,
        num_burn_in_steps=3000,
        epsilon=1e-10,
    ):
        """Initialize a stochastic gradient MCMC sampler.

        Args:
            num_datapoints: int, the total number of training data points.
            lr: float, learning rate.
            mdecay: float, momemtum decay.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            epsilon: float, epsilon for numerical stability.
        """
        dtype = np.float32
        self.sampler_params = {}

        # Apply temperature scaling
        self.sampler_params["scale_grad"] = dtype(num_datapoints) / self.temperature
        self.sampler_params["lr"] = dtype(lr)
        self.sampler_params["mdecay"] = dtype(mdecay)

        if self.sampling_method == "adaptive_sghmc":
            self.sampler_params["num_burn_in_steps"] = num_burn_in_steps
            self.sampler_params["epsilon"] = dtype(epsilon)

            self.sampler = AdaptiveSGHMC(self.net.parameters(), **self.sampler_params)
        elif self.sampling_method == "sghmc":
            self.sampler = SGHMC(self.net.parameters(), **self.sampler_params)

    def _save_sampled_weights(self):
        """Save a set of sampled weights to file.

        Args:
            sampled_weights: a state_dict containing the model's parameters.
        """
        file_path = os.path.join(
            self.sampled_weights_dir,
            "sampled_weights_{0:07d}".format(self.num_saved_sets_weights),
        )
        torch.save({"sampled_weights": self.sampled_weights}, file_path)
        self.num_saved_sets_weights += 1

    def save_map(self):
        """Save map estimate to file."""
        file_path = os.path.join(
            self.sampled_weights_dir,
            "map_estimate",
        )
        torch.save({"map_estimate": self.map}, file_path)

    def _load_sampled_weights(self, file_path):
        """Load a set of sampled weights from a given file.

        Args:
            file_path: str, the path to the file containing a set of sampled
                weights.

        Returns:
            sampled_weights: a state_dict containing the model's parameters.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        sampled_weights = checkpoint["sampled_weights"]

        return sampled_weights

    def load_map(self, file_path):
        """Load map estimate from a given file.

        Args:
            file_path: str, the path to the file containing map estimate.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        self.map = checkpoint["map_estimate"]

    def _load_all_sampled_weights(self):
        """Load all the sampled weights from files.

        Returns: a generator for loading sampled weights.
        """

        def load_weights(file_path):
            checkpoint = torch.load(file_path, weights_only=False)
            sampled_weights = checkpoint["sampled_weights"]

            return sampled_weights

        def sampled_weights_loader(sampled_weights_dir):
            file_paths = glob.glob(os.path.join(sampled_weights_dir, "sampled_weights*"))
            for file_path in file_paths:
                for weights in load_weights(file_path):
                    yield weights

                # self.network_weights.clear()
                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()

        return sampled_weights_loader(self.sampled_weights_dir)

    def sample_multi_chains(
        self,
        x_train=None,
        y_train=None,
        data_loader=None,
        num_samples=None,
        num_chains=1,
        keep_every=100,
        n_discarded=0,
        num_burn_in_steps=3000,
        lr=1e-2,
        batch_size=32,
        epsilon=1e-10,
        mdecay=0.05,
        print_every_n_samples=10,
        resample_prior_every=1000,
        eval_map=False,
    ):
        """
        Use multiple chains of sampling.

        Args:
            x_train: numpy array, input training datapoints.
            y_train: numpy array, input training targets.
            data_loader: instance of DataLoader, the dataloader for training
                data. Notice that we have to choose either numpy arrays or
                dataloader for the input data.
            num_samples: int, number of set of parameters per chain
                we want to sample.
            num_chains: int, number of chains.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            n_discarded: int, the number of first samples will
                be discarded.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            lr: float, learning rate.
            batch_size: int, batch size.
            epsilon: float, epsilon for numerical stability.
            mdecay: float, momemtum decay.
            print_every_n_samples: int, defines after how many samples we want
                to print out the statistics of the sampling process.
            continue_training: bool, defines whether we want to continue
                from the last training run.
            resample_prior_every: int, num ber of sampling steps to perform
                before resampling prior.
            eval_map: bool, uses map estimate to evaluate if true, otherwise posterior predictive mean
        """
        # Pre-build a single DataLoader to share across all chains.  Building a
        # TensorDataset + DataLoader inside train() on every chain restart
        # repeats the numpy→tensor conversion and DataLoader initialisation
        # num_chains times for identical data.  train() already has a fast path
        # when data_loader is not None (skips the rebuild), so we just need to
        # construct it once here and pass it through.
        if data_loader is None and x_train is not None:
            if self.task == "regression":
                # Normalization must happen before the DataLoader is built so
                # that the stored mean/std are available for prediction.
                x_sq, y_sq = x_train.squeeze(), y_train.squeeze()
                x_t, y_t = self._normalize_data(x_sq, y_sq)
            elif self.task == "pref":
                x_t = torch.from_numpy(x_train.squeeze()).float()
                y_t = torch.from_numpy(y_train.squeeze()).float()
            else:
                # Generic fallback (e.g. classification with raw numpy arrays).
                x_t = torch.from_numpy(x_train.squeeze()).float()
                y_t = torch.from_numpy(y_train.squeeze()).float()
            shared_loader = data_utils.DataLoader(
                data_utils.TensorDataset(x_t, y_t),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(self.device.type == "cuda"),
                num_workers=0,
            )
        else:
            # Caller already provided a DataLoader, or no array data was given.
            shared_loader = data_loader

        for chain in range(num_chains):
            self.print_info("Chain: {}".format(chain))
            if isinstance(self.net, torch.nn.DataParallel):
                self.net.module.reset_parameters()
            else:
                self.net.reset_parameters()
            self.train(
                x_train,
                y_train,
                shared_loader,  # pre-built; train() skips DataLoader rebuild
                num_samples,
                keep_every,
                n_discarded,
                num_burn_in_steps,
                lr,
                batch_size,
                epsilon,
                mdecay,
                print_every_n_samples,
                continue_training=False,
                clear_sampled_weights=False,
                resample_prior_every=resample_prior_every,
                eval_map=eval_map,
            )
            if self.task == "classification" or self.task == "pref":
                self._save_sampled_weights()
                self.sampled_weights.clear()
                self._save_checkpoint(mode="last")

    def train(
        self,
        x_train=None,
        y_train=None,
        data_loader=None,
        num_samples=None,
        keep_every=100,
        n_discarded=0,
        num_burn_in_steps=3000,
        lr=1e-2,
        batch_size=32,
        epsilon=1e-10,
        mdecay=0.05,
        print_every_n_samples=10,
        continue_training=False,
        clear_sampled_weights=True,
        resample_prior_every=1000,
        resample_hyper_prior_burn_in=True,
        eval_map=False,
        use_cyclical_lr=False,
        lr_max=None,
        cycle_length=None,
        fraction_cool=0.25,
    ):
        """
        Train a BNN using a given dataset.

        Args:
            x_train: numpy array, input training datapoints.
            y_train: numpy array, input training targets.
            data_loader: instance of DataLoader, the dataloader for training
                data. Notice that we have to choose either numpy arrays or
                dataloader for the input data.
            num_samples: int, number of set of parameters we want to sample.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            n_discarded: int, the number of first samples will
                be discarded.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            lr: float, learning rate.
            batch_size: int, batch size.
            epsilon: float, epsilon for numerical stability.
            mdecay: float, momemtum decay.
            print_every_n_samples: int, defines after how many samples we want
                to print out the statistics of the sampling process.
            continue_training: bool, defines whether we want to continue
                from the last training run.
            clear_sampled_weights: bool, indicates whether we want to clear
                the sampled weights in the case of continuing the training.
            resample_prior_every: int, num ber of sampling steps to perform
                before resampling prior.
            eval_map: bool, uses map estimate to evaluate if true, otherwise posterior predictive mean
        """
        # Setup data loader
        if data_loader is not None:
            num_datapoints = len(data_loader.sampler)
            train_loader = inf_loop(data_loader)
        else:
            num_datapoints = x_train.shape[0]
            if self.task == "regression":
                # Normalize the dataset
                x_train, y_train = x_train.squeeze(), y_train.squeeze()
                x_train_, y_train_ = self._normalize_data(x_train, y_train)
            if self.task == "pref":
                x_train_, y_train_ = (
                    torch.from_numpy(x_train.squeeze()).float(),
                    torch.from_numpy(y_train.squeeze()).float(),
                )

            # Initialize a data loader for training data.
            train_loader = inf_loop(
                data_utils.DataLoader(
                    data_utils.TensorDataset(x_train_, y_train_),
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=(self.device.type == "cuda"),
                    num_workers=0,
                )
            )

        # Pre-compute cyclical schedule parameters.  These are always defined so
        # the loop body can reference them unconditionally; when use_cyclical_lr
        # is False they are never actually used.
        _cycle_len = int(cycle_length) if cycle_length is not None else int(keep_every)
        _lr_max = float(lr_max) if lr_max is not None else float(lr) * 10.0
        _lr_min = float(lr)

        # Estimate the number of update steps.
        # Cyclical mode: each cycle of length _cycle_len produces one posterior
        # sample; n_discarded cycles are run first without collecting.
        if use_cyclical_lr and num_samples is not None:
            num_steps = (num_samples + n_discarded) * _cycle_len
        else:
            num_steps = 0 if num_samples is None else (num_samples + 1) * keep_every

        # Initialize the sampler
        if not continue_training:
            if clear_sampled_weights:
                self.sampled_weights.clear()
            self.net = self.net.float()
            self._initialize_sampler(
                num_datapoints, lr, mdecay, num_burn_in_steps, epsilon
            )
            num_steps += num_burn_in_steps

        # Initialize the batch generator
        batch_generator = islice(enumerate(train_loader), num_steps)

        # Start sampling
        self.net.train()
        n_samples = 0  # used to discard first samples
        for step, (x_batch, y_batch) in batch_generator:
            x_batch, y_batch = (
                x_batch.to(self.device, non_blocking=True),
                y_batch.to(self.device, non_blocking=True),
            )

            # --- Cyclical learning-rate schedule (Zhang et al. 2020) ----------
            # After burn-in the AdaptiveSGHMC preconditioner (τ, ĝ, v̂) is
            # frozen (iteration > num_burn_in_steps stops adaptation).  We then
            # cycle lr between _lr_min (cool/sampling phase) and _lr_max
            # (hot/exploration phase) using a cosine schedule.  Momentum is
            # zeroed at the start of each hot phase so the chain can escape
            # the current basin and explore new modes.
            if use_cyclical_lr and step >= num_burn_in_steps:
                _post_burn = step - num_burn_in_steps
                _cycle_step = _post_burn % _cycle_len
                _cycle_lr = _lr_min + 0.5 * (_lr_max - _lr_min) * (
                    1.0 + math.cos(math.pi * _cycle_step / _cycle_len)
                )
                for _pg in self.sampler.param_groups:
                    _pg["lr"] = np.float32(_cycle_lr)
                if _cycle_step == 0:
                    # Start of a new hot phase: zero momentum so the chain
                    # launches from rest toward unexplored regions.
                    for _pg in self.sampler.param_groups:
                        for _p in _pg["params"]:
                            _s = self.sampler.state.get(_p)
                            if _s is not None and "momentum" in _s:
                                _s["momentum"].zero_()
            # -----------------------------------------------------------------

            # Forward pass
            if self.task == "regression":
                x_batch = x_batch.view(y_batch.shape[0], -1)
                y_batch = y_batch.view(-1, 1)
                fx_batch = self.net(x_batch).view(-1, 1)
            elif self.task == "classification":
                fx_batch = self.net(x_batch, log_softmax=True)
            elif self.task == "pref":
                B, _, T, d_dim = x_batch.size()
                obs_dim = d_dim - 1
                am_1 = x_batch[:, 0, :, obs_dim]
                am_2 = x_batch[:, 1, :, obs_dim]
                x_batch_1 = x_batch[:, 0, :, :obs_dim].reshape(-1, obs_dim)
                x_batch_2 = x_batch[:, 1, :, :obs_dim].reshape(-1, obs_dim)

                # Run both arms through the network in one forward pass instead
                # of two, doubling GPU utilisation on this small MLP.
                pred_both = self.net(torch.cat([x_batch_1, x_batch_2], dim=0)).view(
                    2, B, T
                )
                pred_1 = pred_both[0] * am_1
                pred_2 = pred_both[1] * am_2

                sum_pred_1 = torch.nansum(pred_1, dim=1).view(-1, 1)
                sum_pred_2 = torch.nansum(pred_2, dim=1).view(-1, 1)
                fx_batch = torch.concatenate([sum_pred_1, sum_pred_2], dim=1)

            self.sampler.zero_grad()
            # Calculate the negative log joint density
            loss = self._neg_log_joint(fx_batch, y_batch, num_datapoints)

            # Estimate the gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100.0)

            # Update parameters
            self.sampler.step()
            self.step += 1

            # Resample hyper-parameters of the prior
            if self.prior_module.hyperprior:
                if step % resample_prior_every == 0:
                    if resample_hyper_prior_burn_in:
                        self.prior_module.resample(self._bare_net)
                    else:
                        if step > num_burn_in_steps:
                            self.prior_module.resample(self._bare_net)

            # Save the sampled weight.
            # Cyclical mode: collect one sample at the very end of each cycle
            # (final step of the cool phase, when lr is at its minimum).
            # Fixed-lr mode: original keep_every thinning.
            if use_cyclical_lr and step >= num_burn_in_steps:
                _post_burn = step - num_burn_in_steps
                _cycle_step = _post_burn % _cycle_len
                if _cycle_step == _cycle_len - 1:
                    n_samples += 1
                    if n_samples > n_discarded:
                        self.sampled_weights.append(self.network_weights)
                        self.num_samples += 1
            elif (not use_cyclical_lr) and (step > num_burn_in_steps) and (
                (step - num_burn_in_steps) % keep_every == 0
            ):
                n_samples += 1
                if n_samples > n_discarded:
                    # network_weights getter already returns fresh numpy arrays; deepcopy is redundant
                    self.sampled_weights.append(self.network_weights)
                    self.num_samples += 1

                    # # Print evaluation on training data
                    # if self.num_samples % print_every_n_samples == 0:
                    #     self.net.eval()
                    #     if (x_train is not None) and (y_train is not None):
                    #         self._print_evaluations(x_train, y_train, True, eval_map)
                    #     else:
                    #         self._print_evaluations(x_batch, y_batch, True, eval_map)
                    #     self.net.train()

    def _save_checkpoint(self, mode="best"):
        """Save sampled weights, sampler state into a single checkpoint file.

        Args:
            mode: str, the type of checkpoint to be saved. Possible values
                `last`, `best`.
        """
        if mode == "best":
            file_name = "checkpoint_best.pth"
        elif mode == "last":
            file_name = "checkpoint_last.pth"
        else:
            file_name = "checkpoint_step_{}.pth".format(self.step)

        file_path = os.path.join(self.ckpt_dir, file_name)

        torch.save(
            {
                "step": self.step,
                "num_samples": self.num_samples,
                "num_saved_sets_weights": self.num_saved_sets_weights,
                "sampler_params": self.sampler_params,
                "model_state_dict": self.net.state_dict(),
                "sampler_state_dict": self.sampler.state_dict(),
            },
            file_path,
        )

    def load_checkpoint(self, path, device):
        """Load sampled weights, sampler state from a checkpoint file.

        Args:
            path: str, the path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.step = checkpoint["step"]
        self.num_samples = checkpoint["num_samples"]
        self.num_saved_sets_weights = checkpoint["num_saved_sets_weights"]
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self._load_sampler(
            checkpoint["sampler_state_dict"], checkpoint["sampler_params"]
        )

    def _load_sampler(self, state_dict, sampler_params):
        """Load sampler from state dict and set of parameters"""
        self.sampler_params = sampler_params
        if self.sampling_method == "adaptive_sghmc":
            self.sampler = AdaptiveSGHMC(self.net.parameters(), **sampler_params)
        elif self.sampling_method == "sghmc":
            self.sampler = SGHMC(self.net.parameters(), **sampler_params)
        self.sampler.load_state_dict(state_dict)