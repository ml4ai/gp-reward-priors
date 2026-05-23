"""Bayesian Neural Network for regression."""

import copy
import os

import numpy as np
import torch
import wandb

from ..metrics.metrics_tensor import accuracy
from .bayes_net import BayesNet


def _pref_chain_worker(
    rank, batch_start, base_ckpt_dir, net_args, ckpt_path,
    x_train, y_train, seed, train_kwargs, initial_weights=None,
):
    """Worker for one parallel PrefNet chain, called by mp.spawn.

    Defined at module level (not inside a class or function) so it is
    picklable under the ``spawn`` start method that CUDA requires.  All
    arguments are plain Python objects (no lambdas, no ``__main__``
    references) so they survive the pickle/unpickle round-trip intact.

    Args:
        rank: process rank within the current batch — also the CUDA device
            index assigned to this worker.
        batch_start: chain index of rank 0 in this batch.
        base_ckpt_dir: root directory; chain i writes to
            ``<base_ckpt_dir>/chain_<i>/``.
        net_args: dict of keyword arguments forwarded to MLP(...).
        ckpt_path: path to the OptimGaussianPrior checkpoint file.
        x_train: numpy array, training inputs.
        y_train: numpy array, training targets.
        seed: base random seed; chain i uses seed + i.
        train_kwargs: dict forwarded verbatim to BayesNet.train().
        initial_weights: optional tuple of numpy arrays (one per parameter
            tensor, matching net.parameters() order).  When provided, all
            chains are loaded with these weights before sampling begins, so
            they start from the same basin rather than independent random
            initializations.  Each chain still uses a unique seed for SGHMC
            noise, so the trajectories diverge in a controlled way.
    """
    # Local imports: the spawned process starts fresh and needs its own
    # import chain.  Keeping them here also makes the function self-contained.
    from optbnn.bnn.likelihoods import LikCE
    from optbnn.bnn.nets.mlp import MLP
    from optbnn.bnn.priors import OptimGaussianPrior
    from optbnn.sgmcmc_bayes_net.pref_net import PrefNet
    from optbnn.utils.util import set_seed

    chain_idx = batch_start + rank
    # Bind this process to one GPU before any CUDA allocations.
    torch.cuda.set_device(rank)
    set_seed(seed + chain_idx)

    # Workers are spawned as fresh processes with no wandb session.  Disable
    # wandb so that the wandb.log() calls inside _print_evaluations become
    # silent no-ops.  The parent process logs all final metrics after the
    # workers finish.
    import wandb as _wandb
    _wandb.init(mode="disabled")

    net = MLP(**net_args)
    # Override random initialization with the shared starting point computed
    # by the parent process.  Without this, each chain starts in a different
    # random basin and the chains never mix — the dominant cause of high R-hat.
    # The per-chain seed (set above) still controls SGHMC noise, so trajectories
    # diverge from this common point in a controlled, reproducible way.
    if initial_weights is not None:
        with torch.no_grad():
            for param, w in zip(net.parameters(), initial_weights):
                param.copy_(torch.from_numpy(w))
    prior = OptimGaussianPrior(ckpt_path)
    likelihood = LikCE()

    chain_dir = os.path.join(base_ckpt_dir, f"chain_{chain_idx}")
    os.makedirs(chain_dir, exist_ok=True)

    # n_gpu=1: each worker owns exactly one GPU (the one set above).
    bayes_net = PrefNet(
        net, likelihood, prior, chain_dir,
        n_gpu=1, name=f"chain_{chain_idx}",
    )
    bayes_net.train(x_train, y_train, **train_kwargs)
    bayes_net._save_sampled_weights()


class PrefNet(BayesNet):
    def __init__(
        self,
        net,
        likelihood,
        prior,
        ckpt_dir,
        temperature=1.0,
        sampling_method="adaptive_sghmc",
        logger=None,
        n_gpu=0,
        name="bn",
    ):
        """Bayesian Neural Networks for regression task.

        Args:
            net: instance of nn.Module, the base neural network.
            likelihood: instance of LikelihoodModule, the module of likelihood.
            prior: instance of PriorModule, the module of prior.
            ckpt_dir: str, path to the directory of checkpoints.
            temperature: float, the temperature in posterior.
            sampling_method: specifies the sampling strategy.
            logger: instance of logging.Logger.
            n_gpu: int, the number of used GPUs.
        """
        BayesNet.__init__(
            self,
            net,
            likelihood,
            prior,
            ckpt_dir,
            temperature,
            sampling_method,
            weights_format="tuple",
            task="pref",
            logger=logger,
            n_gpu=n_gpu,
        )
        self.name = name

    def train_and_evaluate(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        num_samples=1000,
        keep_every=100,
        lr=1e-2,
        mdecay=0.05,
        batch_size=20,
        num_burn_in_steps=3000,
        validate_every_n_samples=10,
        print_every_n_samples=5,
        epsilon=1e-10,
        continue_training=False,
    ):
        """
        Train and validates the bayesian neural network

        Args:
            x_train: numpy array, input training datapoints.
            y_train: numpy array, input training targets.
            x_valid: numpy array, input validation datapoints.
            y_valid: numpy array, input validation targets.
            num_samples: int, number of set of parameters we want to sample.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            lr: float, learning rate.
            mdecay: float, momemtum decay.
            batch_size: int, batch size.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            validate_every_n_samples: int, defines after how many samples we
                want to evaluate the sampled weights on validation data.
            print_every_n_samples: int, defines after how many samples we want
                to evaluate the sampled weights on training data.
            epsilon: float, epsilon for numerical stability.
            continue_training: bool, defines whether we want to continue
                from the last training run.
        """
        # Burn-in steps
        self.print_info("Burn-in steps")
        self.train(
            x_train=x_train,
            y_train=y_train,
            num_burn_in_steps=num_burn_in_steps,
            lr=lr,
            epsilon=epsilon,
            mdecay=mdecay,
        )

        self.print_info("Start sampling")
        for i in range(num_samples // validate_every_n_samples):
            self.train(
                x_train=x_train,
                y_train=y_train,
                num_burn_in_steps=0,
                num_samples=validate_every_n_samples,
                batch_size=batch_size,
                lr=lr,
                epsilon=epsilon,
                mdecay=mdecay,
                keep_every=keep_every,
                continue_training=True,
                print_every_n_samples=print_every_n_samples,
            )
            self._print_evaluations(x_valid, y_valid, False)

        self._save_sampled_weights()
        self.print_info("Finish")

    def predict(
        self, x_test, return_individual_predictions=False, use_map=False, map_only=False
    ):
        """Predicts mean and variance for the given test point.

        Args:
            x_test: numpy array, can be (d,) for a single datapoint or (n,d) for a batch of datapoints
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            use_map: bool, also returns predictions using the map estimate. Asserts that self.map is not None
            map_only: bool, only returns map prediction, ignores use_map. Asserts that self.map is not None

        Returns:
            a tuple consisting of mean and variance.
        """

        x_test_ = np.asarray(x_test)
        x_tensor = torch.from_numpy(x_test_).float().to(self.device)

        def network_predict(weights):
            with torch.no_grad():
                self.network_weights = weights
                return self.net(x_tensor).detach().cpu().numpy()

        if map_only:
            assert self.map is not None
            pred_map = network_predict(weights=self.map)
            return pred_map
        else:
            # Make predictions for each sampled weights
            # the shape for this is (n_weights,samples,1)
            predictions = np.array(
                [network_predict(weights=weights) for weights in self.sampled_weights]
            )

            # Calculates the predictive mean and variance
            # The shape for these is (samples,1)
            pred_mean = np.mean(predictions, axis=0)
            pred_var = np.var(predictions, axis=0)

            if return_individual_predictions:
                if use_map:
                    assert self.map is not None
                    pred_map = network_predict(weights=self.map)
                    return pred_mean, pred_var, pred_map, predictions
                return pred_mean, pred_var, predictions

            if use_map:
                assert self.map is not None
                pred_map = network_predict(weights=self.map)
                return pred_mean, pred_var, pred_map
            return pred_mean, pred_var

    def _predict_pairs_batched(self, x_1, x_2, am_1, am_2, T, use_map=False, batch_size=256):
        """Run network predictions over preference pairs in mini-batches.

        Processes ``batch_size`` pairs at a time so that the GPU only
        ever holds ``2 * batch_size * T`` observation vectors, preventing
        OOM on large (N × T) datasets.  The summed rewards that are
        returned are numerically identical to what a single full-dataset
        forward pass would produce.

        Args:
            x_1: numpy (N*T, obs_dim) — arm-1 observations, already flattened.
            x_2: numpy (N*T, obs_dim) — arm-2 observations.
            am_1: numpy (N, T)        — arm-1 attention mask.
            am_2: numpy (N, T)        — arm-2 attention mask.
            T: int, number of timesteps per trajectory.
            use_map: bool, use the MAP weight set instead of the posterior mean.
            batch_size: int, number of preference pairs per mini-batch.

        Returns:
            sum_pred_1: numpy (N,) — masked reward sums for arm 1.
            sum_pred_2: numpy (N,) — masked reward sums for arm 2.
        """
        N = am_1.shape[0]
        parts_1, parts_2 = [], []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start

            x1_b = x_1[start * T : end * T]          # (b*T, obs_dim)
            x2_b = x_2[start * T : end * T]
            x_both = np.concatenate([x1_b, x2_b], axis=0)   # (2*b*T, obs_dim)

            if use_map:
                _, _, pred = self.predict(x_both, use_map=True)
            else:
                pred, _ = self.predict(x_both)

            pred_1 = pred[: b * T].reshape(b, T) * am_1[start:end]
            pred_2 = pred[b * T :].reshape(b, T) * am_2[start:end]

            parts_1.append(np.nansum(pred_1, axis=1))
            parts_2.append(np.nansum(pred_2, axis=1))

        return np.concatenate(parts_1), np.concatenate(parts_2)

    def _ce_and_acc(self, sum_pred_1, sum_pred_2, y):
        """Compute cross-entropy and accuracy from per-pair reward sums."""
        fx = np.stack([sum_pred_1, sum_pred_2], axis=1).astype(np.float32)
        fx_t = torch.from_numpy(fx).to(self.device)
        y_t  = torch.from_numpy(y).float().to(self.device)
        ce  = torch.nn.CrossEntropyLoss()(fx_t, y_t).detach().cpu().numpy()
        acc = accuracy(fx_t, y_t).detach().cpu().numpy()
        return ce, acc

    def _print_evaluations(self, x, y, train=True, eval_map=False, eval_batch_size=256):
        """Evaluate the sampled weights on training/validation data and
            during the training log the results.

        Args:
            x: numpy array, shape [N, 2, T, d_dim], the input data.
            y: numpy array, shape [N], the corresponding labels.
            train: bool, indicate whether we're evaluating on the training data.
            eval_map: bool, use MAP estimate if True, posterior mean otherwise.
            eval_batch_size: int, number of pairs per prediction mini-batch.
        """
        self.net.eval()
        B, _, T, d_dim = x.shape
        obs_dim = d_dim - 1
        am_1 = x[:, 0, :, obs_dim]
        am_2 = x[:, 1, :, obs_dim]
        x_1  = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
        x_2  = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

        if eval_map:
            self.find_map(x, y)

        sum_1, sum_2 = self._predict_pairs_batched(
            x_1, x_2, am_1, am_2, T,
            use_map=eval_map,
            batch_size=eval_batch_size,
        )
        ce, acc = self._ce_and_acc(sum_1, sum_2, y)

        if eval_map:
            self.map = None

        if train:
            wandb.log(
                {
                    f"{self.name}_training_mean_cross_entropy": ce,
                    f"{self.name}_training_mean_accuracy": acc,
                },
                step=self.num_samples,
            )
            self.print_info(
                "Samples # {:5d} : CE = {:.4f} ACC = {:.4f} ".format(
                    self.num_samples, ce, acc
                )
            )
        else:
            wandb.log(
                {
                    f"{self.name}_eval_mean_cross_entropy": ce,
                    f"{self.name}_eval_mean_accuracy": acc,
                }
            )
            self.print_info("Validation: CE = {:.4f} ACC = {:.4f}".format(ce, acc))

        self.net.train()

    def eval_test_data(self, x, y, x_map=None, y_map=None, eval_batch_size=256):
        """Evaluate the sampled weights on test data.

        Predictions are run in mini-batches of ``eval_batch_size`` pairs to
        cap GPU memory usage at ``O(2 * eval_batch_size * T)`` observations
        per forward pass, regardless of total dataset size.

        If ``x_map`` and ``y_map`` are provided, the MAP weight estimate
        (selected from ``self.sampled_weights``) is used for prediction;
        otherwise the posterior predictive mean over all sampled weights is
        used.

        Args:
            x: numpy array, shape [N, 2, T, d_dim], the test inputs.
            y: numpy array, shape [N], the preference labels.
            x_map: numpy array, shape [M, 2, T, d_dim], data for MAP selection.
            y_map: numpy array, shape [M], labels for MAP selection.
            eval_batch_size: int, number of preference pairs per mini-batch.

        Returns:
            ce:  float, mean cross-entropy over the test set.
            acc: float, mean accuracy over the test set.
        """
        self.net.eval()
        B, _, T, d_dim = x.shape
        obs_dim = d_dim - 1
        am_1 = x[:, 0, :, obs_dim]
        am_2 = x[:, 1, :, obs_dim]
        x_1  = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
        x_2  = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

        use_map = (x_map is not None) and (y_map is not None)
        if use_map:
            self.find_map(x_map, y_map)

        sum_1, sum_2 = self._predict_pairs_batched(
            x_1, x_2, am_1, am_2, T,
            use_map=use_map,
            batch_size=eval_batch_size,
        )
        ce, acc = self._ce_and_acc(sum_1, sum_2, y)

        if use_map:
            self.map = None
        self.net.train()
        return ce, acc

    def find_map(self, x, y, max_map_samples=512):
        """find the map estimate given a set of data and set of sampled weights.
           Asserts that self.sampled_weights is not empty

        Args:
            x: numpy array, shape [N, 2, T, d_dim], the preference-pair inputs.
            y: numpy array, shape [N] or [N, 1], the corresponding labels.
            max_map_samples: int, maximum number of pairs to use when computing
                the loss for each candidate weight set.  Only the ranking of
                losses matters (not their absolute values), so a random subset
                is sufficient and avoids OOM for large training sets whose full
                forward pass would exceed GPU memory.
        """
        assert self.sampled_weights

        # Subsample along the pair axis before moving anything to the GPU.
        if x.shape[0] > max_map_samples:
            idx = np.random.choice(x.shape[0], max_map_samples, replace=False)
            x, y = x[idx], y[idx]

        def network_loss(x, y, weights):
            with torch.no_grad():
                self.network_weights = weights
                B, _, T, d_dim = x.size()
                obs_dim = d_dim - 1
                am_1 = x[:, 0, :, obs_dim]
                am_2 = x[:, 1, :, obs_dim]
                x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
                x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

                pred_both = self.net(torch.cat([x_1, x_2], dim=0)).view(2, B, T)
                pred_1 = pred_both[0] * am_1
                pred_2 = pred_both[1] * am_2

                sum_pred_1 = torch.nansum(pred_1, dim=1).view(-1, 1)
                sum_pred_2 = torch.nansum(pred_2, dim=1).view(-1, 1)
                fx_batch = torch.concatenate([sum_pred_1, sum_pred_2], dim=1)
                return (
                    self._neg_log_joint(
                        fx_batch,
                        y,
                        B,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        x_t, y_t = (
            torch.from_numpy(x.squeeze()).float().to(self.device),
            torch.from_numpy(y.squeeze()).float().to(self.device),
        )
        losses = np.array(
            [network_loss(x_t, y_t, weights=weights) for weights in self.sampled_weights]
        )
        self.map = self.sampled_weights[np.argmin(losses)]

    def compute_covariance(self, X):
        """Predicts mean and variance for the given test point.

        Args:
            x_test: numpy array, can be (d,) for a single datapoint or (n,d) for a batch of datapoints
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            use_map: bool, also returns predictions using the map estimate. Asserts that self.map is not None
            map_only: bool, only returns map prediction, ignores use_map. Asserts that self.map is not None

        Returns:
            a tuple consisting of mean and variance.
        """

        def network_predict(X, weights):
            with torch.no_grad():
                self.network_weights = weights
                return self.net(X.float().to(self.device))

        predictions = (
            torch.stack(
                [network_predict(X, weights=weights) for weights in self.sampled_weights]
            )
            .squeeze()
            .T
        )

        return torch.cov(predictions).double()

    def sample_multi_chains_parallel(
        self,
        x_train,
        y_train,
        net_args,
        ckpt_path,
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
        seed=1,
        initial_weights=None,
        use_cyclical_lr=False,
        lr_max=None,
        cycle_length=None,
        fraction_cool=0.25,
    ):
        """Run multiple chains in parallel, one process per GPU.

        If ``num_chains > torch.cuda.device_count()``, chains are dispatched
        in consecutive batches of ``min(num_chains, num_gpus)`` chains.  Each
        batch runs fully in parallel (all processes on separate GPUs) before
        the next batch begins.

        Sampled weights for chain i are written to::

            <self.ckpt_dir>/chain_<i>/sampled_weights/sampled_weights_0000000

        After this method returns, load each chain with::

            self._load_sampled_weights(
                os.path.join(self.ckpt_dir, f"chain_{i}",
                             "sampled_weights", "sampled_weights_0000000")
            )

        Args:
            x_train: numpy array, training inputs.
            y_train: numpy array, training targets.
            net_args: dict of keyword arguments for MLP (input_dim,
                output_dim, hidden_dims, activation_fn, …).  Must be
                picklable (plain Python types only).
            ckpt_path: str, path to the OptimGaussianPrior checkpoint.
            num_chains: int, total number of chains to run.
            seed: int, base random seed.  Chain i uses ``seed + i``.
            initial_weights: optional tuple of numpy arrays (one per
                parameter tensor) to use as the shared starting point for
                all chains.  Compute this in the parent process via a short
                warm-up burn-in and pass the result here to avoid chains
                being trapped in different random basins.
            (remaining args forwarded to BayesNet.train())
        """
        import torch.multiprocessing as mp

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError(
                "sample_multi_chains_parallel requires at least one CUDA device."
            )

        train_kwargs = dict(
            num_samples=num_samples,
            keep_every=keep_every,
            n_discarded=n_discarded,
            num_burn_in_steps=num_burn_in_steps,
            lr=lr,
            batch_size=batch_size,
            epsilon=epsilon,
            mdecay=mdecay,
            print_every_n_samples=print_every_n_samples,
            continue_training=False,
            clear_sampled_weights=True,
            resample_prior_every=resample_prior_every,
            eval_map=eval_map,
            use_cyclical_lr=use_cyclical_lr,
            lr_max=lr_max,
            cycle_length=cycle_length,
            fraction_cool=fraction_cool,
        )

        for batch_start in range(0, num_chains, num_gpus):
            n_parallel = min(num_gpus, num_chains - batch_start)
            self.print_info(
                "Launching chains {:d}–{:d} in parallel on {:d} GPU(s)".format(
                    batch_start, batch_start + n_parallel - 1, n_parallel
                )
            )
            mp.spawn(
                _pref_chain_worker,
                args=(
                    batch_start,
                    self.ckpt_dir,
                    net_args,
                    ckpt_path,
                    x_train,
                    y_train,
                    seed,
                    train_kwargs,
                    initial_weights,
                ),
                nprocs=n_parallel,
                join=True,  # block until all chains in this batch finish
            )