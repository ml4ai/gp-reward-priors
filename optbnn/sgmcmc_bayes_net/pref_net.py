"""Bayesian Neural Network for regression."""

import copy

import numpy as np
import torch
import wandb

from ..metrics.metrics_tensor import accuracy
from .bayes_net import BayesNet


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

    def predict(self, x_test, return_individual_predictions=False, use_map=False):
        """Predicts mean and variance for the given test point.

        Args:
            x_test: numpy array, can be (d,) for a single datapoint or (n,d) for a batch of datapoints
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            use_map: bool, also returns predictions using the map estimate. Asserts that self.map is not None

        Returns:
            a tuple consisting of mean and variance.
        """

        x_test_ = np.asarray(x_test)

        def network_predict(x_test_, weights):
            with torch.no_grad():
                self.network_weights = weights
                return (
                    self.net(torch.from_numpy(x_test_).float().to(self.device))
                    .detach()
                    .cpu()
                    .numpy()
                )

        # Make predictions for each sampled weights
        # the shape for this is (n_weights,samples,1)
        predictions = np.array(
            [
                network_predict(x_test_, weights=weights)
                for weights in self.sampled_weights
            ]
        )

        # Calculates the predictive mean and variance
        # The shape for these is (samples,1)
        pred_mean = np.mean(predictions, axis=0)
        pred_var = np.var(predictions, axis=0)

        if return_individual_predictions:
            if use_map:
                assert self.map is not None
                pred_map = network_predict(x_test_, weights=self.map)
                return pred_mean, pred_var, pred_map, predictions
            return pred_mean, pred_var, predictions

        if use_map:
            assert self.map is not None
            pred_map = network_predict(x_test_, weights=self.map)
            return pred_mean, pred_var, pred_map
        return pred_mean, pred_var

    def _print_evaluations(self, x, y, train=True, eval_map=False):
        """Evaluate the sampled weights on training/validation data and
            during the training log the results.

        Args:
            x: numpy array, shape [batch_size, num_features], the input data.
            y: numpy array, shape [batch_size, 1], the corresponding targets.
            train: bool, indicate whether we're evaluating on the training data.
        """
        self.net.eval()
        if eval_map:
            self.find_map(x, y)
            B, _, T, d_dim = x.shape
            obs_dim = d_dim - 1
            am_1 = x[:, 0, :, obs_dim]
            am_2 = x[:, 1, :, obs_dim]
            x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
            x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

            _, _, pred_mean_1 = self.predict(x_1, use_map=True)
            _, _, pred_mean_2 = self.predict(x_2, use_map=True)
            pred_mean_1 = pred_mean_1.reshape(B, T) * am_1
            pred_mean_2 = pred_mean_2.reshape(B, T) * am_2

            sum_pred_1 = np.nansum(pred_mean_1, axis=1).reshape(-1, 1)
            sum_pred_2 = np.nansum(pred_mean_2, axis=1).reshape(-1, 1)
            # shape is (B,2)
            fx_batch = np.concatenate([sum_pred_1, sum_pred_2], axis=1)
            loss = torch.nn.CrossEntropyLoss()
            ce = (
                loss(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )
            acc = (
                accuracy(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )

            if train:
                wandb.log(
                    {
                        f"{self.name}_training_mean_cross_entropy": ce,
                        f"{self.name}_training_mean_accuracy": acc,
                    },
                    step=self.num_samples,
                )
                self.print_info(
                    "Samples # {:5d} : CE = {:.4f} "
                    "ACC = {:.4f} ".format(self.num_samples, ce, acc)
                )
            else:
                wandb.log(
                    {
                        f"{self.name}_eval_mean_cross_entropy": ce,
                        f"{self.name}_eval_mean_accuracy": acc,
                    }
                )
                self.print_info("Validation: CE = {:.4f} ACC = {:.4f}".format(ce, acc))
            self.map = None
            self.net.train()
        else:
            B, _, T, d_dim = x.shape
            obs_dim = d_dim - 1
            am_1 = x[:, 0, :, obs_dim]
            am_2 = x[:, 1, :, obs_dim]
            x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
            x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

            pred_mean_1, _ = self.predict(x_1)
            pred_mean_2, _ = self.predict(x_2)
            pred_mean_1 = pred_mean_1.reshape(B, T) * am_1
            pred_mean_2 = pred_mean_2.reshape(B, T) * am_2

            sum_pred_1 = np.nansum(pred_mean_1, axis=1).reshape(-1, 1)
            sum_pred_2 = np.nansum(pred_mean_2, axis=1).reshape(-1, 1)
            # shape is (B,2)
            fx_batch = np.concatenate([sum_pred_1, sum_pred_2], axis=1)
            loss = torch.nn.CrossEntropyLoss()
            ce = (
                loss(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )
            acc = (
                accuracy(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )

            if train:
                wandb.log(
                    {
                        f"{self.name}_training_mean_cross_entropy": ce,
                        f"{self.name}_training_mean_accuracy": acc,
                    },
                    step=self.num_samples,
                )
                self.print_info(
                    "Samples # {:5d} : CE = {:.4f} "
                    "ACC = {:.4f} ".format(self.num_samples, ce, acc)
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

    def eval_test_data(self, x, y, x_map=None, y_map=None):
        """Evaluate the sampled weights on training/validation data and
            during the training log the results. If x_map and y_map are not None, then
            the eval is done using map estimate computed from x_map and y_map.

        Args:
            x: numpy array, shape [batch_size, num_features], the input data.
            y: numpy array, shape [batch_size, 1], the corresponding targets.
            x_map: numpy array, shape [training_size, num_features], input data used to find map
            y_map: numpy array, shape [training_size, 1], the targets used to find map
        """
        self.net.eval()
        if (x_map is not None) and (y_map is not None):
            self.find_map(x_map, y_map)
            B, _, T, d_dim = x.shape
            obs_dim = d_dim - 1
            am_1 = x[:, 0, :, obs_dim]
            am_2 = x[:, 1, :, obs_dim]
            x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
            x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

            _, _, pred_mean_1 = self.predict(x_1, use_map=True)
            _, _, pred_mean_2 = self.predict(x_2, use_map=True)
            pred_mean_1 = pred_mean_1.reshape(B, T) * am_1
            pred_mean_2 = pred_mean_2.reshape(B, T) * am_2

            sum_pred_1 = np.nansum(pred_mean_1, axis=1).reshape(-1, 1)
            sum_pred_2 = np.nansum(pred_mean_2, axis=1).reshape(-1, 1)
            # shape is (B,2)
            fx_batch = np.concatenate([sum_pred_1, sum_pred_2], axis=1)
            loss = torch.nn.CrossEntropyLoss()
            ce = (
                loss(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )
            acc = (
                accuracy(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )

            wandb.log(
                {
                    f"{self.name}_eval_mean_cross_entropy": ce,
                    f"{self.name}_eval_mean_accuracy": acc,
                }
            )
            self.print_info("Validation: CE = {:.4f} ACC = {:.4f}".format(ce, acc))
            self.map = None
            self.net.train()
        else:
            B, _, T, d_dim = x.shape
            obs_dim = d_dim - 1
            am_1 = x[:, 0, :, obs_dim]
            am_2 = x[:, 1, :, obs_dim]
            x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
            x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

            pred_mean_1, _ = self.predict(x_1)
            pred_mean_2, _ = self.predict(x_2)
            pred_mean_1 = pred_mean_1.reshape(B, T) * am_1
            pred_mean_2 = pred_mean_2.reshape(B, T) * am_2

            sum_pred_1 = np.nansum(pred_mean_1, axis=1).reshape(-1, 1)
            sum_pred_2 = np.nansum(pred_mean_2, axis=1).reshape(-1, 1)
            # shape is (B,2)
            fx_batch = np.concatenate([sum_pred_1, sum_pred_2], axis=1)
            loss = torch.nn.CrossEntropyLoss()
            ce = (
                loss(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )
            acc = (
                accuracy(
                    torch.from_numpy(fx_batch).float().to(self.device),
                    torch.from_numpy(y).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )

            wandb.log(
                {
                    f"{self.name}_eval_mean_cross_entropy": ce,
                    f"{self.name}_eval_mean_accuracy": acc,
                }
            )
            self.print_info("Validation: CE = {:.4f} ACC = {:.4f}".format(ce, acc))
            self.net.train()

    def find_map(self, x, y):
        """find the map estimate given a set of data and set of sampled weights.
           Asserts that self.sampled_weights is not empty

        Args:
            x: numpy array, shape [batch_size, num_features], the input data.
            y: numpy array, shape [batch_size, 1], the corresponding targets.
        """
        assert self.sampled_weights

        def network_loss(x, y, weights):
            with torch.no_grad():
                self.network_weights = weights
                B, _, T, d_dim = x.size()
                obs_dim = d_dim - 1
                am_1 = x[:, 0, :, obs_dim]
                am_2 = x[:, 1, :, obs_dim]
                x_1 = x[:, 0, :, :obs_dim].reshape(-1, obs_dim)
                x_2 = x[:, 1, :, :obs_dim].reshape(-1, obs_dim)

                pred_1 = self.net(x_1).view(B, T) * am_1
                pred_2 = self.net(x_2).view(B, T) * am_2

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
            torch.from_numpy(y.squeeze()).long().to(self.device),
        )
        losses = np.array(
            [
                network_loss(x_t, y_t, weights=weights)
                for weights in self.sampled_weights
            ]
        )
        self.map = self.sampled_weights[np.argmin(losses)]
