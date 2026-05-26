from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from optbnn.bnn.priors import PriorModule
from optbnn.gp.models.model import LCFModel

TensorBatch = List[torch.Tensor]


class MRTrainer:
    def __init__(
        self,
        net: nn.Module,
        opt: torch.optim.Optimizer,
        num_datapoints: int,
        prior: Optional[PriorModule] = None,
        device: str = "cpu",
    ):
        self.net = net
        self.opt = opt
        self.prior = prior
        self.device = device
        self.like = torch.nn.CrossEntropyLoss(reduction="mean")
        self.num_datapoints = num_datapoints

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            states,
            actions,
            timesteps,
            attn_mask,
            states_2,
            actions_2,
            timesteps_2,
            attn_mask_2,
            labels,
        ) = batch
        if not self.net.training:
            self.net.train()
        log_dict = {}

        B, T, s_dim = states.size()
        _, _, a_dim = actions.size()
        # Run both trajectory segments in a single forward pass by doubling the
        # batch dimension, then split the result. This halves dispatch overhead
        # and gives the GPU a larger matrix multiply to work with.
        X_batch = torch.cat([
            torch.cat([states, actions], dim=-1).reshape(B * T, s_dim + a_dim),
            torch.cat([states_2, actions_2], dim=-1).reshape(B * T, s_dim + a_dim),
        ], dim=0)  # (2*B*T, s_dim+a_dim)
        pred = self.net(X_batch)  # (2*B*T, 1)
        pred_1 = pred[: B * T].reshape(B, T) * attn_mask
        pred_2 = pred[B * T :].reshape(B, T) * attn_mask_2

        sum_pred_1 = torch.nansum(pred_1, dim=1).reshape(-1, 1)
        sum_pred_2 = torch.nansum(pred_2, dim=1).reshape(-1, 1)
        fX_batch = torch.cat([sum_pred_1, sum_pred_2], dim=1)
        if self.prior is None:
            loss = self.like(fX_batch, labels)
        else:
            loss = (
                self.like(fX_batch, labels) + self.prior(self.net) / self.num_datapoints
            )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            log_dict["training_loss"] = loss.item()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["training_acc"] = (
                (predicted_class == target_class).float().mean().item()
            )
            return log_dict

    def evaluation(self, batch: TensorBatch) -> Dict[str, float]:
        (
            states,
            actions,
            timesteps,
            attn_mask,
            states_2,
            actions_2,
            timesteps_2,
            attn_mask_2,
            labels,
        ) = batch
        if self.net.training:
            self.net.eval()
        with torch.no_grad():
            log_dict = {}

            B, T, s_dim = states.size()
            _, _, a_dim = actions.size()
            X_batch = torch.cat([
                torch.cat([states, actions], dim=-1).reshape(B * T, s_dim + a_dim),
                torch.cat([states_2, actions_2], dim=-1).reshape(B * T, s_dim + a_dim),
            ], dim=0)  # (2*B*T, s_dim+a_dim)
            pred = self.net(X_batch)  # (2*B*T, 1)
            pred_1 = pred[: B * T].reshape(B, T) * attn_mask
            pred_2 = pred[B * T :].reshape(B, T) * attn_mask_2

            sum_pred_1 = torch.nansum(pred_1, dim=1).reshape(-1, 1)
            sum_pred_2 = torch.nansum(pred_2, dim=1).reshape(-1, 1)
            fX_batch = torch.cat([sum_pred_1, sum_pred_2], dim=1)
            if self.prior is None:
                loss = self.like(fX_batch, labels)
            else:
                loss = (
                    self.like(fX_batch, labels)
                    + self.prior(self.net) / self.num_datapoints
                )

            log_dict["eval_loss"] = loss.item()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["eval_acc"] = (
                (predicted_class == target_class).float().mean().item()
            )

            return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.net.load_state_dict(state_dict["net"])
        self.opt.load_state_dict(state_dict["optimizer"])


class MRTrainerF:
    """Functional MAP trainer for preference reward learning.

    Minimises the potential energy

        U(f) = -log p(Y_D | f(X_D; w)) + I_0(f)

    where I_0(f) = -(1/2)(f(X_M;w)-m)^T K^{-1}(f(X_M;w)-m) is the
    Onsager-Machlup functional for the GP prior (Wu et al., AISTATS 2025).

    Unlike MRTrainer, the prior is **required** and must be an LCFModel.
    The train() method additionally accepts a batch of measurement points
    X_M (and optional auxiliary inputs aux_meas) used to evaluate the
    functional prior gradient.
    """

    def __init__(
        self,
        net: nn.Module,
        opt: torch.optim.Optimizer,
        num_datapoints: int,
        prior: LCFModel,
        device: str = "cpu",
    ):
        self.net = net
        self.opt = opt
        self.prior = prior
        self.device = device
        self.like = torch.nn.CrossEntropyLoss(reduction="mean")
        self.num_datapoints = num_datapoints

    def train(
        self,
        batch: TensorBatch,
        x_meas,
        aux_meas=None,
        jitter: float = 1e-6,
    ) -> Dict[str, float]:
        (
            states,
            actions,
            timesteps,
            attn_mask,
            states_2,
            actions_2,
            timesteps_2,
            attn_mask_2,
            labels,
        ) = batch
        if not self.net.training:
            self.net.train()
        log_dict = {}

        B, T, s_dim = states.size()
        _, _, a_dim = actions.size()
        # Single forward pass over both trajectory segments.
        X_batch = torch.cat([
            torch.cat([states, actions], dim=-1).reshape(B * T, s_dim + a_dim),
            torch.cat([states_2, actions_2], dim=-1).reshape(B * T, s_dim + a_dim),
        ], dim=0)  # (2*B*T, s_dim+a_dim)
        pred = self.net(X_batch)  # (2*B*T, 1)
        pred_1 = pred[: B * T].reshape(B, T) * attn_mask
        pred_2 = pred[B * T :].reshape(B, T) * attn_mask_2

        sum_pred_1 = torch.nansum(pred_1, dim=1).reshape(-1, 1)
        sum_pred_2 = torch.nansum(pred_2, dim=1).reshape(-1, 1)
        fX_batch = torch.cat([sum_pred_1, sum_pred_2], dim=1)

        loss = self.like(fX_batch, labels)

        # --- Functional MAP gradient step ---
        # 1. Accumulate likelihood gradient: ∇_w [-log p(Y|f)] / B
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # 2. Ensure measurement points are float32 tensors on the right device.
        x_meas_t = torch.as_tensor(x_meas, dtype=torch.float32, device=self.device)
        aux_meas_t = (
            torch.as_tensor(aux_meas, dtype=torch.float32, device=self.device)
            if aux_meas is not None
            else None
        )

        # 3. Compute functional prior gradient: ∇_w log p_GP(f(·;w)).
        #    functional_prior_grad returns a tuple of gradient tensors (one per
        #    network parameter) pointing in the direction of *increasing* log-prior.
        #    Subtracting fg/N adds +∇_w I_0/N to the existing likelihood gradient,
        #    giving the full energy gradient ∇_w U(f) / N for Adam to descend.
        func_grads = self.prior.functional_prior_grad(
            self.net, x_meas_t, aux_X=aux_meas_t, jitter=jitter
        )
        for param, fg in zip(self.net.parameters(), func_grads):
            if param.grad is not None:
                param.grad.add_(-fg.to(param.grad.dtype) / self.num_datapoints)

        self.opt.step()

        with torch.no_grad():
            log_dict["training_loss"] = loss.item()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["training_acc"] = (
                (predicted_class == target_class).float().mean().item()
            )
            return log_dict

    def evaluation(self, batch: TensorBatch) -> Dict[str, float]:
        (
            states,
            actions,
            timesteps,
            attn_mask,
            states_2,
            actions_2,
            timesteps_2,
            attn_mask_2,
            labels,
        ) = batch
        if self.net.training:
            self.net.eval()
        with torch.no_grad():
            log_dict = {}

            B, T, s_dim = states.size()
            _, _, a_dim = actions.size()
            X_batch = torch.cat([
                torch.cat([states, actions], dim=-1).reshape(B * T, s_dim + a_dim),
                torch.cat([states_2, actions_2], dim=-1).reshape(B * T, s_dim + a_dim),
            ], dim=0)  # (2*B*T, s_dim+a_dim)
            pred = self.net(X_batch)  # (2*B*T, 1)
            pred_1 = pred[: B * T].reshape(B, T) * attn_mask
            pred_2 = pred[B * T :].reshape(B, T) * attn_mask_2

            sum_pred_1 = torch.nansum(pred_1, dim=1).reshape(-1, 1)
            sum_pred_2 = torch.nansum(pred_2, dim=1).reshape(-1, 1)
            fX_batch = torch.cat([sum_pred_1, sum_pred_2], dim=1)

            loss = self.like(fX_batch, labels)

            log_dict["eval_loss"] = loss.item()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["eval_acc"] = (
                (predicted_class == target_class).float().mean().item()
            )

            return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.net.load_state_dict(state_dict["net"])
        self.opt.load_state_dict(state_dict["optimizer"])


class PTTrainer:
    def __init__(
        self,
        net: nn.Module,
        opt: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.net = net
        self.opt = opt
        self.device = device
        self.like = torch.nn.CrossEntropyLoss(reduction="mean")

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            states,
            actions,
            timesteps,
            attn_mask,
            states_2,
            actions_2,
            timesteps_2,
            attn_mask_2,
            labels,
        ) = batch
        if not self.net.training:
            self.net.train()
        log_dict = {}

        B, T, _ = states.size()

        # Concatenate both trajectory segments along the batch dimension so the
        # transformer runs a single forward pass of size 2B instead of two of B.
        out, _ = self.net(
            torch.cat([states, states_2], dim=0),
            torch.cat([actions, actions_2], dim=0),
            torch.cat([timesteps, timesteps_2], dim=0).to(torch.int),
            torch.cat([attn_mask, attn_mask_2], dim=0),
        )
        pred = out["weighted_sum"]  # (2B, T, 1)
        sum_pred_1 = pred[:B].reshape(B, T).mean(dim=1, keepdim=True)
        sum_pred_2 = pred[B:].reshape(B, T).mean(dim=1, keepdim=True)
        fX_batch = torch.cat([sum_pred_1, sum_pred_2], dim=1)

        loss = self.like(fX_batch, labels)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            log_dict["training_loss"] = loss.item()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["training_acc"] = (
                (predicted_class == target_class).float().mean().item()
            )
            return log_dict

    def evaluation(self, batch: TensorBatch) -> Dict[str, float]:
        (
            states,
            actions,
            timesteps,
            attn_mask,
            states_2,
            actions_2,
            timesteps_2,
            attn_mask_2,
            labels,
        ) = batch
        if self.net.training:
            self.net.eval()
        with torch.no_grad():
            log_dict = {}

            B, T, _ = states.size()

            out, _ = self.net(
                torch.cat([states, states_2], dim=0),
                torch.cat([actions, actions_2], dim=0),
                torch.cat([timesteps, timesteps_2], dim=0).to(torch.int),
                torch.cat([attn_mask, attn_mask_2], dim=0),
            )
            pred = out["weighted_sum"]  # (2B, T, 1)
            sum_pred_1 = pred[:B].reshape(B, T).mean(dim=1, keepdim=True)
            sum_pred_2 = pred[B:].reshape(B, T).mean(dim=1, keepdim=True)
            fX_batch = torch.cat([sum_pred_1, sum_pred_2], dim=1)

            loss = self.like(fX_batch, labels)

            log_dict["eval_loss"] = loss.item()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["eval_acc"] = (
                (predicted_class == target_class).float().mean().item()
            )

            return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.net.load_state_dict(state_dict["net"])
        self.opt.load_state_dict(state_dict["optimizer"])
