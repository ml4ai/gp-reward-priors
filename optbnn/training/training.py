from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from optbnn.bnn.priors import PriorModule

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
        self.net.train()
        log_dict = {}

        B, T, s_dim = states.size()
        _, _, a_dim = actions.size()
        X_batch_1 = torch.concatenate([states, actions], dim=-1).reshape(
            -1, s_dim + a_dim
        )
        X_batch_2 = torch.concatenate([states_2, actions_2], dim=-1).reshape(
            -1, s_dim + a_dim
        )

        pred_1 = self.net(X_batch_1).reshape(B, T) * attn_mask
        pred_2 = self.net(X_batch_2).reshape(B, T) * attn_mask_2

        sum_pred_1 = torch.nansum(pred_1, dim=1).reshape(-1, 1)
        sum_pred_2 = torch.nansum(pred_2, dim=1).reshape(-1, 1)
        fX_batch = torch.concatenate([sum_pred_1, sum_pred_2], dim=1)
        if self.prior is None:
            loss = self.like(fX_batch, labels)
        else:
            loss = (
                self.like(fX_batch, labels) + self.prior(self.net) / self.num_datapoints
            )
        self.opt.zero_grad()
        loss.backwards()
        self.opt.step()
        with torch.no_grad():
            log_dict["training_loss"] = loss.detach().cpu().numpy()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["training_acc"] = (
                (predicted_class == target_class).float().mean().detach().cpu().numpy()
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
        self.net.eval()
        with torch.no_grad():
            log_dict = {}

            B, T, s_dim = states.size()
            _, _, a_dim = actions.size()
            X_batch_1 = torch.concatenate([states, actions], dim=-1).reshape(
                -1, s_dim + a_dim
            )
            X_batch_2 = torch.concatenate([states_2, actions_2], dim=-1).reshape(
                -1, s_dim + a_dim
            )

            pred_1 = self.net(X_batch_1).reshape(B, T) * attn_mask
            pred_2 = self.net(X_batch_2).reshape(B, T) * attn_mask_2

            sum_pred_1 = torch.nansum(pred_1, dim=1).reshape(-1, 1)
            sum_pred_2 = torch.nansum(pred_2, dim=1).reshape(-1, 1)
            fX_batch = torch.concatenate([sum_pred_1, sum_pred_2], dim=1)
            if self.prior is None:
                loss = self.like(fX_batch, labels)
            else:
                loss = (
                    self.like(fX_batch, labels)
                    + self.prior(self.net) / self.num_datapoints
                )

            log_dict["eval_loss"] = loss.detach().cpu().numpy()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["eval_acc"] = (
                (predicted_class == target_class).float().mean().detach().cpu().numpy()
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
        self.net.train()
        log_dict = {}

        B, T, _ = states.size()
        
        trans_pred_1, _ = self.net(
            states,
            actions,
            timesteps,
            attn_mask,
        )
        trans_pred_2, _ = self.net(
            states_2,
            actions_2,
            timesteps_2,
            attn_mask_2,
        )

        trans_pred_1 = trans_pred_1["weighted_sum"]
        trans_pred_2 = trans_pred_2["weighted_sum"]
        
        sum_pred_1 = torch.mean(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
        sum_pred_2 = torch.mean(trans_pred_2.reshape(B, T), dim=1).reshape(-1, 1)
        fX_batch = torch.concatenate([sum_pred_1, sum_pred_2], dim=1)
        
        loss = self.like(fX_batch, labels)

        self.opt.zero_grad()
        loss.backwards()
        self.opt.step()
        with torch.no_grad():
            log_dict["training_loss"] = loss.detach().cpu().numpy()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["training_acc"] = (
                (predicted_class == target_class).float().mean().detach().cpu().numpy()
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
        self.net.eval()
        with torch.no_grad():
            log_dict = {}

            B, T, _ = states.size()
            
            trans_pred_1, _ = self.net(
                states,
                actions,
                timesteps,
                attn_mask,
            )
            trans_pred_2, _ = self.net(
                states_2,
                actions_2,
                timesteps_2,
                attn_mask_2,
            )
    
            trans_pred_1 = trans_pred_1["weighted_sum"]
            trans_pred_2 = trans_pred_2["weighted_sum"]
            
            sum_pred_1 = torch.mean(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            sum_pred_2 = torch.mean(trans_pred_2.reshape(B, T), dim=1).reshape(-1, 1)
            fX_batch = torch.concatenate([sum_pred_1, sum_pred_2], dim=1)
            
            loss = self.like(fX_batch, labels)

            log_dict["eval_loss"] = loss.detach().cpu().numpy()
            predicted_class = torch.argmax(fX_batch, dim=1)
            target_class = torch.argmax(labels, dim=1)
            log_dict["eval_acc"] = (
                (predicted_class == target_class).float().mean().detach().cpu().numpy()
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
