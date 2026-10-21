"""Non-Markovian reward model (NMR): the LSTM baseline of the Preference Transformer
paper (Kim et al. 2023, after Early et al. 2022), ported to PyTorch from the PT
repository's JAX/Flax `LSTMRewardModel` (flaxmodels/lstm/lstm.py) and `NMR`
(JaxPref/NMR.py).  gp_reward-priors HANDOFF §14.2 item 1.

Architecture, as the reference (e = embd_dim):
    concat(state, action) -> Dense e -> Dense e/2 -> Dense e/2        (act + dropout each)
    -> unidirectional LSTM, hidden e/2, zero initial state
    -> concat(dense output, LSTM output) -> Dense e/2 -> Dense e/4 -> Dense e/4
                                                                       (act + dropout each)
    -> Dense 1: a reward for EVERY step, from that step's prefix.

Like the reference, `timesteps` and `attn_mask` are accepted and unused (an LSTM
reads order from the sequence itself; padding is zeros, left of the real steps
when labelling).  Departures, both shared with this codebase's MR and PT:
  - layer initialisation is PyTorch's default (Flax: lecun_normal kernels,
    orthogonal recurrent kernel); the LSTM has PyTorch's two bias vectors;
  - training pools the per-step rewards by the masked MEAN (bt_pool "mean", as
    every family here), not the reference's sum.
The forward output matches PT's interface -- {"weighted_sum": (B, T, 1),
"value": (B, 1, T, 1)} -- so PTTrainer trains it and the IQL sequence labeller
(iql_eval.qlearning_dataset_pt, reading value[:, 0, -1, 0]) labels with it,
exactly as the reference labels NMR with the PT labeller (label_mode "last").
"""

import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU,
               "sigmoid": nn.Sigmoid}


def _stack(dims_in, dims, activation, dropout):
    layers, d = [], dims_in
    for h in dims:
        layers += [nn.Linear(d, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers)


class NMR(nn.Module):
    def __init__(self, state_dim: int = 29, action_dim: int = 8, embd_dim: int = 256,
                 embd_dropout: float = 0.1, activations: str = "relu"):
        super().__init__()
        if embd_dim % 4:
            raise ValueError(f"embd_dim must be divisible by 4, got {embd_dim}")
        if activations not in ACTIVATIONS:
            raise ValueError(f"activations must be one of {sorted(ACTIVATIONS)}, got {activations!r}")
        e = embd_dim
        self.embd_dim = e
        self.pre = _stack(state_dim + action_dim, [e, e // 2, e // 2], activations, embd_dropout)
        self.lstm = nn.LSTM(e // 2, e // 2, num_layers=1, batch_first=True)
        self.post = _stack(e, [e // 2, e // 4, e // 4], activations, embd_dropout)
        self.out = nn.Linear(e // 4, 1)

    def forward(self, states, actions, timesteps=None, attn_mask=None):
        x = self.pre(torch.cat([states, actions], dim=-1))
        h, _ = self.lstm(x)                              # zero initial state
        r = self.out(self.post(torch.cat([x, h], dim=-1)))  # (B, T, 1)
        B, T, _ = r.shape
        return {"weighted_sum": r, "value": r.reshape(B, 1, T, 1)}, None


def n_params(state_dim, action_dim, embd_dim):
    return sum(p.numel() for p in NMR(state_dim, action_dim, embd_dim).parameters())
