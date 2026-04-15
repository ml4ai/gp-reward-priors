import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation_fns import *
from ..layers.linear import Linear


def split_heads(x, num_heads, head_dim):
    """
    Splits embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
    """
    newshape = x.shape[:-1] + (num_heads, head_dim)
    x = x.reshape(newshape)
    if x.ndim == 5:
        # [batch, blocks, head, block_len, head_dim]
        return x.transpose(2, 3)
    elif x.ndim == 4:
        # [batch, head, seq_len, head_dim]
        return x.transpose(1, 2)
    else:
        raise ValueError(
            f"Input tensor should have rank 4 or 5, but has rank {x.ndim}."
        )


def get_attention_mask(attn_mask, batch_size):
    assert batch_size > 0, "batch_size should be > 0."
    attn_mask = attn_mask.reshape(batch_size, -1)
    attn_mask = attn_mask[:, None, None, ...]
    attn_mask = (1.0 - attn_mask) * -10000.0
    return attn_mask


def attention(
    query,
    key,
    value,
    casual_mask,
    masked_bias,
    dropout,
    scale_attn_weights,
    attn_mask=None,
):
    """
    Computes Dot-Product Attention for the given query, key and value.

    Args:
        query (tensor): Query, shape [B, num_heads, seq_len, embd_dim].
        key (tensor): Key, shape [B, num_heads, seq_len, embd_dim].
        value (tensor): Value, shape [B, num_heads, seq_len, embd_dim].
        casual_mask (tensor): Mask to ensure that attention is only applied to the left of the input sequence,
                              shape [1, 1, key_len - query_len :key_len, :key_len].
        masked_bias (float): Value to insert for masked part of the sequence.
        dropout (nn.Dropout): Dropout module that is applied to the attention output.
        scale_attn_weights (bool): If True, scale the attention weights.
        training (bool): Training mode.
        attn_mask (tensor): Mask to avoid performing attention on padded tokens indices, shape [B, seq_len].
        head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads,] or [num_layers, num_heads].
        feedback (tensor): external feedback with marked points.

    Returns:
        (tensor): Attention output, shape [B, num_heads, seq_len, embd_dim].
        (tensor): Attention weights, shape [B, num_heads, seq_len, seq_len].
        (tensor): KLD loss with external feedback, float.
    """
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if scale_attn_weights:
        attn_weights = attn_weights / (float(value.shape[-1]) ** 0.5)

    attn_weights = torch.where(casual_mask, attn_weights, masked_bias)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    _attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = _attn_weights.to(value.dtype)
    attn_weights = dropout(attn_weights)

    out = torch.matmul(attn_weights, value)
    return out, _attn_weights


def merge_heads(x, num_heads, head_dim):
    """
    Merge embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
    """
    if x.ndim == 5:
        x = x.transpose(2, 3)
    elif x.ndim == 4:
        x = x.transpose(1, 2)
    else:
        raise ValueError(
            f"Input tensor should have rank 4 or 5, but has rank {x.ndim}."
        )

    newshape = x.shape[:-2] + (num_heads * head_dim,)
    x = x.reshape(newshape)
    return x


class GPT2MLP(nn.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        intermediate_dim: int = 256,
        resid_dropout: float = 0.1,
        scaled_variance: bool = True,
    ):
        super(GPT2MLP, self).__init__()
        self.in_linear = Linear(
            embd_dim, intermediate_dim, scaled_variance=scaled_variance
        )
        self.out_linear = Linear(
            intermediate_dim, embd_dim, scaled_variance=scaled_variance
        )
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.in_linear(x)
        x = self.relu(x)
        x = self.out_linear(x)
        x = self.resid_dropout(x)
        return x


class GPT2SelfAttention(nn.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        max_pos: int = 1024,
        scaled_variance: bool = True,
    ):
        super(GPT2SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads
        self.max_pos = max_pos

        self.in_linear = Linear(embd_dim, 3 * embd_dim, scaled_variance=scaled_variance)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out_linear = Linear(embd_dim, embd_dim, scaled_variance=scaled_variance)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x, attn_mask):
        x = self.in_linear(x)

        query, key, value = torch.chunk(x, 3, dim=2)

        query = split_heads(query, self.num_heads, self.head_dim)
        value = split_heads(value, self.num_heads, self.head_dim)
        key = split_heads(key, self.num_heads, self.head_dim)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = torch.tril(torch.ones((1, 1, self.max_pos, self.max_pos)))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.to(torch.bool)

        out, _attn_weights = attention(
            query,
            key,
            value,
            casual_mask,
            -1e4,
            self.attn_dropout,
            True,
            attn_mask,
        )
        out = merge_heads(out, self.num_heads, self.head_dim)

        out = self.out_linear(out)

        out = self.resid_dropout(out)
        return out, _attn_weights


class GPT2Block(nn.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        intermediate_dim: int = 256,
        max_pos: int = 1024,
        eps: float = 1e-05,
        scaled_variance: bool = True,
    ):
        super(GPT2Block, self).__init__()
        self.layer_norm_0 = nn.LayerNorm(embd_dim, eps)
        self.attention = GPT2SelfAttention(
            embd_dim=embd_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            max_pos=max_pos,
            scaled_variance=scaled_variance,
        )
        self.layer_norm_1 = nn.LayerNorm(embd_dim, eps)
        self.mlp = GPT2MLP(
            embd_dim=embd_dim,
            intermediate_dim=intermediate_dim,
            resid_dropout=resid_dropout,
            scaled_variance=scaled_variance,
        )

    def forward(self, x, attn_mask):
        residual = x
        x = self.layer_norm_0(x)
        x, _attn_weights = self.attention(x, attn_mask)
        x += residual
        residual = x
        x = self.layer_norm_1(x)
        x = self.mlp(x)
        x += residual
        return x, _attn_weights


class GPT2Model(nn.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        intermediate_dim: int = 256,
        num_layers: int = 1,
        embd_dropout: float = 0.1,
        max_pos: int = 1024,
        eps: float = 1e-05,
        scaled_variance: bool = True,
    ):
        super(GPT2Model, self).__init__()
        self.dropout = nn.Dropout(embd_dropout)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GPT2Block(
                    embd_dim=embd_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    intermediate_dim=intermediate_dim,
                    max_pos=max_pos,
                    eps=eps,
                    scaled_variance=scaled_variance,
                )
            )
        self.layer_norm = nn.LayerNorm(embd_dim, eps)

    def forward(self, input_embds, attn_mask):
        x = self.dropout(input_embds)
        batch_size = input_embds.shape[0]
        attn_weights_list = []
        attn_mask = get_attention_mask(attn_mask, batch_size)
        for m in self.layers:
            x, attn_weights = m(x, attn_mask)
            attn_weights_list.append(attn_weights)
        x = self.layer_norm(x)
        return {
            "last_hidden_state": x,
            "attn_weights_list": attn_weights_list,
        }


class PT(nn.Module):
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 3,
        max_episode_steps: int = 1219,
        embd_dim: int = 64,
        pref_attn_embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        intermediate_dim: int = 256,
        num_layers: int = 1,
        embd_dropout: float = 0.1,
        max_pos: int = 1024,
        eps: float = 1e-05,
        scaled_variance: bool = True,
    ):
        super(PT, self).__init__()
        self.embd_dim = embd_dim
        self.pref_attn_embd_dim = pref_attn_embd_dim

        self.state_linear = Linear(state_dim, embd_dim, scaled_variance=scaled_variance)
        self.action_linear = Linear(
            action_dim, embd_dim, scaled_variance=scaled_variance
        )
        self.timestep_embed = nn.Embedding(max_episode_steps + 1, embd_dim)
        self.stacked_layer_norm = nn.LayerNorm(embd_dim, eps)
        self.gpt = GPT2Model(
            embd_dim=embd_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            embd_dropout=embd_dropout,
            max_pos=max_pos,
            eps=eps,
            scaled_variance=scaled_variance,
        )
        self.pref_linear = Linear(
            embd_dim, 2 * pref_attn_embd_dim + 1, scaled_variance=scaled_variance
        )
        self.attn_dropout = nn.Dropout(0.0)

    def forward(self, states, actions, timesteps, attn_mask):
        batch_size, seq_length, _ = states.size()

        embd_states = self.state_linear(states)
        embd_actions = self.action_linear(actions)

        embd_timesteps = self.timestep_embed(timesteps)

        embd_states = embd_states + embd_timesteps
        embd_actions = embd_actions + embd_timesteps

        stacked_inputs = (
            torch.stack([embd_states, embd_actions], dim=1)
            .transpose(1, 2)
            .reshape(batch_size, 2 * seq_length, self.embd_dim)
        )

        stacked_inputs = self.stacked_layer_norm(stacked_inputs)

        stacked_attn_mask = (
            torch.stack([attn_mask, attn_mask], dim=1)
            .transpose(1, 2)
            .reshape(batch_size, 2 * seq_length)
        )

        transformer_outputs = self.gpt(
            input_embds=stacked_inputs, attn_mask=stacked_attn_mask
        )

        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        x = x.reshape(batch_size, seq_length, 2, self.embd_dim).transpose(1, 2)
        hidden_output = x[:, 1]

        x = self.pref_linear(hidden_output)

        num_heads = 1

        query, key, value = torch.tensor_split(
            x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], dim=2
        )
        query = split_heads(query, num_heads, self.pref_attn_embd_dim)
        key = split_heads(key, num_heads, self.pref_attn_embd_dim)
        value = split_heads(value, num_heads, 1)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = torch.ones((1, 1, seq_length, seq_length))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.to(torch.bool)

        new_attn_mask = get_attention_mask(attn_mask, batch_size)
        out, last_attn_weights = attention(
            query,
            key,
            value,
            casual_mask,
            -1e-4,
            self.attn_dropout,
            scale_attn_weights=True,
            attn_mask=new_attn_mask,
        )
        attn_weights_list.append(last_attn_weights)

        output = merge_heads(out, num_heads, 1)

        return {"weighted_sum": output, "value": value}, attn_weights_list
