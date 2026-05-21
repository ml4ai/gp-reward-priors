import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation_fns import *


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
    ):
        super(GPT2MLP, self).__init__()
        self.in_linear = nn.Linear(embd_dim, intermediate_dim)
        self.out_linear = nn.Linear(intermediate_dim, embd_dim)
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
    ):
        super(GPT2SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads

        self.in_linear = nn.Linear(embd_dim, 3 * embd_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out_linear = nn.Linear(embd_dim, embd_dim)
        self.resid_dropout = nn.Dropout(resid_dropout)

        # Additive causal bias: 0 where attention is allowed, -inf where masked.
        # Registered as a buffer so it moves with the model on .to(device) calls.
        self.register_buffer(
            "causal_bias",
            torch.triu(
                torch.full((max_pos, max_pos), float("-inf")), diagonal=1
            ).view(1, 1, max_pos, max_pos),
        )

    def forward(self, x, attn_mask):
        x = self.in_linear(x)

        query, key, value = torch.chunk(x, 3, dim=2)

        query = split_heads(query, self.num_heads, self.head_dim)
        value = split_heads(value, self.num_heads, self.head_dim)
        key = split_heads(key, self.num_heads, self.head_dim)

        query_len, key_len = query.shape[-2], key.shape[-2]
        # Slice the pre-computed causal bias to the current sequence lengths and
        # add the additive padding mask (already shaped [B, 1, 1, key_len]).
        combined_mask = self.causal_bias[:, :, key_len - query_len : key_len, :key_len]
        if attn_mask is not None:
            combined_mask = combined_mask + attn_mask

        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=combined_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        out = merge_heads(out, self.num_heads, self.head_dim)

        out = self.out_linear(out)

        out = self.resid_dropout(out)
        return out, None


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
    ):
        super(GPT2Block, self).__init__()
        self.layer_norm_0 = nn.LayerNorm(embd_dim, eps)
        self.attention = GPT2SelfAttention(
            embd_dim=embd_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            max_pos=max_pos,
        )
        self.layer_norm_1 = nn.LayerNorm(embd_dim, eps)
        self.mlp = GPT2MLP(
            embd_dim=embd_dim,
            intermediate_dim=intermediate_dim,
            resid_dropout=resid_dropout,
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
    ):
        super(PT, self).__init__()
        self.embd_dim = embd_dim
        self.pref_attn_embd_dim = pref_attn_embd_dim

        self.state_linear = nn.Linear(state_dim, embd_dim)
        self.action_linear = nn.Linear(action_dim, embd_dim)
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
        )
        self.pref_linear = nn.Linear(embd_dim, 2 * pref_attn_embd_dim + 1)
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

        # Preference attention is full (non-causal); only the padding mask applies.
        new_attn_mask = get_attention_mask(attn_mask, batch_size)
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=new_attn_mask,
            dropout_p=0.0,  # self.attn_dropout already has p=0.0
        )
        attn_weights_list.append(None)

        output = merge_heads(out, num_heads, 1)

        return {"weighted_sum": output, "value": value}, attn_weights_list
