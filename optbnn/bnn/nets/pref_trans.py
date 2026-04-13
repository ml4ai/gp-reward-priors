import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation_fns import *
from ..layers.linear import Linear


class GPT2MLP(nnx.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        intermediate_dim: int = 256,
        resid_dropout: float = 0.1,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.in_linear = nnx.Linear(embd_dim, intermediate_dim, rngs=rngs)
        self.out_linear = nnx.Linear(intermediate_dim, embd_dim, rngs=rngs)
        self.resid_dropout = nnx.Dropout(resid_dropout, rngs=rngs)

    def __call__(self, x, training=False):
        x = self.in_linear(x)
        x = nnx.relu(x)
        x = self.out_linear(x)
        x = self.resid_dropout(x, deterministic=not training)
        return x


class GPT2SelfAttention(nnx.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        max_pos: int = 1024,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads
        self.max_pos = max_pos

        self.in_linear = nnx.Linear(embd_dim, 3 * embd_dim, rngs=rngs)
        self.attn_dropout = nnx.Dropout(attn_dropout, rngs=rngs)

        self.out_linear = nnx.Linear(embd_dim, embd_dim, rngs=rngs)
        self.resid_dropout = nnx.Dropout(resid_dropout, rngs=rngs)

    def __call__(self, x, attn_mask, training=False):
        x = self.in_linear(x)

        query, key, value = jnp.split(x, 3, axis=2)

        query = ops.split_heads(query, self.num_heads, self.head_dim)
        value = ops.split_heads(value, self.num_heads, self.head_dim)
        key = ops.split_heads(key, self.num_heads, self.head_dim)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.tril(jnp.ones((1, 1, self.max_pos, self.max_pos)))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.astype(bool)

        out, _attn_weights = ops.attention(
            query,
            key,
            value,
            casual_mask,
            -1e4,
            self.attn_dropout,
            True,
            training,
            attn_mask,
        )
        out = ops.merge_heads(out, self.num_heads, self.head_dim)

        out = self.out_linear(out)

        out = self.resid_dropout(out, deterministic=not training)
        return out, _attn_weights


class GPT2Block(nnx.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        intermediate_dim: int = 256,
        max_pos: int = 1024,
        eps: float = 1e-05,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.layer_norm_0 = nnx.LayerNorm(embd_dim, epsilon=eps, rngs=rngs)
        self.attention = GPT2SelfAttention(
            embd_dim=embd_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            max_pos=max_pos,
            rngs=rngs,
        )
        self.layer_norm_1 = nnx.LayerNorm(embd_dim, epsilon=eps, rngs=rngs)
        self.mlp = GPT2MLP(
            embd_dim=embd_dim,
            intermediate_dim=intermediate_dim,
            resid_dropout=resid_dropout,
            rngs=rngs,
        )

    def __call__(self, x, attn_mask, training=False):
        residual = x
        x = self.layer_norm_0(x)
        x, _attn_weights = self.attention(x, attn_mask, training)
        x += residual
        residual = x
        x = self.layer_norm_1(x)
        x = self.mlp(x, training)
        x += residual
        return x, _attn_weights


class GPT2Model(nnx.Module):
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
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.dropout = nnx.Dropout(embd_dropout, rngs=rngs)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                GPT2Block(
                    embd_dim=embd_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    intermediate_dim=intermediate_dim,
                    max_pos=max_pos,
                    eps=eps,
                    rngs=rngs,
                )
            )
        self.layer_norm = nnx.LayerNorm(embd_dim, epsilon=eps, rngs=rngs)

    def __call__(self, input_embds, attn_mask, training=False):
        x = self.dropout(input_embds, deterministic=not training)
        batch_size = input_embds.shape[0]
        attn_weights_list = []
        attn_mask = ops.get_attention_mask(attn_mask, batch_size)
        for m in self.layers:
            x, attn_weights = m(x, attn_mask, training)
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
            rngs=rngs,
        )
        self.pref_linear = Linear(
            embd_dim, 2 * pref_attn_embd_dim + 1, scaled_variance=scaled_variance
        )
        self.attn_dropout = nn.Dropout(0.0)

    def __call__(self, states, actions, timesteps, attn_mask, training=False):
        batch_size, seq_length, _ = states.size()

        embd_states = self.state_linear(states)
        embd_actions = self.action_linear(actions)

        embd_timesteps = self.timestep_embed(timesteps)

        embd_states = embd_states + embd_timesteps
        embd_actions = embd_actions + embd_timesteps

        stacked_inputs = (
            torch.stack([embd_states, embd_actions], dim=1)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.embd_dim)
        )

        stacked_inputs = self.stacked_layer_norm(stacked_inputs)

        stacked_attn_mask = (
            jnp.stack([attn_mask, attn_mask], axis=1)
            .transpose(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        transformer_outputs = self.gpt(
            input_embds=stacked_inputs, attn_mask=stacked_attn_mask, training=training
        )

        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        x = x.reshape(batch_size, seq_length, 2, self.embd_dim).transpose(0, 2, 1, 3)
        hidden_output = x[:, 1]

        x = self.pref_linear(hidden_output)

        num_heads = 1

        query, key, value = jnp.split(
            x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], axis=2
        )
        query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
        key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
        value = ops.split_heads(value, num_heads, 1)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.ones((1, 1, seq_length, seq_length))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.astype(bool)

        new_attn_mask = ops.get_attention_mask(attn_mask, batch_size)
        out, last_attn_weights = ops.attention(
            query,
            key,
            value,
            casual_mask,
            -1e-4,
            self.attn_dropout,
            scale_attn_weights=True,
            training=training,
            attn_mask=new_attn_mask,
        )
        attn_weights_list.append(last_attn_weights)

        output = ops.merge_heads(out, num_heads, 1)

        return {"weighted_sum": output, "value": value}, attn_weights_list
