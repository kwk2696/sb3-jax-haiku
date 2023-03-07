"""JAX Haiku GPT-2 model"""
import warnings
from functools import partial
from typing import Any, Optional, Tuple, Union, Dict, Callable

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from jax import lax
from jax import random
from jax import nn

from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPastAndCrossAttentions

"""Activation Functions"""
@jax.jit
def NewGELUActivation(x: Any) -> Any:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


ACT2FN = {
    "gelu_new": NewGELUActivation,
    "relu": nn.relu, 
}

"""Util Functions"""
def has_state(state: list, col: str, name: str) -> bool:
    for s in state: 
        if f'{col}/{name}' == s: return True
    return False

def combine_masks(*masks: Optional[Any], dtype: jnp.dtype = jnp.float32) -> Any:
    mask_list = [m for m in masks if m is not None]
    if not mask_list:
        return None
    assert all(map(lambda x: x.ndim == mask_list[0].ndim, mask_list)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, mask_list))}')
    mask, *other_masks = mask_list
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)

"""Weight Init Functions"""
def init_conv1d(stddev: float = 0.02) -> Dict[str, hk.initializers.Initializer]:
    
    return {"w_init": hk.initializers.RandomNormal(stddev=stddev),
            "b_init": hk.initializers.Constant(constant=0.)}

def init_embed(stddev: float = 0.02) -> Dict[str, hk.initializers.Initializer]:
    return {"w_init": hk.initializers.RandomNormal(stddev=stddev)}


def make_attention_mask(
    query_input: Any, 
    key_input: Any, 
    pairwise_fn: Callable[..., Any] = jnp.multiply,
    extra_batch_dims: int = 0,
    dtype: jnp.dtype = jnp.float32
):
    mask = pairwise_fn(jnp.expand_dims(query_input, axis=-1),
                    jnp.expand_dims(key_input, axis=-2))
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask.astype(dtype)


def make_causal_mask(x: Any, extra_batch_dims: int = 0, dtype: jnp.dtype = jnp.float32) -> Any:
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(idxs, idxs, jnp.greater_equal,
                            extra_batch_dims=extra_batch_dims, dtype=dtype)


class GPT2Attention(hk.Module):

    def __init__(
        self, 
        config: GPT2Config, 
        dtype: jnp.dtype = jnp.float32, 
        causal: bool = True,
        is_cross_attention: bool = False,
    ):
        super(GPT2Attention, self).__init__()
        self.config = config
        self.dtype = dtype
        self.causal = causal
    
        self.embed_dim = self.config.hidden_size
        self.num_heads = 1 #self.config.n_head
        self.head_dim = self.embed_dim // self.num_heads
             
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool"
            )
        
    def __call__(
        self,
        hidden_states,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        if not is_cross_attention:
            qkv_out = hk.Conv1D(3 * self.embed_dim, 1, name='c_attn', **init_conv1d())(hidden_states)
            query, key, value = jnp.split(qkv_out, 3, axis=2)
        else:
            q_out = hk.Conv1D(self.embed_dim, 1, name='q_attn', **init_conv1d())(hidden_states)
            (query,) = jnp.split(q_out, 1, axis=2)
            kv_out = hk.Conv1D(2 * self.embed_dim, 1, name='c_attn', **init_conv1d())(hidden_states)
            key, value = jnp.split(kv_out, 2, axis=2) 
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        query_length, key_length = query.shape[1], key.shape[1]

        if self.causal:
            if has_state(self.state_dict().keys(), self.name, "cached_key"):
                mask_shift = hk.get_state("cache_index")
                max_decoder_length = get_state("cached_key").shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # combine masks if needed
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        dropout_rng = hk.next_rng_key()
        
        if self.causal and (has_state(self.state_dict().keys(), self.name, "cached_key") or init_cache):
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)
        
        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype)
            )
        else:
            attention_bias = None

        # usual dot product attention
        attn_weights = self._dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
        )
        
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = hk.Conv1D(self.embed_dim, 1, name='c_proj', **init_conv1d())(attn_output)
        attn_output = hk.dropout(hk.next_rng_key(), self.config.resid_pdrop, attn_output)
        
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # detect if we're initializing by absence of existing cache data
        is_initialized = has_state(self.state_dict().keys(), self.name, "cached_key")
        cached_key = hk.get_state("cached_key", key.shape, key.dtype, init=jnp.zeros)
        cached_value = hk.get_state("cached_value", value.shape, value.dtype, init=jnp.zeros)
        cached_index = hk.get_state("cache_index", [], jpn.int32, init=jnp.zeros)

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key, key, indices)
            value = lax.dynamic_update_slice(cached_value, value, indices)
            hk.set_state("cached_key", key)
            hk.set_state("cached_value", value)
            num_updated_cache_vectors = query.shape[1]
            hk.set_state("cache_index", cache_index + num_updated_cache_vectors)
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def _dot_product_attention_weights(
        self,
        query: Any,
        key: Any,
        bias: Optional[Any] = None,
        mask: Optional[Any] = None,
        broadcast_dropout: bool = True,
        dropout_rng: Optional[Any] = None,
        dropout_rate: float = 0.,
        dtype: Optional[Any] = None,
        deterministic: bool = False,
    ):
        assert query.ndim == key.ndim, 'q, k must have same rank.'
        assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
        assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        # calculate attention matrix
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(dtype)
        # attn weight shape is (batch ..., num_heads, q_length, kv_length)
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
        
        # apply attention bias: masking, dropout, proximity bias, etc. 
        if bias is not None:
            attn_weights = attn_weights + bias
        # apply attention mask
        if mask is not None:
            big_neg = jnp.finfo(dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        
        # normalize the attention weights
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
        
        # apply attention dropout
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout: 
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = (keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier
        return attn_weights


class GPT2MLP(hk.Module):
    def __init__(
        self,
        config, 
        intermediate_size: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        super(GPT2MLP, self).__init__()
        self.config = config 

        self.intermediate_size = intermediate_size
        self.embed_dim = self.config.hidden_size

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True
    ):
        hidden_states = hk.Conv1D(self.intermediate_size, 1, name='c_fc', **init_conv1d())(hidden_states)
        hidden_states = ACT2FN[self.config.activation_function](hidden_states)
        hidden_states = hk.Conv1D(self.embed_dim, 1, name='c_proj', **init_conv1d())(hidden_states)
        hidden_states = hk.dropout(hk.next_rng_key(), self.config.resid_pdrop, hidden_states)
        return hidden_states


class GPT2Block(hk.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.float32,
    ):
        super(GPT2Block, self).__init__()
        self.config = config
        self.dtype = dtype
        
        self.hidden_size = self.config.hidden_size
        self.inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * self.hidden_size

    def __call__(
        self,
        hidden_states,
        attention_mask = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = hk.LayerNorm(-1, create_scale=True, create_offset=True, eps=self.config.layer_norm_epsilon, name='ln_1')(hidden_states)

        attn_outputs = GPT2Attention(self.config, dtype=self.dtype)(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = attn_outputs[0] # output_attn: a, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            # add one self-attention black for cross-attention
            residual = hidden_states
            hidden_states = hk.LayerNorm(-1, create_scale=True, create_offset=True, eps=self.config.layer_norm_epsilon, name='ln_cross_attn')(hidden_states)
            cross_attn_outputs = GPT2Attention(self.config, dtype=self.dtype, causal=False, is_cross_attention=True)(
                hidden_states,
                key_vaule_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                output_attention=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output 
            outputs = outputs + cross_attn_outputs[1:] # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = hk.LayerNorm(-1, create_scale=True, create_offset=True, eps=self.config.layer_norm_epsilon, name='ln_2')(hidden_states)
        feed_forward_hidden_states = GPT2MLP(self.config, self.inner_dim, dtype=self.dtype)(hidden_states, deterministic=deterministic)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + outputs

        return outputs
    

class GPT2BlockCollection(hk.Module):
    def __init__(
        self,
        config: GPT2Config,
        dtype: jnp.dtype = jnp.float32,
    ):
        super(GPT2BlockCollection, self).__init__()
        self.config = config
        self.dtype = dtype 

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True, 
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for i in range(self.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = GPT2Block(self.config, dtype=self.dtype)(
                hidden_states,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # this contains possible `None` values - `GPT2Module` will filter them out 
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)
        
        return outputs


class GPT2Module(hk.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.float32,
    ):
        super(GPT2Module, self).__init__()
        self.config = config
        self.dtype = dtype 
        
        self.embed_dim = self.config.hidden_size

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] =  None,
        input_embeds: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # input should be one of ...
        # 1. without embedding -> in this case we need ids
        # 2. with embeddings -> in this case we only use inputs_embeds
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            input_embeds = hk.Embed(
                self.config.vocab_size,
                self.embed_dim,
                name="wte",
                **init_embed(stddev=self.config.initializer_range)
            )(input_ids.astype("i4"))
     
            position_embeds = hk.Embed(
                self.config.max_position_embeddings,
                self.embed_dim,
                name="wpe",
                **init_embed(stddev=self.config.initializer_range)
            )(position_ids.astype("i4"))
            hidden_states = input_embeds + position_embeds
        hk.dropout(hk.next_rng_key(), self.config.embd_pdrop, hidden_states),

        outputs = GPT2BlockCollection(self.config, dtype=self.dtype)(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = hk.LayerNorm(-1, create_scale=True, create_offset=True, eps=self.config.layer_norm_epsilon, name='ln_f')(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
        )

GPT2Model=GPT2Module
