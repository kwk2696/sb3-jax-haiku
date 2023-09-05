import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import gym

import jax
import haiku as hk
import jax.numpy as jnp

import transformers
from sb3_jax.dt.gpt2 import GPT2Model


config = transformers.GPT2Config(vocab_size=1, n_embed=16, n_layer=1)
def fn_transformer(input_ids, attention_mask, position_ids):
    transformer = GPT2Model(config)
    return transformer(
        input_ids,
        attention_mask,
        position_ids,
    )

rng = hk.PRNGSequence(0)
input_shape = (1,1)
input_ids = jnp.zeros(input_shape, dtype="i4")
attention_mask = jnp.ones_like(input_ids)
position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)

# without_state: wrapper function ingores state in/out
_params, transformer = hk.transform_with_state(fn_transformer)
params, state = _params(next(rng), input_ids, attention_mask, position_ids)

out, _ = transformer(params, state, next(rng), input_ids, attention_mask, position_ids)
print("Last hidden states:", out['last_hidden_state'])
