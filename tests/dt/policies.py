import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import gym

import jax
import haiku as hk
import jax.numpy as jnp

from sb3_jax.dt import policies

env = gym.make('HalfCheetah-v2')
obs_space, act_space = env.observation_space, env.action_space

dt_policy = policies.DTPolicy(
    observation_space=obs_space,
    action_space=act_space,
    lr_schedule=3e-4,
    max_length=20,
    max_ep_length=1000,
    hidden_size=128,
    n_layer=3,
    n_head=1,
    n_inner=4*128,
    activation_function='relu',
    n_positions=1024,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
)


