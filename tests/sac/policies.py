import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.32'

import gym
import numpy as np
import torch as th
from torch.distributions import Normal, Categorical

from stable_baselines3.common.distributions import sum_independent_dims
from sb3_jax.sac import policies
from sb3_jax.common import utils


env = gym.make('HalfCheetah-v2')
obs_space, act_space = env.observation_space, env.action_space
sac_actor = policies.Actor(
                observation_space=obs_space,
                action_space=act_space,
                log_std_init=-2,
                net_arch=[64,64]
            )

obs = utils.get_dummy_obs(obs_space)
action_det, _, _ = sac_actor.predict(obs, deterministic=True)
print(f"Deterministic Action: {action_det}")
action_stc, _, _ = sac_actor.predict(obs, deterministic=False)
print(f"Stochastic Action: {action_stc}")
log_prob = sac_actor.action_log_prob(obs, action_stc)
print(f"Log probability: {log_prob}")

mean_actions, log_std = sac_actor._actor(obs, sac_actor.params)
mean_actions, log_std = th.Tensor(np.array(mean_actions)), th.Tensor(np.array(log_std)).exp()
torch_dist = Normal(mean_actions, log_std)
action_stc = th.Tensor(action_stc)
torch_log_prob = sum_independent_dims(torch_dist.log_prob(action_stc))
print(f"Torch Log Probability: {torch_log_prob}")
