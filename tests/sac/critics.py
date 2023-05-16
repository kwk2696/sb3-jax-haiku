import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.32'

import gym
import numpy as np

from sb3_jax.common.policies import ContinuousCritic
from sb3_jax.common import utils


env = gym.make('HalfCheetah-v2')
obs_space, act_space = env.observation_space, env.action_space
critic = ContinuousCritic(
            observation_space=obs_space,
            action_space=act_space,
            net_arch=[64,64],
        )


obs = utils.get_dummy_obs(obs_space)
act = utils.get_dummy_act(act_space)

qvals = critic.forward(obs, act)
print(f"Critics Output: {qvals}")
qval1 = critic.q1_forward(obs, act)
print(f"Critic1 Output: {qval1}")
