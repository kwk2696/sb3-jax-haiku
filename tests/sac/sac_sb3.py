import os

import gym
import numpy as np

from stable_baselines3.common import env_util, vec_env
from stable_baselines3.sac import SAC


env = env_util.make_vec_env('HalfCheetah-v3')
env = vec_env.VecNormalize(env, norm_obs=True, norm_reward=True)

sac = SAC(
    policy='MlpPolicy',
    env=env,
    learning_rate=3e-4,
    batch_size=256,
    policy_kwargs=dict(
        net_arch=dict(pi=[256,256], qf=[256,256])
    ),
    verbose=True,
)

sac.learn(total_timesteps=1_000_000, log_interval=1)
