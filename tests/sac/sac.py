import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.32'

import gym
import numpy as np

from stable_baselines3.common import env_util, vec_env
from sb3_jax.sac import SAC
from sb3_jax.common.evaluation import evaluate_policy
from sb3_jax.common.norm_layers import RunningNormLayer


env = env_util.make_vec_env('HalfCheetah-v3')
env = vec_env.VecNormalize(env, norm_obs=True, norm_reward=True)

sac = SAC(
    policy='MlpPolicy',
    env=env,
    learning_rate=3e-4,
    batch_size=256,
    policy_kwargs=dict(
        #normalization_class=RunningNormLayer,
        net_arch=dict(pi=[256,256], qf=[256,256])
    ),
    verbose=True,
)

# check entropy coefficient
log_ent_coef = sac.log_ent_coef(sac.ent_coef_params)
print(f"Log Entropy Coef: {log_ent_coef}")

sac.learn(total_timesteps=100_000, log_interval=1)

env.norm_reward = False
# Evaluate Policy After Learning
mean_reward, _ = evaluate_policy(sac, env, n_eval_episodes=10)
print(f"After Learning: {mean_reward}")
sac.save(path='../model/sac')

# Load Policy
_sac = SAC.load(path='../model/sac')
mean_reward, _ = evaluate_policy(_sac, env, n_eval_episodes=10)
print(f"Load Learning: {mean_reward}")
