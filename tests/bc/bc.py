import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
import gym

from stable_baselines3.common import env_util, vec_env
from sb3_jax import BC
from sb3_jax.bc.policies import MlpPolicy
from sb3_jax.common.evaluation import evaluate_policy
from sb3_jax.common.norm_layers import RunningNormLayer
from sb3_jax.common.buffers import OfflineBuffer

import moirl_jax 
moirl_jax.register_mo_mujoco_envs()


env_train = env_util.make_vec_env('MO-Swimmer-v2')
env_train = vec_env.VecNormalize(env_train, norm_obs=False, norm_reward=True)
env_eval = env_util.make_vec_env('MO-Swimmer-v2')

# Load Buffer
buff = OfflineBuffer(
    buffer_size=2_500_000,
    observation_space=env_train.observation_space,
    action_space=env_train.action_space,
)
buff = buff.load(path='datasets/offline_buffer.pkl')
#print(buff.observations[0][0])

# Make BC
bc = BC(
    policy=MlpPolicy,
    env=env_train,
    replay_buffer=buff,
    learning_rate=3e-4,
    batch_size=64,
    verbose=1,
    policy_kwargs=dict(
        log_std_init=0.,
        net_arch=[64, 64],
        normalization_class=RunningNormLayer
    )
)

mean_reward, _ = evaluate_policy(bc, env_eval, n_eval_episodes=100)
print(f"Before Learning: {mean_reward}")
bc.learn(total_timesteps=100_000, log_interval=1000)
mean_reward, _ = evaluate_policy(bc, env_eval, n_eval_episodes=100)
print(f"After Learning: {mean_reward}")
bc.save(path='./model/bc')

# Loading Model
_bc = BC(
    policy=MlpPolicy,
    env=env_train,
    replay_buffer=buff,
    policy_kwargs=dict(
        log_std_init=-2,
        net_arch=[256, 256],
        normalization_class=RunningNormLayer
    )
)
_bc = _bc.load(path='./model/bc', env=env_train) # should give env for loading
mean_reward, _ = evaluate_policy(_bc, env_eval, n_eval_episodes=100)
print(f"Load Learning: {mean_reward}")
