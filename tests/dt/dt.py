import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import gym
import pickle
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv

from stable_baselines3.common import env_util, vec_env
from sb3_jax import DT
from sb3_jax.dt.policies import MlpPolicy
from sb3_jax.common.buffers import TrajectoryBuffer
from sb3_jax.common.evaluation import evaluate_traj_policy


max_ep_length = 200
env_target = 1500
scale = 1000.

env = HalfCheetahDirEnv([{'direction':1}], include_goal=False)
obs_space, act_space = env.observation_space, env.action_space

# Make Buffer
data_path = f'../data/cheetah_dir/cheetah_dir-0-expert.pkl'
with open(data_path, 'rb') as f:
    trajectories = pickle.load(f)

buff = TrajectoryBuffer(
    trajectories,
    max_length=20,
    max_ep_length=max_ep_length,
    scale=scale,
    observation_space=obs_space,
    action_space=act_space,
)


# Make DT
dt = DT(
    policy=MlpPolicy,
    env=env,
    replay_buffer=buff,
    learning_rate=1e-4,
    batch_size=256,
    verbose=1,
    policy_kwargs=dict(
        max_length=20,
        max_ep_length=max_ep_length,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4*128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=.1,
        attn_pdrop=.1,
        max_grad_norm=.25,
        optimizer_kwargs=dict(
            weight_decay=1e-4
        )
    ),
    wandb_log='halfcheetah-vel-test',
)

mean_reward, _ = evaluate_traj_policy(
    model=dt, 
    env=env, 
    n_eval_episodes=1,
    max_ep_length=max_ep_length,
    deterministic=True,
    obs_mean=buff.obs_mean,
    obs_std=buff.obs_std,
    scale=scale,
    target_return=env_target/scale,
)
print(f"Before Learning: {mean_reward}")
dt.learn(total_timesteps=1_000, log_interval=10)
mean_reward, _ = evaluate_traj_policy(
    model=dt, 
    env=env, 
    n_eval_episodes=1,
    max_ep_length=max_ep_length,
    deterministic=True,
    obs_mean=buff.obs_mean,
    obs_std=buff.obs_std,
    scale=scale,
    target_return=env_target/scale,
)
print(f"After Learning: {mean_reward}")

dt.save(path='./model')

# Loading Model
_dt = DT(
    policy=MlpPolicy,
    env=env,
    replay_buffer=buff,
    learning_rate=1e-4,
    batch_size=256,
    verbose=1,
    policy_kwargs=dict(
        max_length=20,
        max_ep_length=max_ep_length,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4*128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=.1,
        attn_pdrop=.1,
        max_grad_norm=.25,
        optimizer_kwargs=dict(
            weight_decay=1e-4
        )
    ),
)
print("Model Loading...")
_dt = _dt.load(path='./model')
mean_reward, _ = evaluate_traj_policy(
    model=dt, 
    env=env, 
    n_eval_episodes=1,
    max_ep_length=max_ep_length,
    deterministic=True,
    obs_mean=buff.obs_mean,
    obs_std=buff.obs_std,
    scale=scale,
    target_return=env_target/scale,
)
print(f"Load Learning: {mean_reward}")
