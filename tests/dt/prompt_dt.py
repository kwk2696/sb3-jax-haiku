import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import gym
import pickle
import argparse
from mujoco_control_envs.mujoco_control_envs import HalfCheetahVelEnv

import numpy as np
from stable_baselines3.common import env_util, vec_env
from sb3_jax import DT, PDT
from sb3_jax.dt.policies import DTPolicy, PDTPolicy
from sb3_jax.common.buffers import MTTrajectoryBuffer, TrajectoryBuffer
from sb3_jax.common.evaluation import evaluate_mt_traj_policy

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=15)
parser.add_argument("--n", type=int, default=100)
args = parser.parse_args()

max_ep_length = 200
env_target = 0
scale = 500.

eval_episodes = args.n
p_idx = args.idx

envs = []
env_indxes = [p_idx]
for idx in env_indxes:
    with open(f"../data/config/cheetah_vel/config_cheetah_vel_task{idx}.pkl", 'rb') as f:
        task_info = pickle.load(f)
        print(task_info[0])
    env = HalfCheetahVelEnv(task_info, include_goal=False)
    envs.append(env)
"""
target_vel = np.linspace(0.075, 3, 40)
for idx, vel in enumerate(target_vel):
    env = HalfCheetahVelEnv([{'velocity': vel}])
    if idx in env_indxes:
        envs.append(env)
"""
obs_space, act_space = envs[0].observation_space, envs[0].action_space


# Make Buffer
_buff = MTTrajectoryBuffer(
    max_length=20,
    max_ep_length=max_ep_length,
    scale=scale,
    observation_space=obs_space,
    action_space=act_space,
)
for idx in env_indxes:
    data_path = f'../data/cheetah_vel/cheetah_vel-{idx}-expert.pkl'
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    _buff.add_task(trajectories)

obs_means = [buff.obs_mean for buff in _buff.buffers]
obs_stds = [buff.obs_std for buff in _buff.buffers]
env_targets = [env_target/scale for buff in _buff.buffers]

# Loading Model
_dt = DT(
    policy=DTPolicy,
    env=env,
    replay_buffer=_buff,
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
_dt = _dt.load(path='../data/cheetah_vel/pretrained_dt', env=env)
_dt.replay_buffer = _buff
_dt.wandb_log=f'FDT-{p_idx}'
###############################################################



# PDT Fine Tuning #
data_path = f'../data/cheetah_vel/cheetah_vel-{p_idx}-expert.pkl'
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

print("Making Prompt DT ...")
pdt = PDT(
    policy=PDTPolicy,
    env=env,
    replay_buffer=buff,
    learning_rate=3e-4,
    batch_size=16,
    verbose=1,
    wandb_log=f'PDT-{p_idx}',
    policy_kwargs=dict(
        pretrained_policy=_dt.policy,
        prompt_size=1,
        max_grad_norm=.25,
        optimizer_kwargs=dict(
            weight_decay=1e-4
        )
    ),
)

# Sample prompt 
prompt = pdt.policy._prompt(pdt.policy.params)
#print("Init Prompt:", prompt)

# Test actor with prompt
rd = buff.sample(2)
(obs_preds, act_preds, rew_preds), _ = pdt.policy._actor(
    rd.observations, rd.actions, rd.rewards, rd.returns_to_go, rd.timesteps, rd.masks,
    deterministic=False, params=pdt.policy.params, 
    pretrained_params=pdt.policy.pretrained_policy.params, 
    pretrained_state=pdt.policy.pretrained_policy.state, 
    pretrained_rng=next(pdt.policy.pretrained_policy.rng)
)
print("Action Shape:", act_preds.shape)

# PDT Learning ...
#pdt.learn(total_timesteps=100_000, log_interval=100)
#pdt.save(path=f'../data/cheetah_vel/prompt_dt/task-{p_idx}/')

prompt = pdt.policy._prompt(pdt.policy.params)
#print("Trained Prompt:", prompt)
#exit()

# Evaluating ...
mean_reward, _ = evaluate_mt_traj_policy(
    model=pdt, 
    envs=envs, 
    n_eval_episodes=eval_episodes,
    max_ep_length=max_ep_length,
    deterministic=True,
    obs_means=obs_means,
    obs_stds=obs_stds,
    scale=scale,
    target_returns=env_targets,
    random_action=True,
    verbose=True
)
print(f"Random Policy: {np.mean(mean_reward)}")

mean_reward, _ = evaluate_mt_traj_policy(
    model=_dt, 
    envs=envs, 
    n_eval_episodes=eval_episodes,
    max_ep_length=max_ep_length,
    deterministic=True,
    obs_means=obs_means,
    obs_stds=obs_stds,
    scale=scale,
    target_returns=env_targets,
    verbose=True
)
bef_rew = np.mean(mean_reward)
print(f"Before Finetuning: {np.mean(mean_reward)}")

mean_reward, _ = evaluate_mt_traj_policy(
    model=pdt, 
    envs=envs, 
    n_eval_episodes=eval_episodes,
    max_ep_length=max_ep_length,
    deterministic=True,
    obs_means=obs_means,
    obs_stds=obs_stds,
    scale=scale,
    target_returns=env_targets,
    verbose=True
)
aft_rew = np.mean(mean_reward)
print(f"After Finetuning: {np.mean(mean_reward)}")

with open('../data/cheetah_vel/prompt_dt/test.txt', 'a') as f:
    f.write(f"Task idx -- {p_idx} --\n")
    f.write(f"Before Finetuning: {bef_rew}\n")
    f.write(f"After  Finetuning: {aft_rew}\n")
