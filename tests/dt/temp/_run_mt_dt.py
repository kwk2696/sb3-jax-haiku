import time
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import gym
import pickle
import argparse
from mujoco_control_envs.mujoco_control_envs import HalfCheetahVelEnv

import numpy as np

from stable_baselines3.common import env_util, vec_env
from sb3_jax import DT
from sb3_jax.dt.policies import MlpPolicy
from sb3_jax.common.buffers import MTTrajectoryBuffer
from sb3_jax.common.evaluation import evaluate_mt_traj_policy

parser = argparse.ArgumentParser()
parser.add_argument("--use_id", action="store_true")
parser.add_argument("--use_prompt", action="store_true")

parser.add_argument("--tag", type=str, default="t")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--env_idx", type=int, default=15) # env idx
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()

max_ep_length = 200
env_target = 0
scale = 500.

eval_episodes = args.n
eval_idx = args.env_idx


if not args.eval: # Training
    envs = []
    #env_indxes = [0, 3, 6, 9, 12]
    env_indxes = [0, 1, 3, 4, 5, 6, 8, 9, \
            10, 11, 12, 13, 14 ,16, 17, 18, 19, \
            20, 21, 22, 24, 25, 27, 28, 29, \
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,\
        ]
    for idx in env_indxes:
        with open(f"../data/config/cheetah_vel/config_cheetah_vel_task{idx}.pkl", 'rb') as f:
            task_info = pickle.load(f)
            print(task_info[0])
        env = HalfCheetahVelEnv(task_info, include_goal=False)
        envs.append(env)

    obs_space, act_space = env.observation_space, env.action_space


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

    # Make DT
    dt = DT(
        policy=MlpPolicy,
        env=env,
        replay_buffer=_buff,
        learning_rate=1e-4,
        batch_size=8,
        gradient_steps=20,
        verbose=1,
        policy_kwargs=dict(
            num_tasks=len(envs),
            use_id=args.use_id,
            use_prompt=args.use_prompt,
            prompt_size=5,
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
        wandb_log=f'halfcheetah-vel-{args.tag}',
    )

    dt.learn(total_timesteps=10_000, log_interval=100)
    dt.save(path=f'../data/cheetah_vel/pretrained_dt/{args.tag}')

else: # Evaluation

    envs = []
    env_indxes = [eval_idx]
    for idx in env_indxes:
        with open(f"../data/config/cheetah_vel/config_cheetah_vel_task{idx}.pkl", 'rb') as f:
            task_info = pickle.load(f)
            print(task_info[0])
        env = HalfCheetahVelEnv(task_info, include_goal=False)
        envs.append(env)

    obs_space, act_space = env.observation_space, env.action_space


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
        policy=MlpPolicy,
        env=env,
        replay_buffer=_buff,
        learning_rate=1e-4,
        batch_size=8,
        verbose=1,
        policy_kwargs=dict(
            num_tasks=len(envs),
            use_id=args.use_id,
            use_prompt=args.use_prompt, 
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
    _dt = _dt.load(path=f'../data/cheetah_vel/pretrained_dt/{args.tag}')
    
    # Set task id for evaluation!
    _dt.policy._task_id = args.task_id
    #if use_prompt:
        #print(_dt.policy.params)
    print(f"Model Evaluation at env {eval_idx}")
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
    print(f"Load Learning: {np.mean(mean_reward)}")
