import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import argparse
import gym
import pickle
from mujoco_control_envs.mujoco_control_envs import HalfCheetahVelEnv

import numpy as np

from stable_baselines3.common import env_util, vec_env
from sb3_jax import DT
from sb3_jax.dt.policies import MlpPolicy
from sb3_jax.common.buffers import MTTrajectoryBuffer
from sb3_jax.common.evaluation import evaluate_traj_policy
from sb3_jax.common.utils import print_y, print_b


def main(args):
    max_ep_length = 200
    env_target = 0
    scale = 500.
    
    # Make environments
    train_env_indxes = [0, 1, 3, 4, 5, 6, 8, 9, \
            10, 11, 12, 13, 14 ,16, 17, 18, 19, \
            20, 21, 22, 24, 25, 27, 28, 29, \
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,\
        ]
    test_env_indxes = [2, 7, 15, 23, 26]

    train_envs, test_envs = [], []
    for idx in train_env_indxes:
        with open(f"./tests/data/config/cheetah_vel/config_cheetah_vel_task{idx}.pkl", 'rb') as f:
            task_info = pickle.load(f)
        env = HalfCheetahVelEnv(task_info, include_goal=False)
        train_envs.append(env)
    for idx in test_env_indxes:
        with open(f"./tests/data/config/cheetah_vel/config_cheetah_vel_task{idx}.pkl", 'rb') as f:
            task_info = pickle.load(f)
        env = HalfCheetahVelEnv(task_info, include_goal=False)
        test_envs.append(env)

    obs_space, act_space = env.observation_space, env.action_space

    prompt_length = 1 
    if args.type == 'fix' or args.type == 'soft':
        prompt_length = 5

    # Make Buffer
    train_buff = MTTrajectoryBuffer(
        max_length=20,
        max_ep_length=max_ep_length,
        scale=scale,
        prompt_length=prompt_length,
        observation_space=obs_space,
        action_space=act_space,
    )
    for idx in train_env_indxes:
        data_path = f'./tests/data/cheetah_vel/cheetah_vel-{idx}-expert.pkl'
        with open(data_path, 'rb') as f:
            trajectories = pickle.load(f)
        prompt_data_path = f'./tests/data/cheetah_vel/cheetah_vel-{idx}-prompt-expert.pkl'
        with open(prompt_data_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
        train_buff.add_task(trajectories, prompt_trajectories)

    train_obs_means = [buff.obs_mean for buff in train_buff.buffers]
    train_obs_stds = [buff.obs_std for buff in train_buff.buffers]
    train_env_targets = [env_target/scale for buff in train_buff.buffers]

    test_buff = MTTrajectoryBuffer(
        max_length=20,
        max_ep_length=max_ep_length,
        scale=scale,
        prompt_length=prompt_length,
        observation_space=obs_space,
        action_space=act_space,
    )
    for idx in test_env_indxes:
        data_path = f'./tests/data/cheetah_vel/cheetah_vel-{idx}-expert.pkl'
        with open(data_path, 'rb') as f:
            trajectories = pickle.load(f)
        prompt_data_path = f'./tests/data/cheetah_vel/cheetah_vel-{idx}-prompt-expert.pkl'
        with open(prompt_data_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
        test_buff.add_task(trajectories, prompt_trajectories)

    test_obs_means = [buff.obs_mean for buff in test_buff.buffers]
    test_obs_stds = [buff.obs_std for buff in test_buff.buffers]
    test_env_targets = [env_target/scale for buff in test_buff.buffers]

    if args.train:
        print_y("<< Training DT (multi-task) Model >>")
        # Make DT
        dt = DT(
            policy=MlpPolicy,
            env=env,
            replay_buffer=train_buff,
            learning_rate=1e-4,
            batch_size=8,
            gradient_steps=1,
            verbose=1,
            wandb_log=f'halfcheetah-vel-task{len(train_envs)}',
            policy_kwargs=dict(
                num_tasks=len(train_envs),
                prompt_type=args.type,
                prompt_length=prompt_length,
                max_length=20,
                max_ep_length=max_ep_length,
                hidden_size=128,
                n_layer=3,
                n_head=1,
                n_inner=4*128,
                activation_function='gelu_new',
                n_positions=1024,
                resid_pdrop=.1,
                attn_pdrop=.1,
                max_grad_norm=.25,
                optimizer_kwargs=dict(
                    weight_decay=1e-4
                )
            ),
        )

        dt.learn(total_timesteps=20_000, log_interval=100)
        for i, (env, idx) in enumerate(zip(train_envs, train_env_indxes)):
            if args.type == 'fix':
                o, a, r, d, rtg, t, m = train_buff.buffers[i].sample_prompt(1)
                dt.policy.set_prompt((o, a, r, rtg, t, m))
            elif args.type == 'id' or args.type == 'soft':
                dt.policy.set_task_id(i)
            
            mean_reward, std = evaluate_traj_policy(
                model=dt, 
                env=env, 
                n_eval_episodes=1,
                max_ep_length=max_ep_length,
                deterministic=True,
                obs_mean=train_obs_means[i],
                obs_std=train_obs_stds[i],
                scale=scale,
                target_return=train_env_targets[i],
            )
            print(f"After Learning Train Env{idx}: {mean_reward:.2f} +/- {std:.2f}")
        print_b("="*20)
        for i, (env, idx) in enumerate(zip(test_envs, test_env_indxes)):
            if args.type == 'fix':
                o, a, r, d, rtg, t, m = test_buff.buffers[i].sample_prompt(1)
                dt.policy.set_prompt((o, a, r, rtg, t, m))
            elif args.type == 'id' or args.type == 'soft':
                dt.policy.set_task_id(i)
            
            mean_reward, std = evaluate_traj_policy(
                model=dt, 
                env=env, 
                n_eval_episodes=1,
                max_ep_length=max_ep_length,
                deterministic=True,
                obs_mean=test_obs_means[i],
                obs_std=test_obs_stds[i],
                scale=scale,
                target_return=test_env_targets[i],
            )
            print(f"After Learning Test Env{idx}: {mean_reward:.2f} +/- {std:.2f}")
        dt.save(path=f'./tests/model/dt_mt/cheetah_vel/{len(train_envs)}')

    else: # Evaluation
        print_y("<< Testing DT (multi-task) Model >>")
        # Loading Model
        _dt = DT.load(path=f'./tests/model/dt_mt/cheetah_vel/{len(train_envs)}')
        for i, (env, idx) in enumerate(zip(train_envs, train_env_indxes)):
            if args.type == 'fix':
                o, a, r, d, rtg, t, m = train_buff.buffers[i].sample_prompt(1)
                _dt.policy.set_prompt((o, a, r, rtg, t, m))
            elif args.type == 'id' or args.type == 'soft':
                _dt.policy.set_task_id(i)
            
            mean_reward, std = evaluate_traj_policy(
                model=_dt, 
                env=env, 
                n_eval_episodes=1,
                max_ep_length=max_ep_length,
                deterministic=True,
                obs_mean=train_obs_means[i],
                obs_std=train_obs_stds[i],
                scale=scale,
                target_return=train_env_targets[i],
            )
            print(f"After Learning Train Env{idx}: {mean_reward:.2f} +/- {std:.2f}")
        print_b("="*20)
        for i, (env, idx) in enumerate(zip(test_envs, test_env_indxes)):
            if args.type == 'fix':
                o, a, r, d, rtg, t, m = test_buff.buffers[i].sample_prompt(1)
                _dt.policy.set_prompt((o, a, r, rtg, t, m))
            elif args.type == 'id' or args.type == 'soft':
                _dt.policy.set_task_id(i)
            
            mean_reward, std = evaluate_traj_policy(
                model=_dt, 
                env=env, 
                n_eval_episodes=1,
                max_ep_length=max_ep_length,
                deterministic=True,
                obs_mean=test_obs_means[i],
                obs_std=test_obs_stds[i],
                scale=scale,
                target_return=test_env_targets[i],
            )
            print(f"After Learning Test Env{idx}: {mean_reward:.2f} +/- {std:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--type", type=str, default=None)
    args = parser.parse_args()
    main(args)
