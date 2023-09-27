import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import argparse
import gym
import pickle
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv

import numpy as np

from stable_baselines3.common import env_util, vec_env
from sb3_jax import DT
from sb3_jax.dt.policies import MlpPolicy
from sb3_jax.common.buffers import MTTrajectoryBuffer
from sb3_jax.common.evaluation import evaluate_traj_policy
from sb3_jax.common.utils import print_y


def main(args):
    max_ep_length = 200
    env_target = 1500
    scale = 1000.

    env1 = HalfCheetahDirEnv([{'direction':1}], include_goal=False)
    env2 = HalfCheetahDirEnv([{'direction':-1}], include_goal=False)
    envs = [env1, env2]

    env = env1
    obs_space, act_space = env.observation_space, env.action_space

    prompt_length = 1 
    if args.type == 'fix' or args.type == 'soft':
        prompt_length = 5

    # Make Buffer
    buff = MTTrajectoryBuffer(
        max_length=20,
        max_ep_length=max_ep_length,
        scale=scale,
        prompt_length=prompt_length,
        observation_space=obs_space,
        action_space=act_space,
    )
    for i in range(2):
        data_path = f'./tests/data/cheetah_dir/cheetah_dir-{i}-expert.pkl'
        with open(data_path, 'rb') as f:
            trajectories = pickle.load(f)
        print(len(trajectories), trajectories[0].keys())
        prompt_data_path = f'./tests/data/cheetah_dir/cheetah_dir-{i}-prompt-expert.pkl'
        with open(prompt_data_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
        print(len(prompt_trajectories), prompt_trajectories[0].keys())
        buff.add_task(trajectories, prompt_trajectories)

    obs_means = [buff.obs_mean for buff in buff.buffers]
    obs_stds = [buff.obs_std for buff in buff.buffers]
    env_targets = [env_target/scale for buff in buff.buffers]

    if args.train:
        print_y("<< Training DT (multi-task) Model >>")
        # Make DT
        dt = DT(
            policy=MlpPolicy,
            env=env,
            replay_buffer=buff,
            learning_rate=1e-4,
            batch_size=128,
            verbose=1,
            wandb_log=f'test/dt_mt/{args.type}',
            policy_kwargs=dict(
                lr_warmup=0,
                num_tasks=len(envs),
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
        
        for i, env in enumerate(envs):
            if args.type == 'fix':
                o, a, r, d, rtg, t, m = buff.buffers[i].sample_prompt(1)
                dt.policy.set_prompt((o, a, r, rtg, t, m))
            elif args.type == 'id' or args.type == 'soft':
                dt.policy.set_task_id(i)
           
            mean_reward, _ = evaluate_traj_policy(
                model=dt, 
                env=env, 
                n_eval_episodes=1,
                max_ep_length=max_ep_length,
                deterministic=True,
                obs_mean=obs_means[i],
                obs_std=obs_stds[i],
                scale=scale,
                target_return=env_targets[i],
            )
            print(f"Before Learning Env{i}: {mean_reward}")
        dt.learn(total_timesteps=2_000, log_interval=100)
        for i, env in enumerate(envs):
            if args.type == 'fix':
                o, a, r, d, rtg, t, m = buff.buffers[i].sample_prompt(1)
                dt.policy.set_prompt((o, a, r, rtg, t, m))
            elif args.type == 'id' or args.type == 'soft':
                dt.policy.set_task_id(i)
            
            mean_reward, _ = evaluate_traj_policy(
                model=dt, 
                env=env, 
                n_eval_episodes=1,
                max_ep_length=max_ep_length,
                deterministic=True,
                obs_mean=obs_means[i],
                obs_std=obs_stds[i],
                scale=scale,
                target_return=env_targets[i],
            )
            print(f"After Learning Env{i}: {mean_reward}")
        dt.save(path='./tests/model/dt_mt/{type}')
    
    if args.test:
        print_y("<< Testing DT (multi-task) Model >>")
        # Loading Model
        _dt = DT.load(path='./tests/model/dt_mt/{type}')
        for i, env in enumerate(envs):
            if args.type == 'fix':
                o, a, r, d, rtg, t, m = buff.buffers[i].sample_prompt(1)
                _dt.policy.set_prompt((o, a, r, rtg, t, m))
            elif args.type == 'id' or args.type == 'soft':
                _dt.policy.set_task_id(i)
            
            mean_reward, _ = evaluate_traj_policy(
                model=_dt, 
                env=env, 
                n_eval_episodes=1,
                max_ep_length=max_ep_length,
                deterministic=True,
                obs_mean=obs_means[i],
                obs_std=obs_stds[i],
                scale=scale,
                target_return=env_targets[i],
            )
            print(f"Load Learning Env{i}: {mean_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--type", type=str, default=None)
    args = parser.parse_args()
    main(args)
