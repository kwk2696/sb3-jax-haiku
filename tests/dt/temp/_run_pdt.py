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
parser.add_argument("--tag", type=str, default="t")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--prompt_size", type=int, default=5)
parser.add_argument("--env_idx", type=int, default=15) # env idx
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()

max_ep_length = 200
env_target = 0
scale = 500.

eval_episodes = args.n
p_idx = args.env_idx

if not args.eval:
    envs = []
    env_indxes = [p_idx]
    for idx in env_indxes:
        with open(f"../data/config/cheetah_vel/config_cheetah_vel_task{idx}.pkl", 'rb') as f:
            task_info = pickle.load(f)
            print(task_info[0])
        env = HalfCheetahVelEnv(task_info, include_goal=False)
        envs.append(env)
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

    # Loading Pretrained Model Model
    print("Model Loading...")
    _dt = DT.load(path=f'../data/cheetah_vel/pretrained_dt/{args.tag}')
    _dt.replay_buffer = _buff
    _dt.wandb_log = f'FDT-{p_idx}'
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
        learning_rate=1e-4,
        batch_size=8,
        gradient_steps=20,
        verbose=1,
        wandb_log=f'{args.tag}/PDT-{p_idx}',
        policy_kwargs=dict(
            pretrained_policy=_dt.policy,
            prompt_size=args.prompt_size,
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
    print("Testing pdt actor ...")
    rd = buff.sample(1)
    (obs_preds, act_preds, rew_preds), _ = pdt.policy._actor(
        rd.observations, rd.actions, rd.rewards, rd.returns_to_go, rd.timesteps, rd.masks,
        deterministic=False, params=pdt.policy.params, 
        pretrained_params=pdt.policy.pretrained_policy.params, 
        pretrained_state=pdt.policy.pretrained_policy.state, 
        pretrained_rng=next(pdt.policy.pretrained_policy.rng)
    )
    print("Action Shape:", act_preds.shape)

    # Evaluating Random Policy
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
    ran_rew = np.mean(mean_reward)
    print(f"Random Policy: {np.mean(mean_reward)}")

    # Evaluating Pretrained Policy
    _dt.policy._task_id = args.task_id
    # # # # # # # # # # # # # #
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
    
    # PDT Learning ...
    # Disable pretrained prompts
    #_dt.policy.use_id = False
    #_dt.policy.use_prompt = False
    # # # # # # # # # # # # # # 
    pdt.learn(total_timesteps=5_000, log_interval=100)
    pdt.save(path=f'../data/cheetah_vel/prompt_dt/{args.tag}/task-{p_idx}/')

    prompt = pdt.policy._prompt(pdt.policy.params)
    #print("Trained Prompt:", prompt)

    print(f"Done Learning")
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

    with open(f'../data/cheetah_vel/prompt_dt/{args.tag}/test.txt', 'a') as f:
        f.write(f"Task idx -- {p_idx} --\n")
        f.write(f"Ranodm Policy    : {ran_rew:.2f}\n") 
        f.write(f"Before Finetuning: {bef_rew:.2f}\n")
        f.write(f"After  Finetuning: {aft_rew:.2f}\n")

else: # Evaluation

    envs = []
    env_indxes = [p_idx]
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

    _pdt = PDT.load(path=f'../data/cheetah_vel/prompt_dt/{args.tag}/task-{p_idx}')
    
    print(f"Model Evaluation at env {p_idx}")
    mean_reward, _ = evaluate_mt_traj_policy(
        model=_pdt, 
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
