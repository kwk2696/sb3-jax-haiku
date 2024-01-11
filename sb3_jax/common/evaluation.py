import warnings
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tqdm
import gym
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from sb3_jax.common import base_class
from sb3_jax.common.preprocessing import get_flattened_obs_dim, get_act_dim


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 100,
    max_ep_length: int = None,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """Runs policy and returns average reward."""
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    #is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    #if not is_monitor_wrapped and warn:
    #    warnings.warn(
    #        "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
    #        "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
    #        "Consider wrapping environment first with ``Monitor`` wrapper.",
    #        UserWarning,
    #    )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()

    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states, _ = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())
                
                if dones[i] or (max_ep_length is not None and current_lengths[i] == max_ep_length):
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def evaluate_traj_policy(
    model: "base_class.BaseAlgorithm", 
    env: gym.Env,
    n_eval_episodes: int = 100,
    max_ep_length: int = None,
    deterministic: bool = True, 
    obs_mean: float = 0.0,
    obs_std: float = 0.0,
    scale: float = 1000.,
    target_return: float = None,
    return_episode_rewards: bool = False,
    return_episode_infos: bool = False,
    random_action: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """Runs trajectory policy, e.g. DT and returns average reward."""
    obs_dim, act_dim = get_flattened_obs_dim(env.observation_space), get_act_dim(env.action_space) 

    episode_rewards = []
    episode_lengths = []
    ep_return = target_return
    episode_infos = []

    for epi in tqdm.tqdm(range(n_eval_episodes)):
        observation = env.reset()
        
        # we keep all the histories
        # note that the latest action and reward will be "padding"
        observations = np.array(observation, dtype=np.float32).reshape(1, obs_dim)
        actions = np.zeros((0, act_dim), dtype=np.float32)
        rewards = np.zeros(0, dtype=np.float32)
        target_return = np.array(ep_return, dtype=np.float32).reshape(1, 1)
        timesteps = np.array(0, dtype=np.int32).reshape(1,1)
        
        current_reward, current_length, current_info = 0, 0, []
        while True:
            
            actions = np.concatenate([actions, np.zeros((1, act_dim))], axis=0)
            rewards = np.concatenate([rewards, np.zeros(1)])
            
            traj_obs = {
                "observations": (observations - obs_mean) / obs_std, 
                "actions": actions,
                "rewards": rewards, 
                "returns_to_go": target_return, 
                "timesteps": timesteps,
                "attention_mask": None,
            }
            if random_action:
                action = env.action_space.sample()
            else:
                action, _, info = model.predict(traj_obs, deterministic=deterministic)
                actions[-1] = action
            
            observation, reward, done, _ = env.step(action)

            cur_observation = np.array(observation, dtype=np.float32).reshape(1, obs_dim)
            observations = np.concatenate([observations, cur_observation], axis=0)
            rewards[-1] = reward 
            pred_return = target_return[0,-1] - (reward / scale)
            target_return = np.concatenate([target_return, pred_return.reshape(1, 1)], axis=1)
            timesteps = np.concatenate([timesteps, np.ones((1,1), dtype=np.int32) * (current_length+1)], axis=1)
            
            current_reward += reward
            current_length += 1
            current_info.append(info)
            
            if done or (max_ep_length is not None and current_length == max_ep_length):
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                episode_infos.append(current_info)
                break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    if return_episode_infos:
        return mean_reward, episode_infos
    return mean_reward, std_reward


def evaluate_mt_traj_policy(
    model: "base_class.BaseAlgorithm",
    envs: List[gym.Env],
    n_eval_episodes: int = 100,
    max_ep_length: int = None,
    deterministic: bool = True, 
    obs_means: List[float] = None,
    obs_stds: List[float] = None,
    scale: float = 1000.,
    target_returns: List[float] = None,
    return_episode_rewards: bool = False,
    random_action: bool = False,
    verbose: bool = False,
) -> None:
    """evaluation for multi-task traj envs."""
    mean_rewards, std_rewards = [], [] 

    for i, env in enumerate(envs):
        mean_reward, std_reward = evaluate_traj_policy(
            model, 
            env, 
            n_eval_episodes, 
            max_ep_length, 
            deterministic,
            obs_means[i],
            obs_stds[i],
            scale,
            target_returns[i],
            return_episode_rewards, 
            random_action,
        )
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        
        if verbose:
            print("="*10)
            print(f"{i}-th task return mean: {mean_reward}")
            print(f"{i}-th task return stdv: {std_reward}")

    return mean_rewards, std_rewards
