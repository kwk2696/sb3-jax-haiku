import warnings
import pickle
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Tuple, Dict, Generator, List, Optional, Union

import jax
import numpy as np
import jax.numpy as jnp
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from sb3_jax.common.preprocessing import get_obs_shape, get_flattened_obs_dim, get_act_dim
from sb3_jax.common.type_aliases import (
    RolloutBufferSamples,
    ReplayBufferSamples,
    TrajectoryBufferSamples,
)
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.obs_dim = get_flattened_obs_dim(observation_space)
        self.act_dim = get_act_dim(action_space)
        self.pos = 0
        self.full = False
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[RolloutBufferSamples]:
        raise NotImplementedError()
    
    # @partial(jax.jit, static_argnums=0)
    def to_jnp(self, array: np.ndarray, copy: bool = True) -> jnp.ndarray:
        """Convert a numpy array to a jax numpy array."""
        return jnp.array(array)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path:str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)


class RolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        # Convert to numpy
        last_values = np.array(last_values).flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = np.array(value).flatten()
        self.log_probs[self.pos] = np.array(log_prob)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_jnp, data)))


class ReplayBuffer(BaseBuffer):
    """Replay buffer used in off-policy algorithms like SAC/TD3."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeout termination properly if needed
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.act_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
 
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones taht are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_jnp, data)))


class OfflineBuffer(BaseBuffer):
    """Buffer used in offline algorithms like BC."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
    ):
        super(OfflineBuffer, self).__init__(buffer_size, observation_space, action_space)

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=action_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Optional[List[Dict[str, Any]]] = None, 
    ) -> None:

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = obs.reshape((self.n_envs,) + self.obs_shape)

        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.act_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def add_traj(
        self,
        traj: Dict[str, np.ndarray]
    ):
        for t in range(traj['observations'].shape[0]):
            self.add(
                traj['observations'][t], 
                traj['next_observations'][t],
                traj['actions'][t],
                traj['rewards'][t],
                traj['terminals'][t],
            )
 
    def sample(self, batch_size: int, env:Optional[VecNormalize] = None) -> ReplayBufferSamples:
        return super().sample(batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            self.next_observations[batch_inds, env_indices, :],
            self.dones[batch_inds, env_indices].reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_jnp, data)))


class TrajectoryBuffer(BaseBuffer):
    """Buffer used in DT."""

    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        max_length: int,
        max_ep_length: int,
        scale: float, # normalization for rewards/returns
        prompt_trajectories: Optional[List[Dict[str, np.ndarray]]] = None,
        prompt_episode: int = 1,
        prompt_length: int = None,
        buffer_size: int = None,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        n_envs: int = 1, # Not used
    ):
        super(TrajectoryBuffer, self).__init__(buffer_size, observation_space, action_space) 
        self.trajectories = trajectories
        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.scale = scale

        # prompt params
        self.prompt_trajectories = prompt_trajectories
        self.prompt_episode = prompt_episode
        self.prompt_length = prompt_length

        self.setup()

    def setup(self) -> None:
        # 1. traj setup
        observations, actions, traj_lengths, returns = [], [], [], []
        for path in self.trajectories:
            observations.append(path['observations'])
            actions.append(path['actions'])
            traj_lengths.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lengths, returns = np.array(traj_lengths), np.array(returns)
        self.traj_lengths = traj_lengths

        observations, actions = np.concatenate(observations, axis=0), np.concatenate(actions, axis=0)
        self.obs_mean, self.obs_std = np.mean(observations, axis=0), np.std(observations, axis=0) + 1e-6
        self.act_mean, self.act_std = np.mean(actions, axis=0), np.std(actions, axis=0) + 1e-6
        num_timesteps = sum(traj_lengths)
        
        print('=' * 20, " total  traj ", '=' * 20)
        print(f'{len(traj_lengths)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, Min return: {np.min(returns):.2f}')
        print('=' * 55)

        # only train on top pct_traj trajectories
        pct_traj = 1.
        num_timesteps = max(int(pct_traj*num_timesteps), 1)
        sorted_inds = np.argsort(returns) # lowest to highest
        num_trajectories = 1 
        timesteps = traj_lengths[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + traj_lengths[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lengths[sorted_inds[ind]]
            num_trajectories += 1 
            ind -= 1 
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = traj_lengths[sorted_inds] / sum(traj_lengths[sorted_inds])
        self.sorted_inds = sorted_inds
        self.num_trajectories = num_trajectories
        self.num_timesteps = num_timesteps

        # 2. prompt traj setup
        observations, traj_lengths, returns = [], [], []
        for path in self.prompt_trajectories:
            observations.append(path['observations'])
            traj_lengths.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lengths, returns = np.array(traj_lengths), np.array(returns)
        self.prompt_traj_lengths = traj_lengths

        print('=' * 20, " prompt traj ", '=' * 20)
        print(f'{len(traj_lengths)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, Min return: {np.min(returns):.2f}')
        print('=' * 55)

        observations = np.concatenate(observations, axis=0)
        num_timesteps = sum(traj_lengths)

        # only train on top pct_traj trajectories
        pct_traj = 1.
        num_timesteps = max(int(pct_traj*num_timesteps), 1)
        sorted_inds = np.argsort(returns) # lowest to highest
        num_trajectories = 1 
        timesteps = traj_lengths[sorted_inds[-1]]
        ind = len(self.prompt_trajectories) - 2
        while ind >= 0 and timesteps + traj_lengths[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lengths[sorted_inds[ind]]
            num_trajectories += 1 
            ind -= 1 
        sorted_inds = sorted_inds[-num_trajectories:]
        
        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.prompt_sorted_inds = sorted_inds
        self.prompt_num_trajectories = 1 # num_trajectories

    def sample(self, batch_size: int, max_length: int = None, env: Optional[VecNormalize] = None) -> TrajectoryBufferSamples:
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=batch_size,
            replace=True,
            p=self.p_sample
        )
        return self._get_samples(batch_inds, max_length, env=env)
    
    def _get_samples(self, batch_inds: np.ndarray, max_length: int = None, env: Optional[VecNormalize] = None) -> TrajectoryBufferSamples:
        observations, actions, next_observations, rewards, dones, returns_to_go, timesteps, masks = [], [], [], [], [], [], [], []
        if max_length is None: max_length = self.max_length
    
        for i in range(len(batch_inds)):
            traj = self.trajectories[int(self.sorted_inds[batch_inds[i]])]
            si = np.random.randint(0, traj['rewards'].shape[0] - 1)
            
            # get sequences from dataset
            observations.append(traj['observations'][si:si + max_length].reshape(1, -1, self.obs_dim))
            actions.append(traj['actions'][si:si + max_length].reshape(1, -1, self.act_dim))
            next_observations.append(traj['observations'][si + 1: si + max_length + 1].reshape(1, -1, self.obs_dim))
            rewards.append(traj['rewards'][si:si + max_length].reshape(1, -1, 1))
            if 'terminals' in traj: dones.append(traj['terminals'][si:si + max_length].reshape(1, -1))
            else: dones.append(traj['dones'][si:si + max_length].reshape(1, -1))
            timesteps.append(np.arange(si, si + observations[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_length] = self.max_ep_length - 1 # padding cutoff
            returns_to_go.append(self.discount_cumsum(traj['rewards'][si:], gamma=1.)[:observations[-1].shape[1] + 1].reshape(1, -1, 1))
            if returns_to_go[-1].shape[1] <= observations[-1].shape[1]:
                returns_to_go[-1] = np.concatenate([returns_to_go[-1], np.zeros((1, 1, 1))], axis=1)
            
            # padding and state + reward normalization
            tlen, ntlen = observations[-1].shape[1], next_observations[-1].shape[1]
            observations[-1] = np.concatenate([np.zeros((1, max_length - tlen, self.obs_dim)), observations[-1]], axis=1)
            observations[-1] = (observations[-1] - self.obs_mean) / self.obs_std
            actions[-1] = np.concatenate([np.ones((1, max_length - tlen, self.act_dim)) * -10, actions[-1]], axis=1)
            next_observations[-1] = np.concatenate([np.zeros((1, max_length - ntlen, self.obs_dim)), next_observations[-1]], axis=1)
            next_observations[-1] = (next_observations[-1] - self.obs_mean) / self.obs_std
            rewards[-1] = np.concatenate([np.zeros((1, max_length - tlen, 1)), rewards[-1]], axis=1)
            dones[-1] = np.concatenate([np.ones((1, max_length - tlen)) * 2, dones[-1]], axis=1)
            returns_to_go[-1] = np.concatenate([np.zeros((1, max_length - tlen, 1)), returns_to_go[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_length - tlen)), timesteps[-1]], axis=1)
            masks.append(np.concatenate([np.zeros((1, max_length - tlen)), np.ones((1, tlen))], axis=1))

        data = (
            np.concatenate(observations, axis=0).astype(np.float32),
            np.concatenate(actions, axis=0).astype(np.float32),
            np.concatenate(next_observations, axis=0).astype(np.float32),
            np.concatenate(rewards, axis=0).astype(np.float32),
            np.concatenate(dones, axis=0).astype(np.int32),
            np.concatenate(returns_to_go, axis=0)[:,:-1].astype(np.float32),
            np.concatenate(timesteps, axis=0).astype(np.int32),
            np.concatenate(masks, axis=0).astype(np.float32),
        )
        return TrajectoryBufferSamples(*tuple(map(self.to_jnp, data)))
    
    def sample_prompt(self, batch_size: int) -> TrajectoryBufferSamples:
        batch_inds = np.random.choice(
            np.arange(len(self.prompt_trajectories)),
            size=int(self.prompt_episode * batch_size), # J trajectory segments
            replace=True,
            # p=self.p_sample,
        )
        return self._get_prompt_samples(batch_inds)

    def _get_prompt_samples(self, batch_inds: np.ndarray) -> TrajectoryBufferSamples:
        observations, actions, rewards, dones, returns_to_go, timesteps, masks = [], [], [], [], [], [], []
        batch_size = int(len(batch_inds) / self.prompt_episode)

        for i in range(len(batch_inds)):
            traj = self.prompt_trajectories[int(batch_inds[i])] # random select traj
            si = np.random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            observations.append(traj['observations'][si:si + self.prompt_length].reshape(1, -1, self.obs_dim))
            actions.append(traj['actions'][si:si + self.prompt_length].reshape(1, -1, self.act_dim))
            rewards.append(traj['rewards'][si:si + self.prompt_length].reshape(1, -1, 1))
            if 'terminals' in traj: dones.append(traj['terminals'][si:si + self.prompt_length].reshape(1, -1))
            else: dones.append(traj['dones'][si:si + self.prompt_length].reshape(1, -1))
            timesteps.append(np.arange(si, si + observations[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_length] = self.max_ep_length - 1 # padding cutoff
            returns_to_go.append(self.discount_cumsum(traj['rewards'][si:], gamma=1.)[:observations[-1].shape[1] + 1].reshape(1, -1, 1))
            if returns_to_go[-1].shape[1] <= observations[-1].shape[1]:
                returns_to_go[-1] = np.concatenate([returns_to_go[-1], np.zeros((1, 1, 1))], axis=1)
            
            # padding and state + reward normalization
            tlen = observations[-1].shape[1]
            observations[-1] = np.concatenate([np.zeros((1, self.prompt_length - tlen, self.obs_dim)), observations[-1]], axis=1)
            observations[-1] = (observations[-1] - self.obs_mean) / self.obs_std
            actions[-1] = np.concatenate([np.ones((1, self.prompt_length - tlen, self.act_dim)) * -10, actions[-1]], axis=1)
            rewards[-1] = np.concatenate([np.zeros((1, self.prompt_length - tlen, 1)), rewards[-1]], axis=1)
            dones[-1] = np.concatenate([np.ones((1, self.prompt_length - tlen)) * 2, dones[-1]], axis=1)
            returns_to_go[-1] = np.concatenate([np.zeros((1, self.prompt_length - tlen, 1)), returns_to_go[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.prompt_length - tlen)), timesteps[-1]], axis=1)
            masks.append(np.concatenate([np.zeros((1, self.prompt_length - tlen)), np.ones((1, tlen))], axis=1))

        observations = np.concatenate(observations, axis=0).astype(np.float32)
        actions = np.concatenate(actions, axis=0).astype(np.float32)
        rewards = np.concatenate(rewards, axis=0).astype(np.float32)
        dones = np.concatenate(dones, axis=0).astype(np.int32)
        returns_to_go = np.concatenate(returns_to_go, axis=0).astype(np.float32)
        timesteps = np.concatenate(timesteps, axis=0).astype(np.int32)
        masks = np.concatenate(masks, axis=0).astype(np.float32)
        
        data = (
            observations.reshape(batch_size, -1, observations.shape[-1]).astype(np.float32),
            actions.reshape(batch_size, -1, actions.shape[-1]).astype(np.float32),
            None,
            rewards.reshape(batch_size, -1, rewards.shape[-1]).astype(np.float32),
            dones.reshape(batch_size, -1).astype(np.int32),
            returns_to_go[:,:-1,:].reshape(batch_size, -1, returns_to_go.shape[-1]).astype(np.float32),
            timesteps.reshape(batch_size, -1).astype(np.int32),
            masks.reshape(batch_size, -1).astype(np.float32),
        )
        return TrajectoryBufferSamples(*tuple(map(self.to_jnp, data)))

    def discount_cumsum(self, x: np.ndarray, gamma: float):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum


class MTTrajectoryBuffer(BaseBuffer):
    """Multi-task trajectory buffer."""

    def __init__(
        self,
        buff_cls: BaseBuffer,
        max_length: int,
        max_ep_length: int,
        scale: float,
        prompt_episode: int = 1,
        prompt_length: int = None,
        buffer_size: int = None,
        total_mean: bool = True,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        n_envs: int = 1, # Not used
    ):
        super(MTTrajectoryBuffer, self).__init__(buffer_size, observation_space, action_space)
        self.buff_cls = buff_cls if buff_cls is not None else TrajectoryBuffer
        self._buffers = []
        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.scale = scale

        self.prompt_episode = prompt_episode
        self.prompt_length = prompt_length
        
        self.total_num_trajectories = 0
        self.total_num_timesteps = 0

        self.total_mean = total_mean
        self.obs_means, self.obs_stds = [], []

    @property
    def buffers(self):
        return self._buffers

    def sample(self, batch_size: int, max_length: int = None, env: Optional[VecNormalize] = None) -> TrajectoryBufferSamples:
        # sample batch_size per task
        observations, actions, next_observations, rewards, dones, returns_to_go, timesteps, masks = [], [], [], [], [], [], [], []
        for buff in self.buffers:
            samples = buff.sample(batch_size, max_length)
            observations.append(samples.observations)
            actions.append(samples.actions)
            next_observations.append(samples.next_observations)
            rewards.append(samples.rewards)
            dones.append(samples.dones)
            returns_to_go.append(samples.returns_to_go)
            timesteps.append(samples.timesteps)
            masks.append(samples.masks)

        # concatenate
        data = (
            jnp.concatenate(observations),
            jnp.concatenate(actions),
            jnp.concatenate(next_observations),
            jnp.concatenate(rewards),
            jnp.concatenate(dones),
            jnp.concatenate(returns_to_go),
            jnp.concatenate(timesteps),
            jnp.concatenate(masks),
        )
        return TrajectoryBufferSamples(*tuple(data))

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> TrajectoryBufferSamples:
        raise NotImplementedError
    
    def sample_prompt(self, batch_size: int) -> TrajectoryBufferSamples:
        # sample batch_size per task
        observations, actions, rewards, dones, returns_to_go, timesteps, masks = [], [], [], [], [], [], []
        for buff in self.buffers:
            samples = buff.sample_prompt(batch_size)
            observations.append(samples.observations)
            actions.append(samples.actions)
            rewards.append(samples.rewards)
            dones.append(samples.dones)
            returns_to_go.append(samples.returns_to_go)
            timesteps.append(samples.timesteps)
            masks.append(samples.masks)

        # concatenate
        data = (
            jnp.concatenate(observations),
            jnp.concatenate(actions),
            None,
            jnp.concatenate(rewards),
            jnp.concatenate(dones),
            jnp.concatenate(returns_to_go),
            jnp.concatenate(timesteps),
            jnp.concatenate(masks),
        )
        return TrajectoryBufferSamples(*tuple(data))

    def add_task(
        self, 
        trajectories: Dict[str, np.ndarray], 
        prompt_trajectories: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        self.buffers.append(self.buff_cls(
            trajectories,
            self.max_length,
            self.max_ep_length,
            self.scale,
            prompt_trajectories,
            self.prompt_episode,
            self.prompt_length,
            observation_space=self.observation_space,
            action_space=self.action_space,
        ))
        self.obs_means.append(self.buffers[-1].obs_mean)
        self.obs_stds.append(self.buffers[-1].obs_std)

        self.total_num_trajectories += self.buffers[-1].num_trajectories
        self.total_num_timesteps += self.buffers[-1].num_timesteps
        
    def set_total_mean(self, total_obs: Tuple[float] = None, total_act: Tuple[float] = None) -> None:
        
        total_obs_, total_act_ = [], []
        for buff in self.buffers:
            for traj in buff.trajectories:
                total_obs_.extend(traj['observations'])
                total_act_.extend(traj['actions'])

        if total_obs is None:
            self.total_obs_mean, self.total_obs_std = np.mean(total_obs_, axis=0), np.std(total_obs_, axis=0) + 1e-6
        else:
            self.total_obs_mean, self.total_obs_std = total_obs
        if total_act is None:
            self.total_act_mean, self.total_act_std = np.mean(total_act_, axis=0), np.std(total_act_, axis=0) + 1e-6
        else:
            self.total_act_mean, self.total_act_std = total_act

        for buff in self.buffers:
            buff.obs_mean = self.total_obs_mean
            buff.obs_std = self.total_obs_std
            buff.act_mean = self.total_act_mean
            buff.act_std = self.total_act_std
