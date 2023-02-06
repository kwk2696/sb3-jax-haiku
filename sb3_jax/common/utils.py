import glob
import os
import platform
import random
from collections import deque
from itertools import zip_longest
from typing import Dict, Iterable, Optional, Tuple, Union

import gym
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from stable_baselines3.common.logger import Logger, configure
from sb3_jax.common.type_aliases import GymEnv, Schedule, JnpDict, TrainFreq, TrainFrequencyUnit


def obs_as_jnp(obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]]) -> Union[jnp.array, JnpDict]:
    """Moves the observation to the jnp."""
    if isinstance(obs, np.ndarray):
        return jnp.array(obs)
    elif isinstance(obs, dict):
        return {key: jnp.array(_obs) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def get_dummy_obs(observation_space: gym.spaces) -> np.ndarray:
    if isinstance(observation_space, gym.spaces.Dict):
        observation = observation_space.sample()
        for key, _ in observation_space.spaces.items():
            observation[key] = observation[key][None, ...].astype(np.float32)# for expanding dimension [batch, ...]
    else:
        observation = observation_space.sample()[None,...].astype(np.float32)
    return observation


def get_dummy_act(action_space: gym.spaces) -> np.ndarray:
    if isinstance(action_space, gym.spaces.Box):
        action = action_space.sample()[None, ...].astype(np.float32)
    elif isinstance(action_space, gym.spaces.Discrete):
        action = action_space.sample()
        action = nn.one_hot(np.array([action]).astype(np.int32), num_classes=action_space.n).astype('float32')
    else:
        raise NotImplementedError(f"Yet implemented for {action_space}")
    return action


def get_dummy_done() -> np.ndarray:
    done = np.zeros((1,))[None, ...].astype(np.float32)
    return done


def get_dummy_transition(
    observation_space: gym.spaces, 
    action_space: gym.spaces,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observation = get_dummy_obs(observation_space)
    action = get_dummy_act(action_space)
    next_observation = get_dummy_obs(observation_space)
    done = get_dummy_done()
    return observation, action, next_observation, done


def set_random_seed(seed: int) -> None:
    """ Seed the different random generators."""
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # No torch RNG


def get_schedule_fn(value_schedule: Union[Schedule, float, int]) -> Schedule:
    """Transform (if needed) learning rate and clip range (for PPO) to callable."""
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def get_linear_fn(start: float, end: float, end_fraction: float) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_fn(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)
    """

    def func(_):
        return val

    return func


def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """Returns the latest run number for the given log name and log path,"""
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def configure_logger(
    verbose: int = 0,
    tensorboard_log: Optional[str] = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
) -> Logger:
    """Configure the logger's outputs."""
    save_path, format_strings = None, ["stdout"]

    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        if verbose >= 1:
            format_strings = ["stdout", "tensorboard"]
        else:
            format_strings = ["tensorboard"]
    elif verbose == 0:
        format_strings = [""]
    return configure(save_path, format_strings=format_strings)


def check_for_correct_spaces(env: GymEnv, observation_space: gym.spaces.Space, action_space: gym.spaces.Space) -> None:
    """Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm."""
    if observation_space != env.observation_space:
        raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
    if action_space != env.action_space:
        raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")


def is_vectorized_box_observation(observation: np.ndarray, observation_space: gym.spaces.Box) -> bool:
    """Detects and validates the shape, returns whether or not the observation is vectorized."""
    if observation.shape == observation_space.shape:
        return False
    elif observation.shape[1:] == observation_space.shape:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + f"Box environment, please use {observation_space.shape} "
            + "or (n_env, {}) for the observation shape.".format(", ".join(map(str, observation_space.shape)))
        )


def is_vectorized_discrete_observation(observation: Union[int, np.ndarray], observation_space: gym.spaces.Discrete) -> bool:
    """Ddetects and validates the shape, Returns whether or not the observation is vectorized."""
    if isinstance(observation, int) or observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
        return False
    elif len(observation.shape) == 1:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + "Discrete environment, please use () or (n_env,) for the observation shape."
        )


def is_vectorized_multidiscrete_observation(observation: np.ndarray, observation_space: gym.spaces.MultiDiscrete) -> bool:
    """Ddetects and validates the shape, returns whether or not the observation is vectorized."""
    if observation.shape == (len(observation_space.nvec),):
        return False
    elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
            + f"environment, please use ({len(observation_space.nvec)},) or "
            + f"(n_env, {len(observation_space.nvec)}) for the observation shape."
        )


def is_vectorized_multibinary_observation(observation: np.ndarray, observation_space: gym.spaces.MultiBinary) -> bool:
    """Detects and validates the shape, returns whether or not the observation is vectorized."""
    if observation.shape == (observation_space.n,):
        return False
    elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
            + f"environment, please use ({observation_space.n},) or "
            + f"(n_env, {observation_space.n}) for the observation shape."
        )


def is_vectorized_dict_observation(observation: np.ndarray, observation_space: gym.spaces.Dict) -> bool:
    """Detects and validates the shape, returns whether or not the observation is vectorized."""
    # We first assume that all observations are not vectorized
    all_non_vectorized = True
    for key, subspace in observation_space.spaces.items():
        # This fails when the observation is not vectorized
        # or when it has the wrong shape
        if observation[key].shape != subspace.shape:
            all_non_vectorized = False
            break

    if all_non_vectorized:
        return False

    all_vectorized = True
    # Now we check that all observation are vectorized and have the correct shape
    for key, subspace in observation_space.spaces.items():
        if observation[key].shape[1:] != subspace.shape:
            all_vectorized = False
            break

    if all_vectorized:
        return True
    else:
        # Retrieve error message
        error_msg = ""
        try:
            is_vectorized_observation(observation[key], observation_space.spaces[key])
        except ValueError as e:
            error_msg = f"{e}"
        raise ValueError(
            f"There seems to be a mix of vectorized and non-vectorized observations. "
            f"Unexpected observation shape {observation[key].shape} for key {key} "
            f"of type {observation_space.spaces[key]}. {error_msg}"
        )


def is_vectorized_observation(observation: Union[int, np.ndarray], observation_space: gym.spaces.Space) -> bool:
    """Returns whether or not the observation is vectorized."""
    is_vec_obs_func_dict = {
        gym.spaces.Box: is_vectorized_box_observation,
        gym.spaces.Discrete: is_vectorized_discrete_observation,
        gym.spaces.MultiDiscrete: is_vectorized_multidiscrete_observation,
        gym.spaces.MultiBinary: is_vectorized_multibinary_observation,
        gym.spaces.Dict: is_vectorized_dict_observation,
    }

    for space_type, is_vec_obs_func in is_vec_obs_func_dict.items():
        if isinstance(observation_space, space_type):
            return is_vec_obs_func(observation, observation_space)
    else:
        # for-else happens if no break is called
        raise ValueError(f"Error: Cannot determine if the observation is vectorized with the space type {observation_space}.")


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    return np.nan if len(arr) == 0 else np.mean(arr)


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def should_collect_more_steps(
    train_freq: TrainFreq,
    num_collected_steps: int,
    num_collected_episodes: int,
) -> bool:
    """Helper used in ``collect_rollouts()`` of off-policy algorithm."""
    if train_freq.unit == TrainFrequencyUnit.STEP:
        return num_collected_steps < train_freq.frequency

    elif train_freq.unit == TrainFrequencyUnit.EPISODE:
        return num_collected_episodes < train_freq.frequency

    else:
        raise ValueError(
            "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
            f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
        )


def get_system_info(print_info: bool = True) -> Tuple[Dict[str, str], str]:
    """Retrieve system and python env info for the current system."""
    env_info = {
        "OS": f"{platform.platform()} {platform.version()}",
        "Python": platform.python_version(),
        "JAX": jax.__version__,
        "Numpy": np.__version__,
        "Gym": gym.__version__,
    }
    env_info_str = ""
    for key, value in env_info.items():
        env_info_str += f"{key}: {value}\n"
    if print_info:
        print(env_info_str)
    return env_info, env_info_str
