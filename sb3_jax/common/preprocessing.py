import warnings 
from functools import partial
from typing import Dict, Tuple, Union

import jax
import numpy as np
import jax.numpy as jnp
from jax import nn
from gym import spaces


def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0


def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
) -> bool:
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False


def maybe_transpose(observation: np.ndarray, observation_space: spaces.Space) -> np.ndarray:
    # Avoid circular import
    from stable_baselines3.common.vec_env import VecTransposeImage

    if is_image_space(observation_space):
        if not (observation.shape == observation_space.shape or observation.shape[1:] == observation_space.shape):
            # Try to re-order the channels
            transpose_obs = VecTransposeImage.transpose_image(observation)
            if transpose_obs.shape == observation_space.shape or transpose_obs.shape[1:] == observation_space.shape:
                observation = transpose_obs
    return observation


def preprocess_obs(
    obs: jnp.array,
    observation_space: bool, #  spaces.Space,
    normalize_images: bool = True,
) -> Union[np.array, Dict[str, np.array]]:

    if isinstance(observation_space, spaces.Box):
        if is_image_space(observation_space) and normalize_images:
            return obs.astype('float32') / 255.0
        return obs.astype('float32')

    elif isinstance(observation_space, spaces.Discrete):
        obs = nn.one_hot(obs.astype('int32'), num_classes=observation_space.n).astype('float32')
        return np.array(obs)
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")
    

def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def check_for_nested_spaces(obs_space: spaces.Space):
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.

    :param obs_space: an observation space
    :return:
    """
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = obs_space.spaces.values() if isinstance(obs_space, spaces.Dict) else obs_space.spaces
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )
