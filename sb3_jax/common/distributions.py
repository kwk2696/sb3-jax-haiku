"""Probability distributions."""

import math
from abc import ABC, abstractmethod  
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import jax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from gym import spaces

from sb3_jax.common.jax_layers import init_weights


@partial(jax.jit, static_argnums=1)
def sum_independent_dims(jnp_array: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp_array.sum(axis=axis)


class Distribution(hk.Module, ABC):
    def __init__(self):
        super(Distribution, self).__init__()


class DiagGaussianDistribution(Distribution):
    """Gaussian distribution with diagonal covariance matrix, for continuous actions."""

    def __init__(self, action_dim: int, log_std_init: float = 0.0):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.log_std_init = log_std_init
    
    def __call__(self, latent_pi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ :return: Mean and log std of actor."""
        mean_actions = hk.Linear(self.action_dim, **init_weights())(latent_pi)
        log_std = hk.get_parameter("log_std", (self.action_dim,), init=hk.initializers.Constant(self.log_std_init))
        return mean_actions, log_std


class StateDependentNoiseDistribution(Distribution):
    """Distribution class for using generalized State Dependent Exploration (gSDE)."""
    
    def __init__(
        self, 
        action_dim: int, 
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False, 
        learn_features: bool = False,
        epsilon: float = 1e-6,
    ): 
        super(StateDependentNoiseDistribution, self).__init__()
        self.action_dim = action_dim
    
    def __call__(self, latent_pi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean_actions = hk.Linear(self.action_dim, **init_weights())(latent_pi)


class DistributionFn(ABC): 
    def __init__(self):
        super(DistributionFn, self).__init__()

    @abstractmethod
    def sample(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Returns the most likely action (deterministic output)."""

    @abstractmethod
    def mode(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        """Returns a sample from the probabiltiy distribution (stochastic output)."""
    
    @abstractmethod
    def log_prob(self, actions: jnp.ndarray, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        """Returns the log probability."""
    
    @abstractmethod
    def entropy(self, log_std: jnp.ndarray) -> jnp.ndarray:
        """Returns the entropy."""

    def get_actions(
        self, 
        mean_actions: jnp.ndarray, 
        log_std: Optional[jnp.ndarray] = None,
        deterministic: bool = False, 
        key: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        if deterministic:
            return self.mode(mean_actions, log_std)
        return self.sample(mean_actions, log_std, key)


class DiagGaussianDistributionFn(DistributionFn):
    def __init__(self):
        super(DiagGaussianDistributionFn, self).__init__()

    @partial(jax.jit, static_argnums=0)
    def sample(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        action_std = jnp.exp(log_std)
        noise = jax.random.normal(key, action_std.shape)
        return mean_actions + noise * action_std

    @partial(jax.jit, static_argnums=0)
    def mode(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        return mean_actions
    
    @partial(jax.jit, static_argnums=0)
    def log_prob(self, actions: jnp.ndarray, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        action_std = jnp.exp(log_std)  
        noise = (actions - mean_actions) / action_std
        log_prob = -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))
        log_prob = sum_independent_dims(log_prob, 1) if len(log_prob.shape) > 1 else sum_independent_dims(log_prob, None)
        return log_prob

    @partial(jax.jit, static_argnums=0)
    def entropy(self, log_std: jnp.ndarray) -> jnp.ndarray:
        action_std = jnp.exp(log_std)
        entropy = 0.5 + 0.5 * jnp.log(2 * math.pi) + jnp.log(action_std)        
        entropy = sum_independent_dims(entropy, 1) if len(entropy.shape) > 1 else sum_independent_dims(entropy, None)
        return entropy


def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False
) -> Distribution:
    
    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        if use_sde:
            raise NotImplementedError("use_sde is not implemented.")
        else:
            return DiagGaussianDistribution, DiagGaussianDistributionFn()
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


