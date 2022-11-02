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
from jax import nn
from jax import scipy

from sb3_jax.common.jax_layers import init_weights

# CAP the standard deviation of actor
LOG_STD_MAX = 2
LOG_STD_MIN = -2


@partial(jax.jit, static_argnums=1)
def sum_independent_dims(jnp_array: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp_array.sum(axis=axis)


class Distribution(hk.Module, ABC):
    def __init__(self):
        super(Distribution, self).__init__()


class DiagGaussianDistribution(Distribution):
    """Gaussian distribution with diagonal covariance matrix, for continuous actions."""

    def __init__(
        self, 
        action_dim: int, 
        log_std_init: float = 0.0,
        state_std: bool = False,
    ):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.log_std_init = log_std_init
        self.state_std = state_std

    def __call__(self, latent_pi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ :return: Mean and log std of actor."""
        mean_actions = hk.Linear(self.action_dim, **init_weights())(latent_pi)
        if self.state_std: 
            log_std = hk.Linear(self.action_dim, **init_weights(gain=0.01))(latent_pi) 
        else: 
            log_std = hk.get_parameter("log_std", (self.action_dim,), init=hk.initializers.Constant(self.log_std_init))
        #log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std


class CategoricalDistribution(Distribution):
    """Categorical Distribution for discrete actions."""
    def __init__(
        self,
        action_dim: int,
    ):
        super(CategoricalDistribution, self).__init__()
        self.action_dim = action_dim

    def __call__(self, latent_pi: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
        """ :return: probability of each actions. We keep the convention of torch.distributions."""
        action_logits = hk.Linear(self.action_dim, **init_weights())(latent_pi)
        action_logits = action_logits - scipy.special.logsumexp(action_logits, axis=-1, keepdims=True) 
        mean_actions = jnp.exp(action_logits)
        mean_acitons = nn.softmax(mean_actions, axis=1)
        return mean_actions, action_logits


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
    def entropy(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
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
    def entropy(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        action_std = jnp.exp(log_std)
        entropy = 0.5 + 0.5 * jnp.log(2 * math.pi) + jnp.log(action_std)        
        entropy = sum_independent_dims(entropy, 1) if len(entropy.shape) > 1 else sum_independent_dims(entropy, None)
        return entropy
      

class CategoricalDistributionFn(DistributionFn):
    def __init__(self):
        super(CategoricalDistributionFn, self).__init__()
        # Notice recieved actions are not real actions, they are softmaxed probabilities

    @partial(jax.jit, static_argnums=0)
    def sample(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray, key: int) -> jnp.ndarray:
        return jax.random.categorical(key, mean_actions)
    
    @partial(jax.jit, static_argnums=0)
    def mode(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        return jnp.argmax(mean_actions, axis=1)

    @partial(jax.jit, static_argnums=0)
    def log_prob(self, actions: jnp.ndarray, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        """Equivalent to torch log_prob."""
        log_prob = log_std # for Categorical log_std is log_prob
        actions = jnp.expand_dims(actions, axis=1)
        actions, log_pmf = jnp.broadcast_arrays(actions, log_prob)
        actions = actions[...,:1]
        return jnp.take_along_axis(log_pmf, actions, axis=1).squeeze()
    
    @partial(jax.jit, static_argnums=0)
    def entropy(self, mean_actions: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
        p_log_p = mean_actions * log_std
        return -p_log_p.sum(-1)


def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False
) -> Distribution:
    
    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        if use_sde:
            raise NotImplementedError("use_sde is not implemented.")
        else:
            return DiagGaussianDistribution, DiagGaussianDistributionFn()
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution, CategoricalDistributionFn()
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


