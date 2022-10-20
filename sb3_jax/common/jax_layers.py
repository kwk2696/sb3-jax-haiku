import os
from abc import ABC
from itertools import zip_longest
from typing import Callable, Any, Optional, Dict, List, Tuple, Type, Union, Sequence

import gym
import jax.numpy as jnp
from jax import nn
import haiku as hk

from sb3_jax.common.preprocessing import get_flattened_obs_dim


def init_weights(gain: Optional[float] = 1) -> Dict[str, hk.initializers.Initializer]:

    return {"w_init": hk.initializers.Orthogonal(scale=gain),
            "b_init": hk.initializers.Constant(constant=0.)}


class BaseFeaturesExtractor(hk.Module):
    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray: 
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """ JAX module that flatten a vector keeping dimension """
    def __init__(self, observation_space: gym.Space):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return hk.Flatten()(observations)


class NormalizeFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, eps: float = 1e-5):
        super(BaseFeaturesExtractor, self).__init__()
        self._eps = eps
        self._dim = sum(observation_space.shape)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        observations = hk.Flatten()(observations)
        running_mean = hk.get_parameter("running_mean", (self._dim,), init=hk.initializers.Constant(0.))
        running_var = hk.get_parameter("running_var", (self.dim,), init=hk.initializers.Constant(1.))


class MLP(hk.Module):
    def __init__(
        self, 
        net_arch: List[int], 
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        squash_output: bool = False,
    ):
        super(MLP, self).__init__()
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.squash_output = squash_output

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.net_arch):
            x = hk.Linear(size, **init_weights())(x)
            if i + 1 < len(self.net_arch):
                x = self.activation_fn(x)
        if self.squash_output:
            x = nn.tanh(x)
        return x


def create_mlp(
    output_dim: int, 
    net_arch: List[int], 
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
    squash_output: bool = False,
) -> hk.Module:
    if output_dim > 0:
        net_arch = list(net_arch)
        net_arch.append(output_dim)
    return MLP(net_arch, activation_fn, squash_output)


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch


class MlpExtractor(ABC): 

    def __init__(
        self,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
    ):
        super(MlpExtractor, self).__init__()
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.shared_layers, self.policy_only_layers, self.value_only_layers = [], [], []
        for layer in self.net_arch:
            if isinstance(layer, int):
                self.shared_layers.append(layer)
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]
                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        # Build non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                self.policy_only_layers.append(pi_layer_size)

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                self.value_only_layers.append(vf_layer_size)     

    def setup(self) -> Tuple[MLP, MLP, MLP]:
        shared_net = create_mlp(-1, self.shared_layers, self.activation_fn)
        policy_net = create_mlp(-1, self.policy_only_layers, self.activation_fn)
        value_net = create_mlp(-1, self.value_only_layers, self.activation_fn)
        return shared_net, policy_net, value_net
    

def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """Get the actor and critic network architectures for off-policy actor-critic."""
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch
