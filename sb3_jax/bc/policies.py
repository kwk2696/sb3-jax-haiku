from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gym
import jax 
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax import nn

from sb3_jax.common.distributions import (
    Distribution,
    DiagGaussianDistributionFn,
    CategoricalDistributionFn, 
    make_proba_distribution,
)
from sb3_jax.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from sb3_jax.common.jax_layers import (
    init_weights,
    get_actor_critic_arch,
    create_mlp, 
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from sb3_jax.common.policies import BasePolicy
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.utils import is_vectorized_observation, obs_as_jnp, get_dummy_obs
from sb3_jax.common.norm_layers import BaseNormLayer


class BCPolicy(BasePolicy):
    """Policy class for BC."""
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None, 
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        log_std_init: float = 0.0,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        normalization_class: Type[BaseNormLayer] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 1, 
    ):
        super(BCPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalization_class=normalization_class,
            normalization_kwargs=normalization_kwargs,
            squash_output=squash_output,
        )
         
        if net_arch is None:
            net_arch = [64, 64]
        actor_arch = _ = get_actor_critic_arch

        self.net_arch = net_arch
        self.activation_fn = activation_fn 

        self.log_std_init = log_std_init
        self.dist_kwargs = dict()
        
        # Action distribution class
        self.action_dist_class, self.action_dist_fn = make_proba_distribution(action_space, use_sde=False)
        self._build(lr_schedule)
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super._get_constructor_parameters()
        
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn, 
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                normalization_class=normalization_class,
                normalization_kwargs=normalization_kwargs,
            )
        )
        return data

    def _build_actor(self) -> hk.Module:
        return create_mlp(
            output_dim=-1, 
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            squash_output=True
        )

    def _build(self, lr_schedule: Schedule) -> None:
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)

        if isinstance(self.action_dist_fn, DiagGaussianDistributionFn):
            action_dim = get_action_dim(self.action_space)
            self.dist_kwargs.update(dict(log_std_init=self.log_std_init))
        elif isinstance(self.action_dist_fn, CategoricalDistributionFn):
            action_dim = self.action_space.n
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist_fn}'.")

        def fn_actor(observation: jnp.ndarray): 
            features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
            action_net = self._build_actor()
            action = hk.Sequential(
                [features_extractor, action_net, self.action_dist_class(action_dim, **self.dist_kwargs)]
            )
            return action(observation)

        params, self.actor = hk.without_apply_rng(hk.transform(fn_actor))
        self.params = params(next(self.rng), get_dummy_obs(self.observation_space))
        self.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs)
        self.optimizer_state = self.optimizer.init(self.params)
    
    def forward(self, observation: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        return self._predict(observation, deterministic=deterministic)
    
    @partial(jax.jit, static_argnums=0)
    def _actor(self, observation: jnp.ndarray, params: hk.Params) -> jnp.ndarray:
        return self.actor(params, observation)

    def _predict(self, observation: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        observation = self.preprocess(observation)
        mean_actions, log_std = self._actor(observation, self.params)
        return self.action_dist_fn.get_actions(mean_actions, log_std, deterministic, next(self.rng))


MlpPolicy = BCPolicy
