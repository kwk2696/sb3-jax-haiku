from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from copy import deepcopy

import gym
import jax
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax import nn

from sb3_jax.common.distributions import SquashedDiagGaussianDistributionFn, SquashedDiagGaussianDistribution
from sb3_jax.common.policies import BasePolicy, ContinuousCritic, register_policy
from sb3_jax.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from sb3_jax.common.jax_layers import (
    init_weights,
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.utils import is_vectorized_observation, obs_as_jnp, get_dummy_obs
from sb3_jax.common.norm_layers import BaseNormLayer

# CAP the standard deviation of the actor 
LOG_STD_MAX = 2 
LOG_STD_MIN = -20


class Actor(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        
        actor_kwargs = dict()
        if self.use_sde:
            raise NotImplementedError("sde arch is not implemented in jax-haiku yet") 
        else: 
            self.action_dist_class, self.action_dist_fn = SquashedDiagGaussianDistribution, SquashedDiagGaussianDistributionFn()
            actor_kwargs.update({
                "log_std_init": log_std_init,
            })
        self._build(dict())

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data
    
    def _build_actor(self) -> hk.Module:
        return create_mlp(
            output_dim=-1, 
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            squash_output=False,
        )

    def _build(self, actor_kwargs: Dict[str, Any]) -> None:
        if isinstance(self.action_dist_fn, SquashedDiagGaussianDistributionFn):
            action_dim = get_action_dim(self.action_space)
            actor_kwargs.update(dict(log_std_init=self.log_std_init))
        else:
            raise NotImplementedError("Unsupported distribution '{self.action_dist_fn}'.")
        
        def fn_actor(observation: jnp.ndarray):
            action_net = self._build_actor()
            action = hk.Sequential(
                [action_net, self.action_dist_class(action_dim, **actor_kwargs)]
            )
            return action(observation)

        params, self.actor = hk.without_apply_rng(hk.transform(fn_actor))
        self.params = params(next(self.rng), get_dummy_obs(self.observation_space))    

    def forward(self, observation: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        mean_actions, log_std = self._actor(observation, self.params)
        actions = self.action_dist_fn.get_actions(mean_actions, log_std, deterministic, next(self.rng))
        return actions
    
    @partial(jax.jit, static_argnums=0)
    def action_log_prob(
        self, 
        observation: jnp.ndarray, 
        params: jnp.ndarray = None,
        rng=None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ return actions and corresponding log prob"""
        mean_actions, log_std = self._actor(observation, params)
        return self.action_dist_fn.log_prob_from_params(mean_actions, log_std, rng)
            
    @partial(jax.jit, static_argnums=0)
    def _actor(self, observation: jnp.ndarray, params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ return: mean, log_std """
        return self.actor(params, observation)
        
    def _predict(self, observation: jnp.ndarray, deterministic: bool = False) -> Tuple[jnp.ndarray, Optional[Dict[str, Any]]]:
        mean_actions, log_std = self._actor(observation, self.params)
        return self.action_dist_fn.get_actions(mean_actions, log_std, deterministic, next(self.rng)), None


class SACPolicy(BasePolicy): 
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        normalization_class: Type[BaseNormLayer] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 1, 
    ): 
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalization_class=normalization_class,
            normalization_kwargs=normalization_kwargs,
            squash_output=True,
            seed=seed,
        )

        if net_arch is None:
            net_arch = [256, 256]
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed")
        
        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None 
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                normalization_class=normalization_class,
                normalization_kwargs=normalization_kwargs,
            )
        )
        return data
    
    def _build(self, lr_schedule: Schedule) -> None:
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)

        # make actor
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs) 
        self.actor.optimizer_state = self.actor.optimizer.init(self.actor.params)

        # make critic
        self.critic = self.make_critic()
        self.critic.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs)
        self.critic.optimizer_state = self.critic.optimizer.init(self.critic.params)
        
        # make critic target
        self.critic_target = self.make_critic()
        self.critic_target.params = deepcopy(self.critic.params)

    def make_actor(self,) -> Actor:
        return Actor(**self.actor_kwargs)

    def make_critic(self,) -> ContinuousCritic:
        return ContinuousCritic(**self.critic_kwargs)
    
    def forward(self, observation: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        observation = self.preprocess(observation)
        action, _ = self._predict(observtion, deterministic=deterministic)
        return action

    def _predict(self, observation: jnp.ndarray, deterministic: bool = False) -> Tuple[jnp.ndarray, Optional[Dict[str, Any]]]:
        observation = self.preprocess(observation)
        return self.actor._predict(observation, deterministic)

MlpPolicy = SACPolicy

register_policy("MlpPolicy", SACPolicy)
