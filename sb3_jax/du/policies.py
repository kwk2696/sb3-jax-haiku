""" Diffusion Model based Policy """
from functools import partial
from typing import Any, Tuple, Union, List, Dict, Type, Optional, Callable

import gym
import jax
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax import nn

from sb3_jax.du.models import (
    DDPMCoefficients,
    DiffusionBetaScheduler,
    MLPDiffusionModel,
    TFDiffusionModel,
    DiffusionModel,
)
from sb3_jax.common.jax_layers import BaseFeaturesExtractor, FlattenExtractor
from sb3_jax.common.policies import BasePolicy, register_policy
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.utils import get_dummy_obs
from sb3_jax.common.norm_layers import BaseNormLayer


class DUPolicy(BasePolicy):
    """Policy class for Diffuser."""
    supported_policies = ["mlp", "transformer"]

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None, 
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu,
        policy_type: str = 'ddpm_mlp',
        predict_epsilon: bool = True,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_heads: int = 4, # for transformer
        n_denoise: int = 50,
        # variance scheduler
        beta_scheduler: str = 'linear',
        beta: Tuple[float,...] = (1e-4, 0.02),
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
        super(DUPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalization_class=normalization_class,
            normalization_kwargs=normalization_kwargs,
            squash_output=squash_output,
            seed=seed,
        )
        
        if net_arch is None:
            net_arch = [64, 64]
        
        self.denoise_type, self.net_type = policy_type.split('_')
        assert self.net_type in DUPolicy.supported_policies, f"{self.net_type} is not supported diffusion policy."
        self.predict_epsilon = predict_epsilon

        self.net_arch = net_arch
        self.noise_dim = action_space.shape[-1] # noise dim is size of action
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_denoise = n_denoise
        self.activation_fn = activation_fn

        # ddpm scheduler configs
        self.ddpm_dict = DiffusionBetaScheduler(beta[0], beta[1], self.n_denoise, method=beta_scheduler).schedule()

        self._build(lr_schedule) 
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                observation_space=self.observation_space,
                action_space=self.action_space,
                denoise_type=self.denoise_type,
                net_type=self.net_type,
                predict_epsilon=self.predict_epsilon,
                noise_dim=self.noise_dim,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                n_denoise=self.n_denoise,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                normalization_class=self.normalization_class,
                normalization_kwargs=self.normalization_kwargs,
            )
        )
        return data

    def _build_actor(self) -> hk.Module:
        if self.net_type == "mlp":
            du = MLPDiffusionModel(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                noise_dim=self.noise_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
            )
        elif self.net_type == "transformer":
            du = TFDiffusionModel(
                n_heads=self.n_heads,
                embed_dim=self.embed_dim, 
                hidden_dim=self.hidden_dim,
                noise_dim=self.noise_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
            )
        else:
            raise NotImplementedError

        return DiffusionModel(
            du=du,
            n_denoise=self.n_denoise,
            ddpm_dict=self.ddpm_dict,
            denoise_type=self.denoise_type,
            predict_epsilon=self.predict_epsilon,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)

        def fn_actor(y_t: jax.Array, obs: jax.Array, t: jax.Array, denoise: bool, deterministic: bool):
            actor  = self._build_actor()
            return actor(y_t, obs, t, denoise, deterministic)

        params, self.actor = hk.transform_with_state(fn_actor)
        dummy_y_t = jax.random.normal(next(self.rng), shape=(1, self.noise_dim))
        dummy_t = jnp.array([[1 / self.n_denoise]])
        self.params, self.state = params(
            next(self.rng), dummy_y_t, get_dummy_obs(self.observation_space), dummy_t, denoise=False, deterministic=False)
        self.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs)
        self.optimizer_state = self.optimizer.init(self.params)

    def forward(self, obs: jax.Array, deterministic: bool = False) -> jax.Array:
        action, _ = self._predict(obs, deterministic=deterministic)
        return action
    
    @partial(jax.jit, static_argnums=0)
    def _actor(
        self, 
        y_t: jax.Array,
        obs: jax.Array, 
        t: jax.Array,
        params: hk.Params, 
        state: hk.Params,
        rng=None
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return self.actor(params, state, rng, y_t, obs, t, denoise=False, deterministic=False)
    
    # for denoise
    @partial(jax.jit, static_argnums=(0,2))
    def _actor_denoise(
        self,
        obs: jax.Array,
        deterministic: bool,
        params: hk.Params,
        state: hk.Params,
        rng=None
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return self.actor(params, state, rng, None, obs, None, denoise=True, deterministic=deterministic)

    def _predict(self, obs: jax.Array, deterministic: bool = False) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        obs = self.preprocess(obs)
        (y_i, y_i_store), _ = self._actor_denoise(obs, deterministic, self.params, self.state, next(self.rng))
        return y_i, y_i_store


MlpPolicy = DUPolicy

register_policy("MlpPolicy", MlpPolicy)
