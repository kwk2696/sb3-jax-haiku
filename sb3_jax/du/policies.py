from functools import partial
from dataclasses import dataclass
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gym
import jax
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax import random
from jax import nn

from sb3_jax.common.jax_layers import (
    init_weights,
    create_mlp,
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor
)
from sb3_jax.common.jax_utils import jax_print
from sb3_jax.common.policies import BasePolicy, register_policy
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.utils import get_dummy_obs
from sb3_jax.common.norm_layers import BaseNormLayer


@dataclass(frozen=True)
class DDPMCoefficients:
    alpha_t: jax.Array
    oneover_sqrta: jax.Array
    sqrt_beta_t: jax.Array
    alpha_bar_t: jax.Array
    sqrtab: jax.Array
    sqrtmab: jax.Array
    mab_over_sqrtmab_inv: jax.Array
    ma_over_sqrtmab_inv: jax.Array


class DiffusionBetaScheduler:
    supported_schedulers = ["linear", "cosine"]

    def __init__(self, beta1: float, beta2: float, total_denoise_steps: int, method: str = "cosine"):

        self.beta1 = beta1
        self.beta2 = beta2
        self.total_denoise_steps = total_denoise_steps
        self.method = method.lower()

        assert method in DiffusionBetaScheduler.supported_schedulers, f"{method} is not supported beta scheduler."

    def schedule(self) -> DDPMCoefficients:
        if self.method == "linear":
            beta_t = self.linear_schedule()
        elif self.method == "cosine":
            beta_t = self.cosine_schedule()
        else:
            raise NotImplementedError(f"{self.method} is not supported beta scheduler.")

        sqrt_beta_t = jnp.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = jnp.log(alpha_t)
        alpha_bar_t = jnp.exp(jnp.cumsum(log_alpha_t, axis=0))  # = alphas_cumprod

        sqrtab = jnp.sqrt(alpha_bar_t)
        oneover_sqrta = 1 / jnp.sqrt(alpha_t)

        sqrtmab = jnp.sqrt(1 - alpha_bar_t)
        mab_over_sqrtmab_inv = (1 - alpha_bar_t) / sqrtmab
        ma_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return DDPMCoefficients(
            alpha_t=alpha_t,
            oneover_sqrta=oneover_sqrta,
            sqrt_beta_t=sqrt_beta_t,
            alpha_bar_t=alpha_bar_t,
            sqrtab=sqrtab,
            sqrtmab=sqrtmab,
            mab_over_sqrtmab_inv=mab_over_sqrtmab_inv,
            ma_over_sqrtmab_inv=ma_over_sqrtmab_inv
        )

    def linear_schedule(self) -> jnp.ndarray:
        beta_t = (self.beta2 - self.beta1) \
                 * jnp.arange(-1, self.total_denoise_steps, dtype=jnp.float32) \
                 / (self.total_denoise_steps - 1) \
                 + self.beta1
        beta_t = beta_t.at[0].set(self.beta1)

        return beta_t

    def cosine_schedule(self):
        s = 8e-3
        timesteps = jnp.arange(self.total_denoise_steps + 1, dtype=jnp.float32)
        x = (((timesteps / self.total_denoise_steps) + s) / (1 + s)) * (jnp.pi / 2)
        f_t = jnp.cos(x) ** 2

        x_0 = (s / (1 + s)) * (jnp.pi / 2)
        f_0 = jnp.cos(x_0) ** 2

        alpha_bar_t = f_t / f_0

        beta_t = 1 - alpha_bar_t[1:] / alpha_bar_t[: -1]
        beta_t = jnp.clip(beta_t, a_min=0, a_max=0.999)

        return beta_t


class BaseDiffusionModel(hk.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        noise_dim: int, 
        net_arch: Optional[List[int]] = None, 
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn


class MLPDiffusionModel(BaseDiffusionModel):
    """
    MLP based diffusion, this model embeds x, y, t, before input into NN with residual.
    followed by: https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        y: jax.Array, # y for noise
        x: jax.Array, # x for observations (conditioned)
        t: jax.Array, # t for denoising timestep
    ):
        # embeddings
        y_embed = hk.Linear(self.embed_dim, name="ye1", **init_weights())(y)
        # NOTE: we do not use layer norm, because of the denosing chain ... (y_t keep changing) 
        # y_embed = hk.LayerNorm(axis=-1, name="yl", create_scale=True, create_offset=True)(y_embed)
        y_embed = self.activation_fn(y_embed)
        y_embed = hk.Linear(self.embed_dim, name="ye2", **init_weights())(y_embed)

        x_embed = hk.Linear(self.embed_dim, name="xe1", **init_weights())(x)
        # x_embed = hk.LayerNorm(-1, True, True, name="xl")(x_embed)
        x_embed = self.activation_fn(x_embed)
        x_embed = hk.Linear(self.embed_dim, name="xe2", **init_weights())(x_embed)

        t_embed = hk.Linear(self.embed_dim, name="te1", **init_weights())(t)
        t_embed = jnp.sin(t_embed)
        t_embed = hk.Linear(self.embed_dim, name="te2", **init_weights())(t_embed)
        
        nn_input = jnp.concatenate((y_embed, x_embed, t_embed), axis=-1)
        out = hk.Linear(self.hidden_dim, name="fc1", **init_weights())(nn_input)
        # out = hk.LayerNorm(-1, True, True, name="fcl1")(out)
        out = self.activation_fn(out)
        
        for i, size in enumerate(self.net_arch[1:]):
            nn_input = jnp.concatenate((out/1.414, x_embed, t_embed), axis=-1)
            _out = hk.Linear(self.hidden_dim, name=f"fc{i+1}", **init_weights())(nn_input)
            # _out = hk.LayerNorm(-1, True, True, name=f"fcl{i+1}")(_out)
            out = self.activation_fn(_out) + out/1.414 # residual connection & concat input again

        nn_input = jnp.concatenate((out, x_embed, t_embed), axis=-1)
        out = hk.Linear(self.noise_dim, name="fc_out", **init_weights())(nn_input)
        return out


class TimeSiren(hk.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def __call__(self, t):
        t_embed = hk.Linear(self.embed_dim, name="te1", with_bias=False, **init_weights())(t)
        t_embed = jnp.sin(t_embed)
        t_embed = hk.Linear(self.embed_dim, name="te2", **init_weights())(t_embed)
        return t_embed


class TFBlock(hk.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.tf_dim = embed_dim * n_heads
    
    def __call__(self, x: jax.Array) -> jax.Array:

        # [3, batch_size, tf_dim*3]
        qkvs = hk.Linear(self.tf_dim*3, name="qkvs", **init_weights())(x)
        # [3, batch_size, tf_dim]
        qs, ks, vs = self.split_qkv(qkvs)
        
        # [3, batch_size, tf_dim = embed_dim * n_heads]
        attn_a = hk.MultiHeadAttention(self.n_heads, self.embed_dim, name="attn", w_init=init_weights()['w_init'])(qs, ks, vs)

        # [3, batch_size, embed_dim]
        attn_b = hk.Linear(self.embed_dim, name="fc1", **init_weights())(attn_a)
        attn_b = attn_b / 1.414 + x / 1.414 # residual
        # [batch_size, embed_dim, 3]
        attn_b = attn_b.transpose(2, 1, 0).transpose(1, 0, 2)
        attn_b = hk.BatchNorm(True, True, 0.9, name="bn1")(attn_b, True)
        attn_b = attn_b.transpose(1, 0 ,2) .transpose(2, 1, 0)

        # [3, batch_size, embed_dim]
        attn_c = hk.Linear(self.embed_dim*4, name="fc2", **init_weights())(attn_b)
        attn_c = nn.gelu(attn_c)
        attn_c = hk.Linear(self.embed_dim, name="fc3", **init_weights())(attn_c)
        attn_c = attn_c / 1.414 + attn_b / 1.414
        # normalize
        attn_c = attn_c.transpose(2, 1, 0).transpose(1, 0, 2)
        attn_c = hk.BatchNorm(True, True, 0.9, name="bn2")(attn_c, True)
        attn_c = attn_c.transpose(1, 0, 2).transpose(2, 1, 0)
        return attn_c

    def split_qkv(self, qkv) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q = qkv[:,:,:self.tf_dim]
        k = qkv[:,:,self.tf_dim:2*self.tf_dim]
        v = qkv[:,:,2*self.tf_dim:]
        return (q, k, v)


class TFDiffusionModel(BaseDiffusionModel):
    """
    Transformer based diffusion, this model embeds x, y, t, befor input into transformer.
    """
    def __init__(self, n_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads

    def __call__(
        self,
        y: jax.Array, # y for noise
        x: jax.Array, # x for observations (conditioned)
        t: jax.Array, # t for denoising timestep
    ):
        batch_size = y.shape[0]

        # embeddings
        y_embed = hk.Linear(self.embed_dim, name="ye1", **init_weights())(y)
        y_embed = self.activation_fn(y_embed)
        y_embed = hk.Linear(self.embed_dim, name="ye2", **init_weights())(y_embed)

        x_embed = hk.Linear(self.embed_dim, name="xe1", **init_weights())(x)
        x_embed = self.activation_fn(x_embed)
        x_embed = hk.Linear(self.embed_dim, name="xe2", **init_weights())(x_embed)

        t_embed = hk.Linear(self.embed_dim, name="te1", **init_weights())(t)
        t_embed = jnp.sin(t_embed)
        t_embed = hk.Linear(self.embed_dim, name="te2", **init_weights())(t_embed)
        
        # positional encoding
        pos_embedder = TimeSiren(self.embed_dim)
        # [batch_size, embed_dim]
        y_embed += pos_embedder(jnp.zeros((batch_size, 1)) + 1.)
        x_embed += pos_embedder(jnp.zeros((batch_size, 1)) + 2.)
        t_embed += pos_embedder(jnp.zeros((batch_size, 1)) + 3.)
        
        # [3, batch_size, embed_dim]
        nn_input = jnp.concatenate((y_embed[None,:,:], x_embed[None,:,:], t_embed[None,:,:]), axis=0)
        
        for i, size in enumerate(self.net_arch):
            out = TFBlock(self.embed_dim, self.n_heads)(nn_input)
            nn_input = out

        # [batch_size, 3, embed_dim]
        out = out.transpose(1, 0, 2)
        # [batch_size, 3 * embed_dim]
        out = out.reshape(batch_size, -1)
        
        out = hk.Linear(self.embed_dim, name="fc_out1", **init_weights())(out)
        out = self.activation_fn(out)
        out = hk.Linear(self.noise_dim, name="fc_out2", **init_weights())(out)
        return out


class DiffusionModel(hk.Module):
    def __init__(
        self,
        du: hk.Module,
        n_denoise: int,
        ddpm_dict: DDPMCoefficients,
    ):
        super().__init__()
        self.du = du 

        self.n_denoise = n_denoise
        self.noise_dim = self.du.noise_dim
        
        # scheduler params
        self.alpha_t = ddpm_dict.alpha_t
        self.oneover_sqrta = ddpm_dict.oneover_sqrta
        self.sqrt_beta_t = ddpm_dict.sqrt_beta_t
        self.alpha_bar_t = ddpm_dict.alpha_bar_t
        self.sqrtab = ddpm_dict.sqrtab
        self.sqrtmab = ddpm_dict.sqrtmab
        self.ma_over_sqrtmab_inv = ddpm_dict.ma_over_sqrtmab_inv

    def __call__(
        self,
        y_t: jax.Array, # y for noise
        x: jax.Array,   # x for observations (conditioned)
        t: jax.Array,   # t for denoising timestep
        denoise: bool = False,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        n_batch = x.shape[0]
        
        # denoising chain
        if denoise:
            # sample initial noise, y_T ~ N(0, 1)
            y_i = jax.random.normal(hk.next_rng_key(), shape=(n_batch, self.noise_dim))
            # trace denoised outputs
            y_i_trace = dict()
            y_i_trace[self.n_denoise] = (y_i, None)
            # denoising chain
            for i in range(self.n_denoise, 0, -1):
                t_i = jnp.array([[i / self.n_denoise]])
                t_i = jnp.repeat(t_i, n_batch, axis=0)
                noise = random.normal(hk.next_rng_key(), shape=(n_batch, self.noise_dim)) if i > 1 else 0
                eps = self.du(y_i, x, t_i)
                y_i = self.oneover_sqrta[i] * (y_i - self.ma_over_sqrtmab_inv[i] * eps) + self.sqrt_beta_t[i] * noise
                y_i_trace[i-1] = (y_i, eps) # action, eps
            return y_i, y_i_trace
        return self.du(y_t, x, t)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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
        policy_type: str = 'mlp',
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
        
        self.policy_type = policy_type
        assert policy_type in DUPolicy.supported_policies, f"{policy_type} is not supported diffusion policy."
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
        if self.policy_type == "mlp":
            du = MLPDiffusionModel(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                noise_dim=self.noise_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
            )
        elif self.policy_type == "transformer":
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
            ddpm_dict=self.ddpm_dict
        )

    def _build(self, lr_schedule: Schedule) -> None:
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)

        def fn_actor(y_t: jax.Array, observation: jax.Array, t: jax.Array, denoise: bool):
            features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
            actor  = self._build_actor()
            return actor(y_t, observation, t, denoise)

        params, self.actor = hk.transform_with_state(fn_actor)
        dummy_y_t = random.normal(next(self.rng), shape=(1, self.noise_dim))
        dummy_t = jnp.array([[1 / self.n_denoise]])
        self.params, self.state = params(next(self.rng), dummy_y_t, get_dummy_obs(self.observation_space), dummy_t, denoise=False)
        self.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs)
        self.optimizer_state = self.optimizer.init(self.params)

    def forward(self, observation: jax.Array, deterministic: bool = False) -> jax.Array:
        action, _ = self._predict(observation, deterministic=deterministic)
        return action
    
    @partial(jax.jit, static_argnums=0)
    def _actor(
        self, 
        y_t: jax.Array,
        observation: jax.Array, 
        t: jax.Array,
        params: hk.Params, 
        state: hk.Params,
        rng=None
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return self.actor(params, state, rng, y_t, observation, t, denoise=False)
    
    # for denoise option
    @partial(jax.jit, static_argnums=0)
    def _actor_denoise(
        self,
        observation: jax.Array,
        params: hk.Params,
        state: hk.Params,
        rng=None
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return self.actor(params, state, rng, None, observation, None, denoise=True)

    def _predict(self, observation: jax.Array, deterministic: bool = False) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        n_batch = observation.shape[0]
        observation = self.preprocess(observation)
        (y_i, y_i_store), _ = self._actor_denoise(observation, self.params, self.state, next(self.rng))
        return y_i, y_i_store


MlpPolicy = DUPolicy

register_policy("MlpPolicy", MlpPolicy)
