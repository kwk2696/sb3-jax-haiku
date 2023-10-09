"""MLP/Transformer Diffusion Model"""
from dataclasses import dataclass
from typing import Any, Tuple, NamedTuple, List, Dict, Type, Optional, Callable

import jax
import jax.numpy as jnp
from jax import nn
import haiku as hk

from sb3_jax.common.jax_layers import init_weights, create_mlp


# ==================== Scheduler ==================== # 

@dataclass(frozen=True)
class DDPMCoefficients:
    beta_t: jax.Array
    alpha_t: jax.Array
    oneover_sqrta: jax.Array
    sqrt_beta_t: jax.Array
    alpha_bar_t: jax.Array
    alpha_bar_prev_t: jax.Array
    sqrtab: jax.Array
    sqrtmab: jax.Array
    mab_over_sqrtmab_inv: jax.Array
    ma_over_sqrtmab_inv: jax.Array
    posterior_beta: jax.Array
    posterior_log_beta: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array


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
        alpha_t = 1. - beta_t
        log_alpha_t = jnp.log(alpha_t)
        alpha_bar_t = jnp.exp(jnp.cumsum(log_alpha_t, axis=0))  # = alphas_cumprod
        alpha_bar_prev_t = jnp.concatenate([jnp.array([1]), alpha_bar_t[:-1]])

        # calcuations for forward q(x_t | x_{t-1}), backward q(x_{t-1} | x_t)
        sqrtab = jnp.sqrt(alpha_bar_t)
        oneover_sqrta = 1. / jnp.sqrt(alpha_t)
        sqrtmab = jnp.sqrt(1. - alpha_bar_t)
        mab_over_sqrtmab_inv = (1. - alpha_bar_t) / sqrtmab
        ma_over_sqrtmab_inv = (1. - alpha_t) / sqrtmab

        # calculations for foward posterior q(x_{t-1} | x_t, x_0)
        posterior_beta = beta_t * (1. - alpha_bar_prev_t) / (1. - alpha_bar_t)
        posterior_log_beta = jnp.log(jnp.clip(posterior_beta, a_min=1e-20)) # log calcuation clipped becaused 0 at the beginning
        posterior_mean_coef1 = beta_t * jnp.sqrt(alpha_bar_prev_t) / (1. - alpha_bar_t)
        posterior_mean_coef2 = jnp.sqrt(alpha_t) * (1. - alpha_bar_prev_t) / (1. - alpha_bar_t)

        return DDPMCoefficients(
            beta_t=beta_t,
            alpha_t=alpha_t,
            oneover_sqrta=oneover_sqrta,
            sqrt_beta_t=sqrt_beta_t,
            alpha_bar_t=alpha_bar_t,
            alpha_bar_prev_t=alpha_bar_prev_t,
            sqrtab=sqrtab,
            sqrtmab=sqrtmab,
            mab_over_sqrtmab_inv=mab_over_sqrtmab_inv,
            ma_over_sqrtmab_inv=ma_over_sqrtmab_inv,
            posterior_beta=posterior_beta,
            posterior_log_beta=posterior_log_beta,
            posterior_mean_coef1=posterior_mean_coef1,
            posterior_mean_coef2=posterior_mean_coef2,
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

# ==================== Diffusion Models  ==================== # 

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
        denoise_type: str = 'ddpm',
    ):
        super().__init__()
        self.du = du 

        self.n_denoise = n_denoise
        self.noise_dim = self.du.noise_dim if isinstance(self.du.noise_dim, tuple) else (self.du.noise_dim,)
        self.denoise_type = denoise_type
        
        # scheduler params
        self.alpha_t = ddpm_dict.alpha_t
        self.oneover_sqrta = ddpm_dict.oneover_sqrta
        self.sqrt_beta_t = ddpm_dict.sqrt_beta_t
        self.alpha_bar_t = ddpm_dict.alpha_bar_t
        self.alpha_bar_prev_t = ddpm_dict.alpha_bar_prev_t
        self.sqrtab = ddpm_dict.sqrtab
        self.sqrtmab = ddpm_dict.sqrtmab
        self.ma_over_sqrtmab_inv = ddpm_dict.ma_over_sqrtmab_inv
        self.posterior_log_beta = ddpm_dict.posterior_log_beta
        self.posterior_mean_coef1 = ddpm_dict.posterior_mean_coef1
        self.posterior_mean_coef2 = ddpm_dict.posterior_mean_coef2

    def __call__(
        self,
        y_t: jax.Array, # y for noise
        x: jax.Array,   # x for observations (conditioned)
        t: jax.Array,   # t for denoising timestep
        denoise: bool = False,
        deterministic: bool = False, 
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        n_batch = x.shape[0]
        
        # denoising chain
        if denoise:
            # sample initial noise, y_T ~ N(0, 1)
            y_i = jax.random.normal(hk.next_rng_key(), shape=(n_batch,) + self.noise_dim)
            # trace denoised outputs
            y_i_trace = dict()
            y_i_trace[self.n_denoise] = (y_i, None)
            # denoising chain
            for i in range(self.n_denoise, 0, -1):
                t_i = jnp.array([[i / self.n_denoise]])
                t_i = jnp.repeat(t_i, n_batch, axis=0)
                noise = jax.random.normal(hk.next_rng_key(), shape=(n_batch,) + self.noise_dim) if (i > 1 and not deterministic) else 0.
                eps = self.du(y_i, x, t_i)
                
                # ddpm generative process
                if self.denoise_type == 'ddpm':
                    y_i = self.oneover_sqrta[i] * (y_i - self.ma_over_sqrtmab_inv[i] * eps) + self.sqrt_beta_t[i] * noise
                # TODO: ddim generative process, need to implement noise part
                elif self.denoise_type == 'ddim':
                    # prediction of y_0
                    pred_y_0 = (y_i - self.sqrtmab[i] * eps) / self.sqrtab[i]
                    y_i = jnp.sqrt(self.alpha_bar_prev_t[i]) * pred_y_0 + jnp.sqrt(1. - alpha_bar_prev_t[i]) * eps 

                y_i_trace[i-1] = (y_i, eps) # action, eps
            return y_i, y_i_trace
        return self.du(y_t, x, t)
