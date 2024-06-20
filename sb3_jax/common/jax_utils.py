from functools import partial
from typing import Any, Callable, Tuple, Dict

import jax
import optax
import jax.numpy as jnp
import haiku as hk
from optax._src.base import Schedule

@partial(jax.jit, static_argnums=(0, 1, 4))
def jit_optimize(
    loss_function: Callable[..., Any],
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    params: hk.Params,
    max_grad_norm: float, 
    *args,
    **kwargs,
) -> Tuple[Any, hk.Params, jax.Array, Any]:
    (loss, aux), grad = jax.value_and_grad(loss_function, has_aux=True)(
        params,
        *args, 
        **kwargs,
    )
    if max_grad_norm is not None:
        grad = clip_gradient_norm(grad, max_grad_norm)
    update, optimizer_state = optimizer.update(grad, optimizer_state, params)
    params = optax.apply_updates(params, update)
    return optimizer_state, params, loss, aux


@partial(jax.jit, static_argnums=(0, 1, 5))
def jit_optimize_with_state(
    loss_function: Callable[..., Any],
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    params: hk.Params,
    state: hk.Params, 
    max_grad_norm: float, 
    *args,
    **kwargs,
) -> Tuple[Any, hk.Params, jax.Array, Any]:
    (loss, (new_state, aux)), grad = jax.value_and_grad(loss_function, has_aux=True)(
        params,
        state,
        *args, 
        **kwargs,
    )
    if max_grad_norm is not None:
        grad = clip_gradient_norm(grad, max_grad_norm)
    update, optimizer_state = optimizer.update(grad, optimizer_state, params)
    params = optax.apply_updates(params, update)
    return optimizer_state, params, new_state, loss, aux


def stop_grad(x: jax.Array):
    return jax.lax.stop_gradient(x)


@jax.jit
def clip_gradient(
    grad: Dict,
    max_value: float,
) -> Any:
    return jax.tree_map(lambda g: jnp.clip(g, -max_value, max_value), grad)


@jax.jit
def clip_gradient_norm(
    grad: Any,
    max_grad_norm: float,
) -> Any:
    def _clip_gardient_norm(g):
        clip_coef = max_grad_norm / (jax.lax.stop_gradient(jnp.linalg.norm(g)) + 1e-6)
        clip_coef = jnp.clip(clip_coef, a_max=1.0)
        return g * clip_coef

    return jax.tree_map(lambda g: _clip_gardient_norm(g), grad)


@jax.jit
def polyak_update(
    params: hk.Params,
    params_target: hk.Params,
    tau: float,
) -> hk.Params:
    """Perform a Polyak average update on target_params using params."""
    return jax.tree_multimap(lambda t, s: (1 - tau) * t + tau * s, params_target, params)


@jax.jit
def explained_variance(
    y_pred: jax.Array,
    y_true: jax.Array,
) -> jax.Array:
    """Computes fraction of variance that ypred expalined about y."""
    var_y = jnp.var(y_true)
    return 1 - jnp.var(y_true - y_pred) / var_y


@jax.jit
def kl_divergence(
    p_mean: jax.Array,
    p_std: jax.Array,
    q_mean: jax.Array = None,
    q_std: jax.Array = None,
) -> jax.Array:
    if q_mean is None:
        return 0.5 * jnp.mean(-jnp.log(jnp.square(p_mean)) - 1. + jnp.square(p_std) + jnp.square(p_mean), axis=-1)
    return jnp.mean((jnp.log(q_std) - jnp.log(p_std)) + \
        ((jnp.square(p_std) + jnp.square(q_mean - p_mean)) / (2 * jnp.square(q_std))) - 0.5, axis=-1)


# Learning Schedulers for Jax
def warmup_scheduler(init_value: float, warmup_steps: int) -> Schedule:
    def schedule(count):
        return jnp.minimum((count + 1)/warmup_steps, 1.) * init_value
    return schedule


def jax_print(x: Any):
    jax.debug.print("{x}", x=x)
