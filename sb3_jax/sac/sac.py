from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import copy

import gym
import jax
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax import nn

from sb3_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from sb3_jax.sac.policies import SACPolicy
from sb3_jax.common.buffers import ReplayBuffer
from sb3_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sb3_jax.common.utils import polyak_update
from sb3_jax.common.jax_utils import jit_optimize, stop_grad, jax_print


class LogEntropyCoef(hk.Module):
    def __init__(self, init_value: float = 1.0):
        super(LogEntropyCoef, self).__init__()
        self.init_value = init_value
        
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = hk.get_parameter("log_ent_coef", (1,), init=hk.initializers.Constant(jnp.log(self.init_value)))
        return log_ent_coef


class SAC(OffPolicyAlgorithm):
    """Soft-Actor-Critic algorithm (SAC)."""
    
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000, # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional = None, # not implemented
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto", 
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log : Optional[str] = None,
        wandb_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = 1,
        _init_setup_model: bool = True,
    ):
        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau, 
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            wandb_log=wandb_log,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        if self.target_entropy == "auto":
            # automatically set target entropy
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)
        
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            
            def fn_log_ent_coef():
                log_ent_coef = LogEntropyCoef(init_value)
                return log_ent_coef()
            params, self.log_ent_coef = hk.without_apply_rng(hk.transform(fn_log_ent_coef))
            self.ent_coef_params = params(next(self.actor.rng))
            self.ent_coef_optimizer = optax.adam(learning_rate=self.lr_schedule(1))
            self.ent_coef_optimizer_state = self.ent_coef_optimizer.init(self.ent_coef_params)
        else:
            self.ent_coef = float(self.ent_coef) 
    
    @partial(jax.jit, static_argnums=0)
    def _log_ent_coef(self, params: hk.Params) -> jnp.ndarray:
        return self.log_ent_coef(params)

    def _create_aliases(self)-> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            observations = self.policy.preprocess(replay_data.observations, training=True)
            actions = replay_data.actions
            rewards = replay_data.rewards
            next_observations = self.policy.preprocess(replay_data.next_observations, training=True)
            dones = replay_data.dones

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                raise NotImplementedError
            
            
            target_q_values = jax.lax.stop_gradient(
                self._get_td_target(self.actor.params, self.critic_target.params, self.ent_coef_params,
                    next_observations, rewards, dones, next(self.actor.rng))
            )

            # Optimize critic
            self.critic.optimizer_state, self.critic.params, critic_loss, critic_info = jit_optimize(
                self._loss_critic,
                self.critic.optimizer,
                self.critic.optimizer_state,
                self.critic.params,
                max_grad_norm=None,
                observations=observations,
                actions=actions,
                target_q_values=target_q_values,
                rng=next(self.actor.rng)
            )
            critic_losses.append(critic_info['critic_loss'])

            # Optimize actor
            self.actor.optimizer_state, self.actor.params, actor_loss, actor_info = jit_optimize(
                self._loss_actor,
                self.actor.optimizer,
                self.actor.optimizer_state,
                self.actor.params,
                max_grad_norm=None,
                critic_params=self.critic.params,
                ent_coef_params=self.ent_coef_params,
                observations=observations,
                rng=next(self.actor.rng),
            )
            actor_losses.append(actor_info['actor_loss'])
            # Optimize ent_coef
            if self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer_state, self.ent_coef_params, ent_coef_loss, ent_coef_info = jit_optimize(
                    self._loss_ent_coef,
                    self.ent_coef_optimizer,
                    self.ent_coef_optimizer_state,
                    self.ent_coef_params,
                    max_grad_norm=None,
                    log_prob=actor_info['log_prob'],
                    rng=next(self.actor.rng),
                )
                ent_coef = ent_coef_info['ent_coef']
                ent_coef_losses.append(ent_coef_info['ent_coef_loss'])
            else: 
                ent_coef = self.ent_coef
            ent_coefs.append(ent_coef)

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                self.critic_target.params = polyak_update(self.critic.params, self.critic_target.params, self.tau)

        self._n_updates += gradient_steps
        
        self.logger.record("train/log_prob", actor_info['log_prob'].mean())
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/q_target", np.mean(target_q_values))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        
        if self.wandb_log is not None:
            wandb_log({
                "train/n_updates", self._n_updates,
                "train/actor_loss", np.mean(actor_losses),
                "train/critic_loss", np.mean(critic_losses),
                "train/ent_coef_loss", np.mean(ent_coef_losses),
                "train/ent_coef", np.mean(ent_coefs),
                "train/log_prob", actor_info['log_prob'].mean()
            })

    @partial(jax.jit, static_argnums=0)
    def _loss_ent_coef(
        self,
        params: hk.Params,
        log_prob: jnp.ndarray,
        rng=None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        
        log_prob = jax.lax.stop_gradient(log_prob) 

        log_ent_coef = self._log_ent_coef(params)
        ent_coef_loss = (log_ent_coef * (-log_prob -self.target_entropy)).mean()
        
        info = {
            'ent_coef_loss': ent_coef_loss,
            'ent_coef': jnp.exp(log_ent_coef)
        }
        return ent_coef_loss, info
    
    @partial(jax.jit, static_argnums=0)
    def _get_td_target(
        self,
        actor_params: hk.Params,
        critic_target_params: hk.Params,
        ent_coef_params: hk.Params,
        next_observations: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        rng=None,
    ):
        next_actions, next_log_prob = self.actor.action_log_prob(next_observations, actor_params, rng=rng)
        next_q_values = jnp.concatenate(self.critic_target._critic(next_observations, next_actions, critic_target_params), axis=1)
        next_q_values = jnp.min(next_q_values, axis=1, keepdims=True)
        log_ent_coef = self._log_ent_coef(ent_coef_params)
        # add entropy term
        next_q_values = next_q_values - jnp.exp(log_ent_coef) * next_log_prob.reshape(-1, 1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        return target_q_values

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params: hk.Params,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        target_q_values: jnp.ndarray, 
        rng=None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        
        # compute current Q-value 
        current_q_values = self.critic._critic(observations, actions, params)
        critic_loss = 0.5 * jnp.sum(jnp.array([jnp.square(current_q - target_q_values).mean() for current_q in current_q_values]))
        
        info = {
            'critic_loss': critic_loss,
        }
        return critic_loss, info
    
    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params: hk.Params,
        critic_params: hk.Params,
        ent_coef_params: hk.Params,
        observations: jnp.ndarray,
        rng=None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

        # sample action
        actions_pi, log_prob = self.actor.action_log_prob(observations, params, rng=rng)
        log_prob = log_prob.reshape(-1, 1)
        
        # compuate q value
        q_values_pi = jnp.concatenate(self.critic._critic(observations, actions_pi, critic_params), axis=1)
        min_qf_pi = jnp.min(q_values_pi, axis=1, keepdims=True)

        # get temperature
        log_ent_coef = jax.lax.stop_gradient(
            self._log_ent_coef(ent_coef_params)
        ) 

        actor_loss = jnp.mean(jnp.exp(log_ent_coef) * log_prob - min_qf_pi)
        
        info = {
            'actor_loss': actor_loss,
            'log_prob': log_prob,
        }
        return actor_loss, info

    def _save_jax_params(self) -> Dict[str, hk.Params]:
        params_dict = {}
        params_dict['actor'] = self.actor.params
        params_dict['critic'] = self.critic.params
        params_dict['critic_target'] = self.critic_target.params
        params_dict['log_ent_coef'] = self.ent_coef_params
        return params_dict

    def _load_jax_params(self, params: Dict[str, hk.Params]) -> None:
        self.actor.params = params['actor']
        self.critic.params = params['critic']
        self.critic_target.params = params['critic_target']
        self.ent_coef_params = params['log_ent_coef']

    def _save_norm_layer(self, path: str) -> None:
        if self.actor.normalization_class is not None:
            self.actor.normalization_layer.save(path)

    def _load_norm_layer(self, path: str) -> None:
        if self.actor.normalization_class is not None:
            self.actor.normalization_class.load(path)
