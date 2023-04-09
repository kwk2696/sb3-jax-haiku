from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import wandb
import gym
import jax
import numpy as np
import haiku as hk
import jax.numpy as jnp

from sb3_jax.common.offline_algorithm import OfflineAlgorithm
from sb3_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from sb3_jax.common.buffers import BaseBuffer, OfflineBuffer, ReplayBuffer
from sb3_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sb3_jax.common.jax_utils import jit_optimize
from sb3_jax.common.policies import BasePolicy
from sb3_jax.bc.policies import BCPolicy


class BC(OfflineAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[BCPolicy]],
        env: Union[GymEnv, str],
        replay_buffer: Type[BaseBuffer] = OfflineBuffer,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256, 
        gamma: float = 0.99,
        gradient_steps: int = 1,
        ent_coef: float = 0.0,
        loss_type: str = 'mse', # loss type: mse/neglogp 
        create_eval_env: bool = False,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = 1,
        _init_setup_model: bool = True,
    ): 
        assert loss_type in ['mse', 'neglogp'], "Loss type should be one of mse and neglogp" 
        super(BC, self).__init__(
            policy,
            env,
            replay_buffer=replay_buffer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            gradient_steps=gradient_steps,
            tensorboard_log=tensorboard_log,
            wandb_log=wandb_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(gym.spaces.Discrete, gym.spaces.Box),
            support_multi_env=False,
        )
         
        self.loss_type = loss_type
        self.ent_coef = ent_coef
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(BC, self)._setup_model()
    
    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """Update policy using samples from replay buffer."""

        actor_losses, entropys, neglogps = [], [], []
        
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            
            observations = self.policy.preprocess(replay_data.observations, training=True)
            actions = replay_data.actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.squeeze()

            self.policy.optimizer_state, self.policy.params, loss, info = jit_optimize(
                self._loss,
                self.policy.optimizer,
                self.policy.optimizer_state,
                self.policy.params,
                None, 
                observations=observations,
                actions=actions,
            )

            actor_losses.append(np.array(info["actor_loss"]))
            entropys.append(np.array(info["entropy"]))
            neglogps.append(np.array(info["neglogp"]))

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/entropy", np.mean(entropys))
        self.logger.record("train/neglogp", np.mean(neglogps))
        
        # wandb log
        if self.wandb_log is not None:
            wandb.log({
                "train/n_updates": self._n_updates,
                "train/actor_loss": np.mean(actor_losses),
                "train/entropy": np.mean(entropys),
                "train/neglogp": np.mean(neglogps),
            })

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

        
        if self.loss_type == 'neglogp':
            mean_actions, log_std = self.policy._actor(observations, params)
            log_prob = self.policy.action_dist_fn.log_prob(actions, mean_actions, log_std)
            log_prob = jnp.mean(log_prob)
            entropy = self.policy.action_dist_fn.entropy(mean_actions, log_std)
            entropy = jnp.mean(entropy)
            ent_loss = -self.ent_coef * entropy
            neglogp = -log_prob
            loss = neglogp + ent_loss
        elif self.loss_type == 'mse':
            mean_actions = self.policy._actor(observations, params)
            neglogp, entropy = 0., 0.
            loss = jnp.mean(jnp.square(mean_actions - actions)) 

        info = {
            'neglogp': neglogp,
            'entropy': entropy,
            'actor_loss': loss,
        }
        return loss, info
 
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "BC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":
        
        # wandb configs
        self.wandb_config = dict( 
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            gradient_steps=self.gradient_steps,
            ent_coef=self.ent_coef,
            loss_type=self.loss_type,
        )
        self.wandb_config.update(self.policy._get_constructor_parameters())

        return super(BC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _save_jax_params(self) -> Dict[str, hk.Params]:
        params_dict = {}
        params_dict['policy'] = self.policy.params 
        return params_dict

    def _load_jax_params(self, params: Dict[str, hk.Params]) -> None:
        self.policy.params = params['policy']

    def _save_norm_layer(self, path: str) -> None:
        if self.policy.normalization_class is not None:
            self.policy.normalization_layer.save(path)
    
    def _load_norm_layer(self, path: str) -> None:
        if self.policy.normalization_class is not None:
            self.policy.normalization_layer = self.policy.normalization_layer.load(path)


class OnlineBC(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[BCPolicy]],
        env: Union[GymEnv, str],
        replay_buffer: Type[BaseBuffer] = ReplayBuffer,
        policy_base: Type[BasePolicy] = BCPolicy, 
        buffer_size: int = 1_000_000, # 1e6
        learning_starts: int = 100,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256, 
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, 'step'),
        gradient_steps: int = 1,
        ent_coef: float = 0.0,
        loss_type: str = 'mse', # loss type: mse/negplgp
        replay_buffer_class: Optional[BaseBuffer] = ReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        create_eval_env: bool = False,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ): 
        super(OnlineBC, self).__init__(
            policy,
            env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            wandb_log=wandb_log,
            verbose=verbose,
            support_multi_env=False,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(gym.spaces.Discrete, gym.spaces.Box),
        )
        
        self.loss_type = loss_type
        self.ent_coef = ent_coef
        if _init_setup_model:
            self._setup_model()
    
        # replace replay buffer for given one
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer

    def _setup_model(self) -> None:
        super(OnlineBC, self)._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """Update policy using samples from replay buffer."""
 
        actor_losses, entropys, neglogps = [], [], []
        
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            
            observations = self.policy.preprocess(replay_data.observations, training=True)
            actions = replay_data.actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.squeeze()

            self.policy.optimizer_state, self.policy.params, loss, info = jit_optimize(
                self._loss,
                self.policy.optimizer,
                self.policy.optimizer_state,
                self.policy.params,
                None, 
                observations=observations,
                actions=actions,
            )

            actor_losses.append(np.array(info["actor_loss"]))
            entropys.append(np.array(info["entropy"]))
            neglogps.append(np.array(info["neglogp"]))
        
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/entropy", np.mean(entropys))
        self.logger.record("train/neglogp", np.mean(neglogps))

        # wandb log
        if self.wandb_log is not None:
            wandb.log({
                "train/n_updates": self._n_updates,
                "train/actor_loss": np.mean(actor_losses),
                "train/entropy": np.mean(entropys),
                "train/neglogp": np.mean(neglogps),
            })

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

        if self.loss_type == 'neglogp':
            mean_actions, log_std = self.policy._actor(observations, params)
            log_prob = self.policy.action_dist_fn.log_prob(actions, mean_actions, log_std)
            log_prob = jnp.mean(log_prob)
            entropy = self.policy.action_dist_fn.entropy(mean_actions, log_std)
            entropy = jnp.mean(entropy)
            ent_loss = -self.ent_coef * entropy
            neglogp = -log_prob
            loss = neglogp + ent_loss
        elif self.loss_type == 'mse':
            mean_actions = self.policy._actor(observations, params)
            neglogp, entropy = 0., 0.
            loss = jnp.mean(jnp.square(mean_actions - actions)) 

        info = {
            'neglogp': neglogp,
            'entropy': entropy,
            'actor_loss': loss,
        }
        return loss, info 

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1, 
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnlineBC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        # wandb configs
        self.wandb_config = dict( 
            learning_rate=self.learning_rate,
            learning_starts=self.learning_starts,
            tau=self.tau,
            batch_size=self.batch_size,
            gamma=self.gamma,
            gradient_steps=self.gradient_steps,
            ent_coef=self.ent_coef,
            loss_type=self.loss_type,
        )
        self.wandb_config.update(self.policy._get_constructor_parameters())

        return super(OnlineBC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _save_jax_params(self) -> Dict[str, hk.Params]:
        params_dict = {}
        params_dict['policy'] = self.policy.params
        return params_dict
 
    def _load_jax_params(self, params: Dict[str, hk.Params]) -> None:
        self.policy.params = params['policy']

    def _save_norm_layer(self, path: str) -> None:
        if self.policy.normalization_class is not None:
            self.policy.normalization_layer.save(path)
    
    def _load_norm_layer(self, path: str) -> None:
        if self.policy.normalization_class is not None:
            self.policy.normalization_layer = self.policy.normalization_layer.load(path) 
