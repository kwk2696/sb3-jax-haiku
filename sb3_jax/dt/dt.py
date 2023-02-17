import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from sb3_jax.common.offline_algorithm import OfflineAlgorithm
from sb3_jax.common.buffers import BaseBuffer, TrajectoryBuffer
from sb3_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sb3_jax.common.jax_utils import jit_optimize
from sb3_jax.dt.policies import DTPolicy


class DT(OfflineAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[DTPolicy]],
        env: Union[GymEnv, str],
        replay_buffer: Type[BaseBuffer] = TrajectoryBuffer,  
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        gamma: float = 0.99,
        gradient_steps: int = 1,
        create_eval_env: bool = False,
        tensorboard_log: Optional[str] = None, 
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):
        super(DT, self).__init__(
            policy,
            env,
            replay_buffer=replay_buffer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            gradient_steps=gradient_steps,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=(gym.spaces.Discrete, gym.spaces.Box),
            support_multi_env=False,
        )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DT, self)._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        
        actor_losses = []

        for gradient_step in range(gradient_steps):
            
            # TODO: need faster replay buffer
            #start = time.time()
            replay_data = self.replay_buffer.sample(batch_size)
            #end =  time.time()
            #print("buff", end - start)
            # TODO: do we need preprocessing the other inputs? 
            observations = self.policy.preprocess(replay_data.observations, training=True)
            actions = replay_data.actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.squeeze()
            rewards = replay_data.rewards
            dones = replay_data.dones
            returns_to_go = replay_data.returns_to_go
            timesteps = replay_data.timesteps
            masks = replay_data.masks
            # # # # # # # # # # # # # # # # # # # # # # # # # #
            
            observation_preds, action_preds, reward_preds = self.policy._actor(
                observations, actions, rewards, returns_to_go, timesteps, masks, deterministic=False, params=self.policy.params
            )
            action_dim = action_preds.shape[2] 
            action_preds = action_preds.reshape(-1, action_dim)[masks.reshape(-1) > 0]
            action_targets = actions.reshape(-1, action_dim)[masks.reshape(-1) > 0]
           
            self.policy.optimizer_state, self.policy.params, loss, info = jit_optimize(
                self._loss,
                self.policy.optimizer,
                self.policy.optimizer_state,
                self.policy.params,
                None,
                action_preds=action_preds,
                action_targets=action_targets,
            )

            actor_losses.append(np.array(info["actor_loss"]))
        
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        action_preds: jnp.ndarray,
        action_targets: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

        loss = jnp.mean(jnp.square(action_preds - action_targets)) 
        info = {
            'actor_loss': loss
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
        tb_log_name: str = "DT",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "DT":

        return super(DT, self).learn(
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
