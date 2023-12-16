
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import wandb
import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
from gym import spaces

from sb3_jax.common.offline_algorithm import OfflineAlgorithm
from sb3_jax.common.buffers import BaseBuffer, OfflineBuffer 
from sb3_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sb3_jax.common.jax_utils import stop_grad, jit_optimize_with_state
from sb3_jax.common.policies import BasePolicy
from sb3_jax.du.policies import DUPolicy


class DU(OfflineAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[DUPolicy]],
        env: Union[GymEnv, str],
        replay_buffer: Type[BaseBuffer] = OfflineBuffer,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        gamma: float = 0.99,
        gradient_steps: int = 1,
        create_eval_env: bool = False,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = 1,
        _init_setup_model: bool = True,
    ):
        super(DU, self).__init__(
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
            supported_action_spaces=(spaces.Discrete, spaces.Box),
            support_multi_env=False,
        )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DU, self)._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
         
        actor_losses = []
        
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            
            observations = self.policy.preprocess(replay_data.observations, training=True)
            actions = replay_data.actions

            self.policy.optimizer_state, self.policy.params, self.policy.state, loss, info = jit_optimize_with_state(
                self._loss,
                self.policy.optimizer,
                self.policy.optimizer_state,
                self.policy.params,
                self.policy.state,
                None,
                observations=observations,
                actions=actions,
                rng=next(self.policy.rng),
            )
            actor_losses.append(np.array(info["actor_loss"]))

        self._n_updates += gradient_steps 
        
        # checking action mse ...
        if self._n_updates % 1000 == 0:
            pred_actions, pred_infos = self.policy._predict(replay_data.observations)
            action_mse = jnp.mean(jnp.square(replay_data.actions - pred_actions))
            self.logger.record("train/mse", action_mse)
            if self.wandb_log is not None:
                wandb.log({"train/mse": action_mse})
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))

        # wandb log
        if self.wandb_log is not None:
            wandb.log({
                # "train/n_updates": self._n_updates,
                "train/actor_loss": np.mean(actor_losses),
            })

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array,
        actions: jax.Array,
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:

        batch_size = observations.shape[0]
        rng_n, rng_t, rng_m, rng_a = jax.random.split(rng, num=4)
        
        # randomly sample some noise
        noise = jax.random.normal(rng_n, shape=(batch_size, self.policy.noise_dim))

        # add noise to clean target actions
        _ts = jax.random.randint(rng_t, (batch_size, 1), minval=1, maxval=self.policy.n_denoise+1)
        y_t = self.policy.ddpm_dict.sqrtab[_ts] * actions + self.policy.ddpm_dict.sqrtmab[_ts] * noise
        mask = jax.random.choice(rng_m, jnp.array([0., 1.]), shape=(batch_size, 1), \
            p=jnp.array([self.policy.cf_drop_rate, 1-self.policy.cf_drop_rate]))

        # use diffusion model to predict noise
        noise_pred, new_state = self.policy._actor(y_t, observations * mask, _ts / self.policy.n_denoise, params, state, rng_a)
        

        # return mse between predicted and true noise
        if self.policy.predict_epsilon:
            loss = jnp.mean(jnp.square(noise_pred - noise))
        else:
            loss = jnp.mean(jnp.square(noise_pred - actions))
        info = {
            'actor_loss': loss
        }
        return loss, (new_state, info)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DU",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "DU":

        # wandb_config
        self.wandb_config = dict(
            algo='du',
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            gradient_steps=self.gradient_steps,
        )
        self.wandb_config.update(self.policy._get_constructor_parameters())

        return super(DU, self).learn(
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
        params_dict['policy_params'] = self.policy.params
        params_dict['policy_state'] = self.policy.state
        return params_dict

    def _load_jax_params(self, params: Dict[str, hk.Params]) -> None:
        self.policy.params = params['policy_params']
        self.policy.state = params['policy_state']

    def _save_norm_layer(self, path: str) -> None:
        if self.policy.normalization_class is not None:
            self.policy.normalization_layer.save(path)

    def _load_norm_layer(self, path: str) -> None:
        if self.policy.normalization_class is not None:
            self.policy.normalization_layer = self.policy.normalization_layer.load(path)
