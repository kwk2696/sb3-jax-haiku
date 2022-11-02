import warnings
from functools import partial
from typing import Any, Dict, Optional, Type, Union, Tuple

import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
from gym import spaces


from sb3_jax.common.on_policy_algorithm import OnPolicyAlgorithm 
from sb3_jax.common.policies import ActorCriticPolicy
from sb3_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples
from sb3_jax.common.utils import get_schedule_fn
from sb3_jax.common.jax_utils import jit_optimize, explained_variance


class PPO(OnPolicyAlgorithm):
    """Proximal Policy Optimization algorithm (PPO) (clip version)."""

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )


        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer."""

        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                observations = self.policy.preprocess(rollout_data.observations, training=True)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.astype('int32').flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Re-sample the noise matrix because the log_std has change
                if self.use_sde:
                    raise NotImplementedError

                # Update using jit  
                self.policy.optimizer_state, self.policy.params, loss, info = jit_optimize(
                    self._loss,
                    self.policy.optimizer,
                    self.policy.optimizer_state,
                    self.policy.params,
                    self.max_grad_norm,
                    observations=observations,
                    actions=actions,
                    old_values=rollout_data.old_values,
                    old_log_prob=rollout_data.old_log_prob,
                    advantages=advantages,
                    returns=rollout_data.returns,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
                )
                
                entropy_losses.append(np.array(info["entropy_loss"]))
                pg_losses.append(np.array(info["policy_gradient_loss"]))
                value_losses.append(np.array(info["value_loss"]))
                loss = np.array(np.array(info["loss"])) 
                approx_kl_divs.append(np.array(info["approx_kl_div"]))
                clip_fractions.append(np.array(info["clip_fraction"]))

            if not continue_training:
                break
        
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        #if hasattr(self.policy, "log_std"):
        #    self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf) 

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        observations: jnp.ndarray, 
        actions: jnp.ndarray,
        old_values: jnp.ndarray, 
        old_log_prob: jnp.ndarray, 
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        clip_range: float,
        clip_range_vf: float,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        
        mean_actions, log_std = self.policy._actor(observations, params)
        log_prob = self.policy.action_dist_fn.log_prob(actions, mean_actions, log_std)
        entropy = self.policy.action_dist_fn.entropy(mean_actions, log_std)
        values = self.policy._value(observations, params)
        values = values.flatten()

        # ratio between old and new policy
        ratio = jnp.exp(log_prob - old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()
        
        # clip fraction
        clip_fraction = jnp.mean((jnp.abs(ratio - 1) > clip_range))

        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = old_values + jnp.clip(
                values - old_values, -clip_range_vf, clip_range_vf
            )
        # Value loss using the TD target
        value_loss = jnp.square(returns - values_pred).mean()
        
        # Entropy loss
        entropy_loss = -jnp.mean(entropy)
        
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            
        log_ratio = log_prob - old_log_prob
        approx_kl_div = jnp.mean((jnp.exp(log_ratio) - 1) - log_ratio)
        
        info = {
            'log_prob': log_prob,
            'entropy_loss': entropy_loss,
            'policy_gradient_loss': policy_loss,
            'value_loss': value_loss,       
            'loss': loss,
            'clip_fraction': clip_fraction,
            'approx_kl_div': approx_kl_div,
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
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
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
