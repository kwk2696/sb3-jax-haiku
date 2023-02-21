import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import jax
import numpy as np
import haiku as hk

from sb3_jax.common.base_class import BaseAlgorithm
from sb3_jax.common.policies import BasePolicy, ActorCriticPolicy
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.buffers import BaseBuffer
from sb3_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule
 
class OfflineAlgorithm(BaseAlgorithm):
    """Base class for offline algorithms."""
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        replay_buffer: Type[BaseBuffer],
        learning_rate: Union[float, Schedule],
        batch_size: int = 256, 
        gamma: float = 0.99,
        gradient_steps: int = 1,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None, 
        verbose: int = 0,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None, 
    ):
        super(OfflineAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            support_multi_env=False,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        if replay_buffer is None:
            raise ValueError(f"Replay buffer should be provided")
        self.replay_buffer = replay_buffer

        self.actor = None
        if _init_setup_model:
            self._setup_model() 

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            seed=self.seed,
            **self.policy_kwargs,
        )

    def train(sef) -> None:
        """implemented by individual algorithms."""
        raise NotImplementedError()
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5, 
        tb_log_name: str = "OfflineAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OfflineAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)
     
            self.num_timesteps += 1
            if log_interval is not None and self.num_timesteps % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
            
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

        callback.on_training_end()
        
        return self 
