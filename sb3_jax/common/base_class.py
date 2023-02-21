"""Abstract base classes for RL algorithms."""

import io
import pathlib
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)

from sb3_jax.common import utils
from sb3_jax.common.policies import BasePolicy, get_policy_from_name
from sb3_jax.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from sb3_jax.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from sb3_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sb3_jax.common.utils import (
    check_for_correct_spaces,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    #update_learning_rate,
)


def maybe_make_env(env: Union[GymEnv, str, None], verbose: int) -> Optional[GymEnv]:
    """If env is a string, make the environment; otherwise, return env."""
    if isinstance(env, str):
        if verbose >= 1:
            print(f"Creating environment from the given name '{env}'")
        env = gym.make(env)
    return env


class BaseAlgorithm(ABC):

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str, None],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        
        if isinstance(policy, str) and policy_base is not None:
            self.policy_class = get_policy_from_name(policy_base, policy)
        else:
            self.policy_class = policy

        self.env = None  # type: Optional[GymEnv]
        # get VecNormalize object if needed
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None  # type: Optional[gym.spaces.Space]
        self.action_space = None  # type: Optional[gym.spaces.Space]
        self.n_envs = None
        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.eval_env = None
        self.seed = seed
        self.action_noise = None  # type: Optional[ActionNoise]
        self.start_time = None
        self.policy = None
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.lr_schedule = None  # type: Optional[Schedule]
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde 
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1
        # Buffers for logging
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # The logger object
        self._logger = None  # type: Logger
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        
        # Create and wrap the env if needed
        if env is not None:
            if isinstance(env, str):
                if create_eval_env:
                    self.eval_env = maybe_make_env(env, self.verbose)

            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose, monitor_wrapper)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space, gym.spaces.Dict):
                raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}")
            
            if self.use_sde and not isinstance(self.action_space, gym.spaces.Box):
                raise ValueError("gSDE can only be used with continuous actions.")

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """Wrap environment with the appropriate wrappers if needed."""
        if not isinstance(env, VecEnv):
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if isinstance(env.observation_space, gym.spaces.Dict):
            for space in env.observation_space.spaces.values():
                if isinstance(space, gym.spaces.Dict):
                    raise ValueError("Nested observation spaces are not supported (Dict spaces inside Dict space).")

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, gym.spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _get_eval_env(self, eval_env: Optional[GymEnv]) -> Optional[GymEnv]:
        """
        Return the environment that will be used for evaluation.

        :param eval_env:)
        :return:
        """
        if eval_env is None:
            eval_env = self.eval_env

        if eval_env is not None:
            eval_env = self._wrap_env(eval_env, self.verbose)
            assert eval_env.num_envs == 1
        return eval_env

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """Compute current progress remaining (starts from 1 and ends to 0)"""
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    
    def _update_learning_rate(self) -> None:
        raise NotImplementedError

    def _excluded_save_params(self) -> List[str]:
        """Returns the names of the parameters that should be excluded from being saved by pickling."""
        """E.g. replay buffers are skipped by default."""
        
        return [
            "policy",
            "env",
            "key",
            "eval_env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _get_jax_save_params(self) -> Tuple[List[str], List[str]]:
        """Get the name of variables that will be saved."""
        state_dicts = ["policy"]
        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
    ) -> BaseCallback:
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Create eval callback in charge of the evaluation
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """Initialize different variables needed for training."""
        self.start_time = time.time()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times 
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """Retrieve reward, episode length, episode success and update the buffer."""
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        """Returns the current environment (can be None if not defined)."""
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """Return the ``VecNormalize`` wrapper of the training env, if it exists."""
        return self._vec_normalize_env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """Checks the validity of the environment, Furthermore wrap any non vectorized env into a vectorized."""
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env = self._wrap_env(env, self.verbose)
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "BaseAlgorithm":
        """Return a trained model."""

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Get the policy action from an observation (and optional hidden state)."""
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """Set the seed of the pseudo-random generators (python, numpy, pytorch, gym, action_space)."""
        if seed is None:
            return
        set_random_seed(seed)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> "BaseAlgorithm":
        """Load the model from a zip-file."""
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params = load_from_zip_file(
            path, 
            custom_objects=custom_objects, 
            verbose=1, 
            print_system_info=print_system_info,
        )

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )
        
        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )
        
        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model._load_jax_params(params) 
        model._load_norm_layer(path)

        if model.use_sde:
            raise NotImplementedError
        return model

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """Save all the attributes of the object and the model parameters in a zip-file."""
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params()) 

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of state_dicts
        params_to_save = self._save_jax_params()
        
        save_to_zip_file(path, data=data, params=params_to_save, verbose=1)
        self._save_norm_layer(path)
