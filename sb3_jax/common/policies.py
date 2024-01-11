"""Policies: abstract base class and concrete implementations."""

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import jax
import optax 
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax import nn

from sb3_jax.common.distributions import (
    Distribution,
    DiagGaussianDistributionFn,
    CategoricalDistributionFn, 
    make_proba_distribution,
)
from sb3_jax.common.preprocessing import get_act_dim, is_image_space, maybe_transpose, preprocess_obs
from sb3_jax.common.jax_layers import (
    init_weights,
    create_mlp,
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.utils import is_vectorized_observation, obs_as_jnp, get_dummy_obs, get_dummy_act
from sb3_jax.common.jax_utils import jax_print
from sb3_jax.common.norm_layers import BaseNormLayer


class BaseModel(ABC):
    """Base model obejct."""
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[hk.Module] = None,
        normalize_images: bool = True,
        optimizer_class: Callable = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        normalization_class: Type[BaseNormLayer] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 1,
    ):
        super(BaseModel, self).__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        if normalization_kwargs is None:
            normalization_kwargs = {} 

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # 

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        
        self.normalization_class = normalization_class
        self.normalization_kwargs = normalization_kwargs
        self.normalization_layer = None

        self.seed = seed
        print(f"Seeding {self.seed}")
        self.rng = hk.PRNGSequence(seed) # Random seed

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """ Get data that need to be saved in order to re-create the model when loading it from disk."""
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    def save(self, path: str) -> None:
        """Save model to path."""

    def load(self, path: str) -> "BaseModel":
        """Load model from path."""
    
    def preprocess(self, observation: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        observation = preprocess_obs(observation, self.observation_space, self.normalize_images)
        if self.normalization_class is not None:
            observation = self.normalization_layer(observation, training=training)  
        return observation

    def obs_to_jnp(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[jnp.ndarray, bool]:
        """Convert an input observation to a Jax array."""
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            # for some algo e.g. DT, observation_space and predict input does not match
            # thus, in this case we check the observation_space
            if isinstance(self.observation_space, gym.spaces.Dict): 
                for key, obs in observation.items():
                    obs_space = self.observation_space.spaces[key]
                    if is_image_space(obs_space):
                        obs_ = maybe_transpose(obs, obs_space)
                    else:
                        obs_ = np.array(obs)
                    vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                    # Add batch dimension if needed
                    observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)
        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_jnp(observation)
        return observation, vectorized_env


class BasePolicy(BaseModel):
    """The base policy object."""

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super(BasePolicy, self).__init__(*args, **kwargs)
        self._squash_output = squash_output
 
    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @abstractmethod
    def _predict(self, observation: jnp.ndarray, deterministic: bool = False) -> Tuple[jnp.ndarray, Optional[Dict[str, Any]]]:
        """Get the action according to the policy for a given observation."""
 
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]], Optional[Dict[str, Any]]]:
        """Get the policy action from an observation (and optional hidden state)."""
        
        observation, vectorized_env = self.obs_to_jnp(observation)

        actions, info = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = np.array(actions)

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions, state, info

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale the action from [low, high] to [-1, 1]."""
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """Rescale the action from [-1, 1] to [low, high]."""
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


class ActorCriticPolicy(BasePolicy):
    """ Policy class for actor-critic algorithms (has both policy and value prediction).""" 

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
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

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
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

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        
        # Feature Extractor Class
        self.features_extractor_class = features_extractor_class
        
        # Normalization Layer Class
        self.normalization_class = normalization_class

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = dict()
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution class
        self.action_dist_class, self.action_dist_fn = make_proba_distribution(action_space, use_sde=use_sde)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                # squash_output=default_none_kwargs["squash_output"],
                # full_std=default_none_kwargs["full_std"],
                # use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                normalization_class=self.normalization_class,
                normalization_kwargs=self.normalization_kwargs,
            )
        )
        return data
    
    def _build_mlp_extractor(self) -> Tuple[hk.Module, hk.Module, hk.Module]:
        """Create the policy and value networks. Part of the layers can be shared."""
        return MlpExtractor(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        ).setup()

    def _build(self, lr_schedule: Schedule) -> None:
        """Create the networks and the optimizer."""
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)

        if isinstance(self.action_dist_fn, DiagGaussianDistributionFn):
            action_dim = get_act_dim(self.action_space)
            self.dist_kwargs.update(dict(log_std_init=self.log_std_init))
        elif isinstance(self.action_dist_fn, CategoricalDistributionFn):
            action_dim = self.action_space.n
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist_fn}'.")
        
        def fn_actor_critic():
            features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
            shared_net, action_net, value_net = self._build_mlp_extractor()
            action = hk.Sequential(
                [features_extractor, shared_net, action_net, self.action_dist_class(action_dim, **self.dist_kwargs)]
            )
            value = hk.Sequential(
                [features_extractor, shared_net, value_net, hk.Linear(1, **init_weights())]
            )

            def init(observation: jnp.ndarray):
                return action(observation), value(observation)
            return init, (action, value)
 
        params, (self.actor, self.value) = hk.without_apply_rng(hk.multi_transform(fn_actor_critic))
        self.params = params(next(self.rng), get_dummy_obs(self.observation_space))  
        self.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs)
        self.optimizer_state = self.optimizer.init(self.params)
    
    def forward(self, observation: jnp.ndarray, deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        observation = self.preprocess(observation)
        mean_actions, log_std = self._actor(observation, self.params)
        actions = self.action_dist_fn.get_actions(mean_actions, log_std, deterministic, next(self.rng))
        values = self._value(observation, self.params)
        log_prob = self.action_dist_fn.log_prob(actions, mean_actions, log_std)
        
        return actions, values, log_prob

    @partial(jax.jit, static_argnums=0)
    def _actor(self, observation: jnp.ndarray, params: hk.Params) -> jnp.ndarray:
        return self.actor(params, observation) 
    
    @partial(jax.jit, static_argnums=0)
    def _value(self, observation: jnp.ndarray, params: hk.Params) -> jnp.ndarray:
        return self.value(params, observation)

    def _predict(self, observation: jnp.ndarray, deterministic: bool = False) -> Tuple[jnp.ndarray, Optional[Dict[str, Any]]]:
        observation = self.preprocess(observation)
        mean_actions, log_std = self._actor(observation, self.params)
        return self.action_dist_fn.get_actions(mean_actions, log_std, deterministic, next(self.rng)), None
    
    def evaluate_actions(self, observation: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        observation = self.preprocess(observation)
        mean_actions, log_std = self._actor(observation, self.params)
        log_prob = self.action_dist_fn.log_prob(actions, mean_actions, log_std)            
        entropy = self.action_dist_fn.entropy(mean_actions, log_std)  
        values = self._value(observation, self.params)
        return values, log_prob, entropy  

    def predict_values(self, observation: jnp.ndarray) -> jnp.ndarray:
        observation = self.preprocess(observation)
        value = self._value(observation, self.params)
        return value


class ContinuousCritic(BaseModel):
    """Critic network(s) for SAC."""
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        seed: int = 1,
    ):
        super().__init__(
            observation_space,
            action_space,
            normalize_images=normalize_images,
            seed=seed,
        )
        
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.n_critics = n_critics

        self._build()

    def _build_critic(self, net_arch: List[int]) -> hk.Module:
        return create_mlp(
            output_dim=1,
            net_arch=net_arch,
            activation_fn=self.activation_fn,
            squash_output=False,
        )
            
    def _build(self,) -> None:
        """Create critics."""
        def fn_critic():
            q_networks = []
            for idx in range(self.n_critics):
                q_networks.append(self._build_critic(self.net_arch))

            def init(qvalue_input: jnp.ndarray):
                return tuple(q_net(qvalue_input) for q_net in q_networks)
            return init, q_networks 

        params, self.q_networks = hk.without_apply_rng(hk.multi_transform(fn_critic))
        self.params = params(next(self.rng), jnp.concatenate([get_dummy_obs(self.observation_space), get_dummy_act(self.action_space)], axis=1))
    
    def forward(self, observation: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        # observation is should be preprocessed 
        return self._critic(observation, actions, self.params)
    
    def q1_forward(self, observation: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        # observation is should be preprocessed 
        return self._critic(observation, actions, self.params)[0]

    @partial(jax.jit, static_argnums=0)
    def _critic(self, observation: jnp.ndarray, actions: jnp.ndarray, params: hk.Params) -> jnp.ndarray:
        qvalue_input = jnp.concatenate([observation, actions], axis=1)
        return tuple(q_net(params, qvalue_input) for q_net in self.q_networks) 


_policy_registry = dict() # type: Dict[Type[BasePolicy], Dict[str, Type[BasePolicy]]]


def get_policy_from_name(base_policy_type: Type[BasePolicy], name: str) -> Type[BasePolicy]:
    """Returns the registered policy from the base type and name."""
    if base_policy_type not in _policy_registry:
        raise KeyError(f"Error: the policy type {base_policy_type} is not regisetered!")
    if name not in _policy_registry[base_policy_type]:
        raise KeyError(
                f"Error: unknown policy type {name},"
                f"the only registered type are: {list(_policy_registry[base_policy_type].keys())}!"
        )
    return _policy_registry[base_policy_type][name]


def register_policy(name: str, policy: Type[BasePolicy]) -> None:
    """Register a policy, so it can be called using its name."""
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break 
    if sub_class is None:
        raise ValueError(f"Error: the policy {policy} is not of any known subclasses of BasePolicy!")

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        if _policy_registry[sub_class][name] != policy:
            raise ValueError(f"Error: the name {name} is already registered for a different policy, will not override.")
    _policy_registry[sub_class][name] = policy

