from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gym
import jax
import optax
import transformers
import numpy as np
import haiku as hk
import jax.numpy as jnp
from jax import nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from sb3_jax.dt.gpt2 import GPT2Model, init_embed
from sb3_jax.common.preprocessing import get_action_dim, get_flattened_obs_dim
from sb3_jax.common.jax_layers import (
    init_weights, 
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from sb3_jax.common.jax_utils import warmup_scheduler
from sb3_jax.common.policies import BasePolicy
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.utils import get_dummy_decision_transformer
from sb3_jax.common.norm_layers import BaseNormLayer


class TrajectoryModel(hk.Module):
    """
    This model uses GPT to model (return_1, state_1, action_1, return_2, state_2, ...)
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        hidden_size: int = 128,
        max_length: int = None,
        max_ep_length: int = 4096,
        squash_action: bool = True,       
        config: transformers.GPT2Config = None
    ):
        super().__init__()
        self.observation_dim = get_flattened_obs_dim(observation_space) 
        self.action_dim = get_action_dim(action_space)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.squash_action = squash_action
        self.config = config 

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray, 
        returns_to_go: jnp.ndarray,
        timesteps: jnp.ndarray,
        attention_mask: jnp.ndarray = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size, seq_length = observations.shape[0], observations.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attened to, - if not 
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

        # embed each modality with a different head
        observation_embeddings = hk.Linear(self.hidden_size, **init_weights())(observations)
        action_embeddings = hk.Linear(self.hidden_size, **init_weights())(actions)
        return_embeddings = hk.Linear(self.hidden_size, **init_weights())(returns_to_go)
        timestep_embeddings = hk.Embed(self.max_ep_length, self.hidden_size, **init_embed())(timesteps)

        # timestep embeddings are treated similar to positional embeddings
        observation_embeddings = observation_embeddings + timestep_embeddings
        action_embeddings = action_embeddings + timestep_embeddings 
        return_embeddings = return_embeddings + timestep_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = jnp.stack(
            (return_embeddings, observation_embeddings, action_embeddings), axis=1
        ).transpose(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = hk.LayerNorm(-1, create_scale=True, create_offset=True)(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = jnp.stack(
            (attention_mask, attention_mask, attention_mask), axis=1
        ).transpose(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = GPT2Model(self.config)(
            input_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            deterministic=deterministic
        )
        x = transformer_outputs["last_hidden_state"]
        
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).transpose(0, 2, 1, 3)  # [b, 3, 1, d]

        # get predictions
        return_preds = hk.Linear(1, **init_weights())(x[:,2]) # predict next return given state and action
        observation_preds = hk.Linear(self.observation_dim, **init_weights())(x[:,2]) # predict next state given state and action
        action_preds = hk.Linear(self.action_dim, **init_weights())(x[:,1]) # predict next action given state
        if self.squash_action: 
            action_preds = nn.tanh(action_preds)
        return observation_preds, action_preds, return_preds 

        
class DTPolicy(BasePolicy):
    """Policy class for DT."""
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        max_grad_norm: float = .25,
        # gpt configs
        max_length: int = None,
        max_ep_length: int = None,
        hidden_size: int = 128,
        n_layer: int = None,
        n_head: int = None,
        n_inner: int = None,
        activation_function: str = 'gelu_new',
        n_positions: int = 1024,
        resid_pdrop: float = None,
        attn_pdrop: float = None,
        # gpt configs end
        squash_output: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable = optax.adamw,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        normalization_class: Type[BaseNormLayer] = None, # Not Used
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 1,
    ):
        super(DTPolicy, self).__init__(
            observation_space, 
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalization_class=normalization_class,
            normalization_kwargs=normalization_kwargs,
            squash_output=squash_output,
        )
        
        self.max_grad_norm = max_grad_norm

        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head, 
        self.n_inner = n_inner
        self.activation_function = activation_function 
        self.n_positions = n_positions 
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop

        self._build(lr_schedule)
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                squash_output=self.squash_output,
                max_grad_norm=self.max_grad_norm,
                max_length=self.max_length,
                max_ep_length=self.max_ep_length,
                hidden_size=self.hidden_size,
                n_layer=self.n_layer,
                n_head=self.n_head,
                n_inner=self.n_inner,
                activation_function=self.activation_function,
                n_positions=self.n_positions,
                resid_prdrop=self.resid_pdrop,
                attn_pdrop=self.attn_pdrop, 
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                normalization_class=self.normalization_class,
                normalization_kwargs=self.normalization_kwargs,
            )
        )
        return data

    def _build_actor(self) -> hk.Module:
        config = transformers.GPT2Config(
            vocab_size=1, # doesn't matter -- we don't use the vocab
            hidden_size=self.hidden_size, 
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_inner=self.n_inner,
            activation_function=self.activation_function,
            n_positions=self.n_positions,
            resid_pdrop=self.resid_pdrop,
            attn_pdrop=self.attn_pdrop,
        )
        return TrajectoryModel(
            observation_space=self.observation_space, 
            action_space=self.action_space,
            hidden_size=self.hidden_size,
            max_length=self.max_length,
            max_ep_length=self.max_ep_length,
            squash_action=True, 
            config=config,
        )
    
    def _build(self, lr_schedule: Schedule) -> None:
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)
        
        def fn_actor(observations, actions, rewards, returns_to_go, timesteps, attention_mask, deterministic):
            features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
            action = self._build_actor()
            return action(observations, actions, rewards, returns_to_go, timesteps, attention_mask, deterministic)
        
        # transform_with_state function returns a pair of pure functions
        # explicityly collecting and injecting parameter and state values
        # 1. init: ``params, state = init(rng, *a, **k)``
        # 2. apply: ``out, state = apply(params, rng, *a, **k)``
        params, self.actor = hk.transform_with_state(fn_actor)
        self.params, self.state = params(
            next(self.rng), 
            *get_dummy_decision_transformer(self.observation_space, self.action_space), 
            attention_mask=None,
            deterministic=False,
        )
        # TODO: optimizer with LambdaLR scheduler
        """
        scheduler = warmup_scheduler(init_value=1e-4, warmup_steps=10000)
        def make_optimizer():
            return optax.chain(
            self.optimizer_class(learning_rate=1e-4, **self.optimizer_kwargs),
            #optax.scale_by_schedule(scheduler),
            #optax.scale(-1.0)
        )
        self.optimizer = make_optimizer
        """
        self.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs)
        self.optimizer_state = self.optimizer.init(self.params)
    
    def forward(
        self, 
        observations: jnp.ndarray, 
        actions: jnp.ndarray, 
        rewards: jnp.ndarray, 
        returns_to_go: jnp.ndarray, 
        timesteps: jnp.ndarray,
        attention_mask: jnp.ndarray,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._actor(observations, actions, rewards, returns_to_go, timesteps, attention_mask, deterministic, self.params, self.state, next(self.rng))
    
    @partial(jax.jit, static_argnums=0)
    def _actor(
        self, 
        observations: jnp.ndarray, 
        actions: jnp.ndarray, 
        rewards: jnp.ndarray, 
        returns_to_go: jnp.ndarray, 
        timesteps: jnp.ndarray,
        attention_mask: jnp.ndarray,
        deterministic: bool,
        params: hk.Params,
        state: hk.Params,
        rng=None, 
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict]: 
        # returns dt output & haiku state 
        return self.actor(params, state, rng, observations, actions, rewards, returns_to_go, timesteps, attention_mask, deterministic)

    
    def _predict(
        self, 
        traj_observations: Dict[str, jnp.ndarray], 
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:        
        observations, actions, reward, returns_to_go, timesteps, attention_mask = self._preprocess(**traj_observations)
        (_, action_preds, _), _ = self._actor(observations, actions, None, returns_to_go, timesteps, attention_mask, True, self.params, self.state, next(self.rng))
        return action_preds[0,-1].reshape(1, -1)
    
    @partial(jax.jit, static_argnums=0)
    def _preprocess(
        self, 
        observations: jnp.ndarray,
        actions: jnp.ndarray, 
        rewards: jnp.ndarray, 
        returns_to_go: jnp.ndarray, 
        timesteps: jnp.ndarray, 
        attention_mask: jnp.ndarray, 
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        observation_dim, action_dim = observations.shape[-1], actions.shape[-1]
        
        observations = observations.reshape(1, -1, observation_dim)
        actions = actions.reshape(1, -1, action_dim)
        returns_to_go = returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1) 

        if self.max_length is not None:
            observations = observations[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokents to sequence length
            attention_mask = jnp.concatenate([jnp.zeros(self.max_length-observations.shape[1]), jnp.ones(observations.shape[1])], dtype=jnp.float32).reshape(1, -1)
            observations = jnp.concatenate(
                [jnp.zeros((observations.shape[0], self.max_length-observations.shape[1], observation_dim)), observations], axis=1, dtype=jnp.float32)
            actions = jnp.concatenate(
                [jnp.zeros((actions.shape[0], self.max_length-actions.shape[1], action_dim)), actions], axis=1, dtype=jnp.float32)
            returns_to_go = jnp.concatenate(
                [jnp.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1)), returns_to_go], axis=1, dtype=jnp.float32)
            timesteps = jnp.concatenate(
                [jnp.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1])), timesteps], axis=1, dtype=jnp.int32)
        else:
            attention_mask = None
        
        return observations, actions, rewards, returns_to_go, timesteps, attention_mask

    def save(self, path: str) -> None:
        """Save model to path."""

    def load(self, path: str):
        """Load model from path."""

MlpPolicy = DTPolicy