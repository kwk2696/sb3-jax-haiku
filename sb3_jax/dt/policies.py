from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import math

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
from sb3_jax.common.jax_utils import jax_print
from sb3_jax.common.policies import BasePolicy, register_policy
from sb3_jax.common.type_aliases import Schedule
from sb3_jax.common.utils import get_dummy_decision_transformer
from sb3_jax.common.norm_layers import BaseNormLayer


class TrajectoryModel(hk.Module):
    """
    This model uses GPT to model (return_1, state_1, action_1, return_2, state_2, ...)
    followed by: https://github.com/mxu34/prompt-dt
    """
    supported_prompts = [None, 'id', 'fix', 'soft']
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        hidden_size: int = 128,
        max_length: int = None,
        max_ep_length: int = 4096,
        squash_action: bool = True,      
        num_tasks: int = None,
        prompt_type: str = None,
        prompt_length: int = 1,
        config: transformers.GPT2Config = None
    ):
        super().__init__()
        self.observation_dim = get_flattened_obs_dim(observation_space) 
        self.action_dim = get_action_dim(action_space)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.max_ep_length = max_ep_length
        self.squash_action = squash_action

        self.num_tasks = num_tasks
        self.prompt_type = prompt_type
        assert prompt_type in TrajectoryModel.supported_prompts, f"{prompt_type} is not supported for DT"
        self.prompt_length = prompt_length

        self.config = config 

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray, 
        returns_to_go: jnp.ndarray,
        timesteps: jnp.ndarray,
        attention_mask: jnp.ndarray = None,
        task_id: int = None, 
        prompt: jnp.ndarray = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size, seq_length = observations.shape[0], observations.shape[1]
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attened to, - if not 
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

        # embed each modality with a different head
        observation_embed = hk.Linear(self.hidden_size, **init_weights())(observations)
        action_embed = hk.Linear(self.hidden_size, **init_weights())(actions)
        return_embed = hk.Linear(self.hidden_size, **init_weights())(returns_to_go)
        timestep_embed = hk.Embed(self.max_ep_length, self.hidden_size, **init_embed())(timesteps)

        # timestep embeddings are treated similar to positional embeddings
        observation_embed = observation_embed + timestep_embed
        action_embed = action_embed + timestep_embed 
        return_embed = return_embed + timestep_embed

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = jnp.stack(
            (return_embed, observation_embed, action_embed), axis=1
        ).transpose(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = hk.LayerNorm(-1, create_scale=True, create_offset=True)(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = jnp.stack(
            (attention_mask, attention_mask, attention_mask), axis=1
        ).transpose(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        if self.prompt_type == 'id': # using task id
            if task_id is None: 
                task_ids = jax.nn.one_hot(jnp.arange(0, self.num_tasks), self.num_tasks)
            else: # for evaluation
                task_ids = jax.nn.one_hot(jnp.array([task_id]), self.num_tasks)
            id_embed = hk.Linear(self.hidden_size, **init_weights())(task_ids)
        elif self.prompt_type == 'soft': # using soft prompt
            id_embed = hk.get_parameter("soft_prompt", [self.num_tasks, self.prompt_length, self.hidden_size], init=hk.initializers.TruncatedNormal())
            if task_id is not None: # for evaluation
                id_embed = id_embed[task_id]

        if prompt is None and (self.prompt_type == 'id' or self.prompt_type == 'soft'): 
            """ Used for given task id or soft prompt for pretraining """ 
            # no prompt then cat task_id -> batch_size/task_nums
            id_stacked_embed = jnp.repeat(id_embed, math.ceil(batch_size/id_embed.shape[0]), axis=0).reshape(batch_size, -1, self.hidden_size) 
            id_stacked_attention_mask = jnp.ones((batch_size, self.prompt_length), dtype=jnp.int32)
            stacked_inputs = jnp.concatenate((id_stacked_embed, stacked_inputs), axis=1)
            stacked_attention_mask = jnp.concatenate((id_stacked_attention_mask, stacked_attention_mask), axis=1)
        elif prompt is not None and self.prompt_type == 'fix':
            """ Used for fixed prompt for pretraining """
            prompt_observations, prompt_actions, prompt_rewards, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_observation_embed = hk.Linear(self.hidden_size, **init_weights())(prompt_observations)
            prompt_action_embed = hk.Linear(self.hidden_size, **init_weights())(prompt_actions)
            prompt_return_embed = hk.Linear(self.hidden_size, **init_weights())(prompt_returns_to_go)
            prompt_timestep_embed = hk.Embed(self.max_ep_length, self.hidden_size, **init_embed())(prompt_timesteps)
            
            prompt_observation_embed = prompt_observation_embed + prompt_timestep_embed
            prompt_action_embed = prompt_action_embed + prompt_timestep_embed
            prompt_return_embed = prompt_return_embed + prompt_timestep_embed
            
            prompt_seq_length = prompt_observations.shape[1]
            prompt_stacked_inputs = jnp.stack(
                (prompt_return_embed, prompt_observation_embed, prompt_action_embed), axis=1
            ).transpose(0, 2, 1, 3).reshape(prompt_observations.shape[0], 3 * prompt_seq_length, self.hidden_size)
            prompt_stacked_attention_mask = jnp.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), axis=1
            ).transpose(0, 2, 1).reshape(prompt_observations.shape[0], 3 * prompt_seq_length)
            
            # prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
            # prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
            # prompt_stacked_inputs = jnp.repeat(prompt_stacked_inputs, batch_size, 0)
            # prompt_stacked_attention_mask = jnp.repeat(prompt_stacked_attention_mask, batch_size, 0)
            stacked_inputs = jnp.concatenate((prompt_stacked_inputs, stacked_inputs), axis=1)
            stacked_attention_mask = jnp.concatenate((prompt_stacked_attention_mask, stacked_attention_mask), axis=1)
        elif prompt is not None:
            """ Used for finetuning prompts """
            # yes prompt then cat prompt 
            prompt_length = prompt.shape[0]
            prompt_stacked_inputs = jnp.tile(prompt, (batch_size, 1)).reshape(batch_size, -1, self.hidden_size)
            prompt_stacked_attention_mask = jnp.ones((batch_size, prompt_length), dtype=jnp.int32)
        
            stacked_inputs = jnp.concatenate((prompt_stacked_inputs, stacked_inputs), axis=1)
            stacked_attention_mask = jnp.concatenate((prompt_stacked_attention_mask, stacked_attention_mask), axis=1)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = GPT2Model(self.config)(
            input_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            deterministic=deterministic
        )
        x = transformer_outputs["last_hidden_state"]
        info = {"last_hidden_state": x} 

        if prompt is None and (self.prompt_type == 'id' or self.prompt_type == 'soft'):
            # discard prompt output, then reshaping 
            x = x[:, -seq_length*3:, :]
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).transpose(0, 2, 1, 3)
        elif prompt is not None and self.prompt_type == 'fix':
            x = x.reshape(batch_size, -1, 3, self.hidden_size).transpose(0, 2, 1, 3)
        else:
            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).transpose(0, 2, 1, 3)  # [b, 3, 1, d]

        # get predictions
        return_preds = hk.Linear(1, **init_weights())(x[:,2])[:, -seq_length:, :] # predict next return given state and action
        observation_preds = hk.Linear(self.observation_dim, **init_weights())(x[:,2])[:, -seq_length:, :] # predict next state given state and action
        action_preds = hk.Linear(self.action_dim, **init_weights())(x[:,1])[:, -seq_length:, :] # predict next action given state
        if self.squash_action: 
            action_preds = nn.tanh(action_preds)
        return observation_preds, action_preds, return_preds, info
       

class PromptModel(hk.Module):
    """
    This model is for soft-prompt.
    """
    def __init__(
        self,
        prompt_length: int,
        hidden_size: int = 128,
    ):
        super(PromptModel, self).__init__()
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.prompt_shape = [self.prompt_length, self.hidden_size]

    def __call__(self) -> jnp.ndarray:
        prompt_embeddings = hk.get_parameter("prompt", self.prompt_shape, init=hk.initializers.TruncatedNormal()) # N(0,1)
        #prompt_embeddings = hk.Linear(self.hidden_size, **init_weights())(prompt_embeddings)
        #prompt_embeddings = hk.LayerNorm(-1, create_scale=True, create_offset=True)(prompt_embeddings)
        return prompt_embeddings


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DTPolicy(BasePolicy):
    """Policy class for DT."""
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        max_grad_norm: float = .25,
        num_tasks: int = None,
        prompt_type: str = None,
        prompt_length: int = 1, # prompt size per task
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
            seed=seed,
        )

        self.num_tasks = num_tasks
        self.prompt_type = prompt_type
        self.prompt_length = prompt_length

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

        self._task_id = None    # for evaluation task id
        self._prompt = None     # for evaluation prompt

        self._build(lr_schedule)
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                squash_output=self.squash_output,
                max_grad_norm=self.max_grad_norm,
                num_tasks=self.num_tasks,
                prompt_type=self.prompt_type,
                prompt_length=self.prompt_length,
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
            num_tasks=self.num_tasks,
            prompt_type=self.prompt_type,
            prompt_length=self.prompt_length,
            config=config,
        ) 

    def _build(self, lr_schedule: Schedule) -> None:
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)
        
        def fn_actor(observations, actions, rewards, returns_to_go, timesteps, attention_mask, task_ids, prompt, deterministic):    
            features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
            action = self._build_actor()
            return action(observations, actions, rewards, returns_to_go, timesteps, attention_mask, task_ids, prompt, deterministic)
        
        # transform_with_state function returns a pair of pure functions
        # explicityly collecting and injecting parameter and state values
        # 1. init: ``params, state = init(rng, *a, **k)``
        # 2. apply: ``out, state = apply(params, rng, *a, **k)``
        params, self.actor = hk.transform_with_state(fn_actor)
        self.params, self.state = params(
            next(self.rng), 
            *get_dummy_decision_transformer(self.observation_space, self.action_space, repeat=self.num_tasks), 
            attention_mask=None,
            task_ids=None,
            prompt=(*get_dummy_decision_transformer(self.observation_space, self.action_space, repeat=self.num_tasks), \
                    np.repeat(np.zeros((1,))[None, ...].astype(np.float32), self.num_tasks, axis=0)) if self.prompt_type == 'fix' else None,
            deterministic=False,
        )
        # TODO: optimizer with LambdaLR scheduler
        def fn_lr_scheduler(lr):
            return lr
        self.optimizer = self.optimizer_class(learning_rate=fn_lr_scheduler(lr_schedule), **self.optimizer_kwargs)
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
        task_ids: jnp.ndarray,
        prompt: jnp.ndarray,
        deterministic: bool,
        params: hk.Params,
        state: hk.Params,
        rng=None,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict]: 
        # returns dt output & haiku state 
        return self.actor(params, state, rng, observations, actions, rewards, returns_to_go, timesteps, attention_mask, task_ids, prompt, deterministic)
 
    def _predict(
        self, 
        traj_observations: Dict[str, jnp.ndarray], 
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[Dict[str, Any]]]:
        observations, actions, reward, returns_to_go, timesteps, attention_mask = self._preprocess(**traj_observations)

        (_, action_preds, _, info), _ = self._actor(
            observations, actions, None, returns_to_go, timesteps, attention_mask, self._task_id, self._prompt, 
            True, self.params, self.state, next(self.rng)
        )
        return action_preds[0,-1].reshape(1, -1), info
    
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
    
    def set_task_id(self, task_id) -> None:
        self._task_id = task_id
    
    def get_task_id(self) -> int:
        return self._task_id

    def set_prompt(self, prompt) -> None:
        self._prompt = prompt

    def get_prompt(self) -> jnp.array:
        return self._prompt

    def save(self, path: str) -> None:
        """Save model to path."""

    def load(self, path: str):
        """Load model from path."""


class TDTPolicy(BasePolicy):
    """ Policy class for finetuning DT. """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        pretrained_policy: DTPolicy, 
        prompt_length: int,
        max_grad_norm: float = .25,
        # prompt configs
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
        super(TDTPolicy, self).__init__(
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

        self._pretrained_policy = pretrained_policy
        self.prompt_length = prompt_length
        
        # set hidden size of prompt embedding as pretrained embedding size
        self.hidden_size = self.pretrained_policy.hidden_size

        self.max_grad_norm = max_grad_norm

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]: 
        data = super()._get_constructor_parameters()
        
        data.update(
            dict(
                prompt_length=self.prompt_length,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                normalization_class=self.normalization_class,
                normalization_kwargs=self.normalization_kwargs,
            )
        )
        return data
    
    def _build_prompt(self) -> hk.Module:
        return PromptModel(self.prompt_length, self.hidden_size)

    def _build(self, lr_schedule: Schedule) -> None:
        if self.normalization_class is not None:
            self.normalization_layer = self.normalization_class(self.observation_space.shape, **self.normalization_kwargs)

        def fn_prompt():
            prompt = self._build_prompt()
            return prompt()

        params, self.prompt = hk.without_apply_rng(hk.transform(fn_prompt))
        self.params = params(next(self.rng))
        self.optimizer = self.optimizer_class(learning_rate=lr_schedule, **self.optimizer_kwargs)
        self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnums=0)
    def _prompt(
        self,
        params: hk.Params,
    ) -> jnp.ndarray: 
        # returns soft prompt
        return self.prompt(params)
    
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
        pretrained_params: hk.Params,
        pretrained_state: hk.Params,
        pretrained_rng=None,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict]: 
        prompt = self._prompt(params)
        return self.pretrained_policy._actor(observations, actions, rewards, returns_to_go, timesteps, attention_mask, None, deterministic,
                pretrained_params, pretrained_state, pretrained_rng, prompt)

    def _predict(
        self, 
        traj_observations: Dict[str, jnp.ndarray], 
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[Dict[str, Any]]]:
        observations, actions, reward, returns_to_go, timesteps, attention_mask = self.pretrained_policy._preprocess(**traj_observations)
        (_, action_preds, _, info), _ = self._actor(observations, actions, None, returns_to_go, timesteps, attention_mask, True, self.params,
            self.pretrained_policy.params, self.pretrained_policy.state, next(self.pretrained_policy.rng))
        return action_preds[0,-1].reshape(1, -1), info
    
    @property
    def pretrained_policy(self) -> DTPolicy:
        return self._pretrained_policy


MlpPolicy = DTPolicy
PromptTunePolicy = TDTPolicy

register_policy("MlpPolicy", DTPolicy)
register_policy("PromptTunePolicy", TDTPolicy)
