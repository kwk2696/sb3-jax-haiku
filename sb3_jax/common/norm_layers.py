"""Normalization layer (not using haiku module)"""
import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import pickle
import jax
import jax.numpy as jnp
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class BaseNormLayer(ABC):
    """ Base class for Normalization Layer """
    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        clip_obs: float = 10.0,
        eps: float = 1e-8,
    ):
        super(BaseNormLayer, self).__init__()
        self.shape = shape
        self.clip_obs = clip_obs
        self.eps = eps
        self.reset_running_stats()

    def reset_running_stats(self,) -> None:
        self.running_mean = np.zeros(self.shape, np.float32)
        self.running_var = np.ones(self.shape, np.float32)
        self.count = self.eps
    
    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if training:
            self.update_stats(x)
        return np.clip((x - self.running_mean) / np.sqrt(self.running_var + self.eps), -self.clip_obs, self.clip_obs)

    def update_stats(self, batch: jnp.ndarray) -> None:
        running_mean, running_var, count = \
            self._update_stats(self.running_mean, self.running_var, self.count, batch)

        self.running_mean = running_mean
        self.running_var = running_var
        self.count = count
    
    @abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _update_stats(self, running_mean: jnp.ndarray, running_var: jnp.ndarray, count: int, batch: jnp.ndarray):
        """jit method for update_stats."""
    
    def save(self, save_path: str) -> None:
        with open(os.path.join(save_path, "norm_layer.zip"), "wb") as fp:
            pickle.dump(self, fp)
    
    @staticmethod
    def load(load_path: str) -> "BaseNormLayer":
        with open(os.path.join(load_path, "norm_layer.zip"), "rb") as fp:
            norm_layer = pickle.load(fp)
        return norm_layer 


class RunningNormLayer(BaseNormLayer):
    """ Class for Running Normalization Layer """
    def __init___(self):
        super(RunningNormLayer, self).__init__()

    @partial(jax.jit, static_argnums=0)
    def _update_stats(
        self, 
        running_mean: jnp.ndarray, 
        running_var: jnp.ndarray, 
        count: int, 
        batch: jnp.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - running_mean
        tot_count = count + batch_count 
        new_running_mean = running_mean + (delta * batch_count / tot_count)

        m_a = running_var * count 
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_running_var = m_2 / tot_count

        new_count = count + batch_count
        return new_running_mean, new_running_var, new_count
