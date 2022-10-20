# for PPO
from sb3_jax.common.policies import (
    ActorCriticPolicy,
    register_policy,
)

MlpPolicy = ActorCriticPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
