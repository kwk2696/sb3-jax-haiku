import gym

from sb3_jax.common import offline_algorithm
from sb3_jax.bc import policies
from sb3_jax.common import buffers

env = gym.make('HalfCheetah-v2')
obs_space, act_space = env.observation_space, env.action_space
on_policy_algo = offline_algorithm.OfflineAlgorithm(
                        policy=policies.MlpPolicy,
                        env=env,
                        replay_buffer=buffers.OfflineBuffer,
                        learning_rate=3e-4,
                    )
