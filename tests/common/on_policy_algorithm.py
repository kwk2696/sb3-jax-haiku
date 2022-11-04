import gym

from sb3_jax.common import on_policy_algorithm, policies


env = gym.make('HalfCheetah-v2')
obs_space, act_space = env.observation_space, env.action_space
on_policy_algo = on_policy_algorithm.OnPolicyAlgorithm(
                       policy=policies.ActorCriticPolicy,
                       env=env,
                       learning_rate=3e-4,
                       n_steps=1024,
                       gamma=0.99,
                       gae_lambda=0.95,
                       ent_coef=0.0,
                       vf_coef=0.5,
                       max_grad_norm=0.5,
                       use_sde=False, 
                       sde_sample_freq=-1,
                    )
