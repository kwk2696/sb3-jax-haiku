import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.32'
import gym
import jax
import numpy as np
import torch as th
from torch.distributions import Normal, Categorical

from stable_baselines3.common.distributions import sum_independent_dims
from sb3_jax.common import policies, utils


print("\n<<Continuous Env>>\n")

env = gym.make('HalfCheetah-v2')
obs_space, act_space = env.observation_space, env.action_space
actor_critic_policy = policies.ActorCriticPolicy(
                        observation_space=obs_space,
                        action_space=act_space,
                        lr_schedule=3e-4, 
                        log_std_init=-2,
                        net_arch=[dict(pi=[256,256], vf=[256,256])]
                    )
# print(jax.eval_shape(lambda: actor_critic_policy.params))


obs = utils.get_dummy_obs(obs_space)
action_det, _ = actor_critic_policy.predict(obs, deterministic=True)
print(f"Deterministic Action: {action_det}")
action_stc, _ = actor_critic_policy.predict(obs, deterministic=False)
print(f"Stochastic Action: {action_stc}")
values, log_prob, entropy = actor_critic_policy.evaluate_actions(obs, action_stc)
print(f"Value Function: {values}")
print(f"Log probability: {log_prob}, {entropy}")
_actions, _values, _log_prob = actor_critic_policy.forward(obs, deterministic=False)
#print(f"{_actions} {_values} {_log_prob}")


mean_actions, log_std = actor_critic_policy._actor(obs, actor_critic_policy.params)
mean_actions, log_std = th.Tensor(np.array(mean_actions)), th.Tensor(np.array(log_std)).exp()
torch_dist = Normal(mean_actions, log_std)
action_stc = th.Tensor(action_stc)
torch_log_prob = sum_independent_dims(torch_dist.log_prob(action_stc))
torch_entropy = sum_independent_dims(torch_dist.entropy())
print(f"Torch Log Probability: {torch_log_prob}, {torch_entropy}")


print("\n<<Discrete Env>>\n")

env = gym.make('MountainCar-v0')
obs_space, act_space = env.observation_space, env.action_space
print(obs_space, act_space)
actor_critic_policy = policies.ActorCriticPolicy(
                        observation_space=obs_space,
                        action_space=act_space,
                        lr_schedule=3e-4, 
                        log_std_init=-2,
                        net_arch=[dict(pi=[256,256], vf=[256,256])]
                    )
# print(jax.eval_shape(lambda: actor_critic_policy.params))


obs = utils.get_dummy_obs(obs_space)
action_det, _ = actor_critic_policy.predict(obs, deterministic=True)
print(f"Deterministic Action: {action_det}")
action_stc, _ = actor_critic_policy.predict(obs, deterministic=False)
print(f"Stochastic Action: {action_stc}")
values, log_prob, entropy = actor_critic_policy.evaluate_actions(obs, action_stc)
print(f"Value Function: {values}")
print(f"Log probability: {log_prob}, {entropy}")
_actions, _values, _log_prob = actor_critic_policy.forward(obs, deterministic=False)
#print(f"{_actions} {_values} {_log_prob}")


mean_actions, logits = actor_critic_policy._actor(obs, actor_critic_policy.params) 
mean_actions, logits = th.Tensor(np.array(mean_actions)), th.Tensor(np.array(logits))
torch_dist = Categorical(probs=mean_actions) 
action_stc = th.Tensor(action_stc)
torch_log_prob = torch_dist.log_prob(action_stc)
torch_entropy = torch_dist.entropy()
print(f"Torch Log Probability: {torch_log_prob}, {torch_entropy}")
