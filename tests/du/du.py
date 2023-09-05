import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import argparse
import gym

from stable_baselines3.common import env_util, vec_env
from sb3_jax import DU
from sb3_jax.du.policies import MlpPolicy
from sb3_jax.common.evaluation import evaluate_policy
from sb3_jax.common.norm_layers import RunningNormLayer
from sb3_jax.common.buffers import OfflineBuffer
from sb3_jax.common.utils import print_y


def main(args):
    env_train = env_util.make_vec_env('Swimmer-v3')
    env_train = vec_env.VecNormalize(env_train, norm_obs=False, norm_reward=False)
    env_eval = env_util.make_vec_env('Swimmer-v3')
    
    if args.train:
        print_y("<< Training DU Model >>")
        # Load Buffer
        buff = OfflineBuffer(
            buffer_size=250_000,
            observation_space=env_train.observation_space,
            action_space=env_train.action_space,
        )
        buff = buff.load(path='./tests/data/offline_buffer.pkl')

        # Make DU
        du = DU(
            policy=MlpPolicy,
            env=env_train,
            replay_buffer=buff,
            learning_rate=1e-4,
            batch_size=64,
            verbose=1,
            seed=777,
            wandb_log='test/du',
            policy_kwargs=dict(
                net_arch=[128]*5,
                embed_dim=64,
                hidden_dim=128,
                n_denoise=50,
                beta_scheduler='cosine',
                beta=(1e-4, 0.02),
                normalization_class=RunningNormLayer,
            )
        )

        mean_reward, _ = evaluate_policy(du, env_eval, n_eval_episodes=10, max_ep_length=500)
        print(f"Before Learning: {mean_reward:.3f}")
        du.learn(total_timesteps=100_000, log_interval=1000)
        mean_reward, _ = evaluate_policy(du, env_eval, n_eval_episodes=10, max_ep_length=500)
        print(f"After Learning: {mean_reward:.3f}")
        du.save(path='./tests/model/du')
        
    if args.test:
        print_y("<< Testing DU Model >>")
        # Loading Model
        _du = DU.load(path='./tests/model/du')
        mean_reward, _ = evaluate_policy(_du, env_eval, n_eval_episodes=10, max_ep_length=500)
        print(f"Load Learning: {mean_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    main(args)
