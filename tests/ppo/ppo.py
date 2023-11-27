import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
import argparse
import gym

from stable_baselines3.common import env_util, vec_env
from sb3_jax import PPO 
from sb3_jax.common.evaluation import evaluate_policy
from sb3_jax.common.norm_layers import RunningNormLayer
from sb3_jax.common.utils import print_y


def main(args):
    env_train = env_util.make_vec_env('HalfCheetah-v3')
    env_train = vec_env.VecNormalize(env_train, norm_obs=False, norm_reward=True)
    env_eval = env_util.make_vec_env('HalfCheetah-v3')
    
    if args.train:
        print_y("<< Training PPO Model >>")
        
        # Make PPO
        ppo = PPO(
            policy='MlpPolicy',
            env=env_train,
            learning_rate=3e-4,
            n_steps=512, 
            batch_size=64,
            n_epochs=20,
            gamma=0.98,
            gae_lambda=0.92,
            clip_range=0.1,
            max_grad_norm=0.8, 
            verbose=1,
            seed=777,
            wandb_log=dict(
                project='sb3-jax-haiku_tests',
                name='ppo',
            ),
            policy_kwargs=dict(
                log_std_init=-2, 
                net_arch=[dict(pi=[256,256], vf=[256,256])],
                normalization_class=RunningNormLayer
            )
        )

        # Evaluate POlict before learning
        mean_reward, _ = evaluate_policy(ppo, env_eval, n_eval_episodes=10)
        print(f"Before Learning: {mean_reward}")
        ppo.learn(total_timesteps=50_000, log_interval=10)
        mean_reward, _ = evaluate_policy(ppo, env_eval, n_eval_episodes=10)
        print(f"After Learning: {mean_reward}")
        ppo.save(path='./tests/model/ppo')
    
    if args.test:
        print_y("<< Testing PPO Model >>")
        # Loading Model
        _ppo = PPO.load(path='./tests/model/ppo')
        mean_reward, _ = evaluate_policy(_ppo, env_eval, n_eval_episodes=10)
        print(f"Load Learning: {mean_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    main(args)
