import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.32'
import argparse
import gym

from stable_baselines3.common import env_util, vec_env
from sb3_jax.sac import SAC
from sb3_jax.common.evaluation import evaluate_policy
from sb3_jax.common.norm_layers import RunningNormLayer
from sb3_jax.common.utils import print_y


def main(args):
    env = env_util.make_vec_env('HalfCheetah-v3')
    env = vec_env.VecNormalize(env, norm_obs=True, norm_reward=True)

    if args.train:
        print_y("<< Training SAC Model >>")
        sac = SAC(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            batch_size=256,
            wandb_log=dict(
                project='sb3-jax-haiku_tests',
                name='sac',
            ),
            policy_kwargs=dict(
                #normalization_class=RunningNormLayer,
                net_arch=dict(pi=[256,256], qf=[256,256])
            ),
            verbose=True,
        )

        # check entropy coefficient
        log_ent_coef = sac.log_ent_coef(sac.ent_coef_params)
        print(f"Log Entropy Coef: {log_ent_coef}")

        mean_reward, _ = evaluate_policy(sac, env, n_eval_episodes=10)
        print(f"Before Learning: {mean_reward}")
        sac.learn(total_timesteps=100_000, log_interval=1)
        env.norm_reward = False
        mean_reward, _ = evaluate_policy(sac, env, n_eval_episodes=10)
        print(f"After Learning: {mean_reward}")
        sac.save(path='../model/sac')
    
    if args.test:
        print_y("<< Testing SAC Model >>")
        # Loading Policy
        _sac = SAC.load(path='../model/sac')
        mean_reward, _ = evaluate_policy(_sac, env, n_eval_episodes=10)
        print(f"Load Learning: {mean_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    main(args)
