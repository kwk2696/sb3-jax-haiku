import argparse
import pickle

import gym
from sb3_jax.common.buffers import OfflineBuffer
from moirl_jax.utils.utils import load_trajs

def collect(args):
    # Load Trajectories
    with open("datasets/dataset.pkl", "rb") as f:
        expert_traj = pickle.load(f) 
    # Check the buffer size
    buffer_size = 0 
    for i, traj in enumerate(expert_traj):
        for j in range(len(traj.acts)):
            buffer_size += 1
    print(f"Buffer Size: {buffer_size}")
    
    # Make env
    env = gym.make('Swimmer-v3')
    print(f"Obs: {env.observation_space.shape}, Act: {env.action_space.shape}")

    # Make OfflineBuffer
    buff = OfflineBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Convert Trajs to Offline Buffer     
    for _, traj in enumerate(expert_traj):
        obs = traj.obs[0]
        for i in range(len(traj.acts)):
            prev_obs = obs
            obs, act, rew, info = traj.obs[i+1], traj.acts[i], traj.rews[i], traj.infos[i]
            done = True if i == (len(traj.acts)-1) else False 
            buff.add(prev_obs, obs, act, rew, done, None)
    print(f"Buffer Size: {buff.buffer_size}")
     
    print(buff.observations[0][0])
    # Save the OfflineBuffer
    buff.save(path='datasets/offline_buffer.pkl')
    
    # Load Buffer
    _buff = OfflineBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    _buff = _buff.load(path='datasets/offline_buffer.pkl')

    print(_buff.observations[0][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--env", type=str, default="MoMountainCar-v1")
    parser.add_argument("--reward_type", nargs='+', type=int, action='store')
    args = parser.parse_args()
    
    collect(args)
