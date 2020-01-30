# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import torch.nn as nn
import gym
import numpy as np
from modules.replay_buffers.replay_buffer import ReplayBuffer
from modules.algorithms.DQN import DQN
from utils import resize_to_standard_dim_numpy, channel_first_numpy, INPUT_DIM, transition_to_torch
import time

def main(env):
    buffer = ReplayBuffer(5000)
    obs_dim = INPUT_DIM
    a_dim = (env.action_space.n,)
    alg = DQN(obs_dim, a_dim)

    while alg.train_steps < 50000:
        s_t = env.reset()
        s_t = channel_first_numpy(resize_to_standard_dim_numpy(s_t)) / 256  # Reshape and normalise input
        # env.render('human')
        done = False
        total = 0
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32))
            s_tp1, r_t, done, _ = env.step(a_t)
            total += r_t
            s_tp1 = channel_first_numpy(resize_to_standard_dim_numpy(s_tp1)) / 256  # Reshape and normalise input
            # env.render('human')
            buffer.add(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1
            batch = transition_to_torch(*buffer.sample(32))
            alg.train(*batch)
            if alg.train_steps % 100 == 0:
                print('Step:', alg.train_steps, 'WM loss:', alg.losses[-1], alg.epsilon)
            if done:
                break

        print(total)
    env.close()





if __name__ == "__main__":
    environment = gym.make('Breakout-v0')
    try:
        main(environment)
    finally:
        environment.close()
