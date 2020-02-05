# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import torch.nn as nn
import gym
import numpy as np
from modules.replay_buffers.replay_buffer import ReplayBuffer
from modules.algorithms.DQN import DQN
from utils import standardize_state, clip_rewards, INPUT_DIM, transition_to_torch
import random


def random_start(env, a_dim):
    frame_skip = random.randint(0,3)
    s_t = np.zeros((4, 84, 84))
    s = env.reset()
    s = standardize_state(s, grayscale=True)  # Reshape and normalise input
    s_t[-frame_skip-1:-frame_skip] = s
    for i in range(1, 1 + frame_skip):
        s_tp1, _, _, _ = env.step(random.randint(0, a_dim[0]-1))
        s_tp1 = standardize_state(s_tp1, grayscale=True)  # Reshape and normalise input
        s_t[-frame_skip+i-1:-frame_skip+i] = s_tp1
    return s_t


def sticky_step(env, a_t, fsr=4):
    s_tp1 = np.zeros((fsr, 84, 84))
    r_t = 0
    s, r, done, _ = env.step(a_t)
    s_tp1[:1] = standardize_state(s, grayscale=True)
    r_t += r
    for i in range(1, fsr):
        s, r, done, _ = env.step(a_t)
        s_tp1[i:i+1] = standardize_state(s, grayscale=True)   # Oldest frame is at index 0
        r_t += r
    return s_tp1, r_t, done


def step(env, a_t, s_t):
    """"
    Add the new frame to the top of the stack.
    """
    new_frame, r_t, done, _ = env.step(a_t)
    new_frame = standardize_state(new_frame, grayscale=True)
    s_tp1 = np.concatenate((s_t[1:], new_frame), axis=0)  # Oldest frame is at index 0
    return s_tp1, r_t, done


def main(env):
    buffer = ReplayBuffer(50000)
    obs_dim = (4,84,84)#INPUT_DIM
    a_dim = (env.action_space.n,)
    train_steps = 5e6
    alg = DQN(obs_dim, a_dim, train_steps, device='cuda')
    while alg.train_steps < train_steps:
        s_t = random_start(env, a_dim)
        # env.render('human')
        done = False
        total = 0
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32))
            s_tp1, r_t, done = step(env, a_t, s_t)
            total += r_t
            r_t = clip_rewards(r_t)
            # env.render('human')
            buffer.add(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1
            batch = transition_to_torch(*buffer.sample(64))
            alg.train(*batch)
            if alg.train_steps % 1000 == 0:
                print('Step:', alg.train_steps, 'WM loss:', alg.losses[-1], 'Eps:', alg.epsilon)
            if done:
                print(total)
                break
    env.close()


if __name__ == "__main__":
    environment = gym.make('Breakout-v0')
    try:
        main(environment)
    finally:
        environment.close()
