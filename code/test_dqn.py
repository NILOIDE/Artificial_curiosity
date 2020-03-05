# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import gym
import numpy as np
from modules.replay_buffers.replay_buffer import ReplayBuffer
from modules.algorithms.DQN import DQN
from utils.utils import standardize_state, clip_rewards, transition_to_torch,\
    plot_list_in_dict
import random
from datetime import datetime



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


def evaluate(env_name, alg):
    env = gym.make(env_name)
    returns = []
    print('Evaling')
    for i in range(5):
        s_t = env.reset()
        done = False
        total = 0.0
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32), eval=True)
            s_tp1, r_t, done, info = env.step(a_t)
            total += r_t
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main(env, env_name):
    buffer = ReplayBuffer(50000)
    # obs_dim = (4,84,84)#INPUT_DIM
    obs_dim = (len(env.observation_space.sample()),)
    a_dim = (env.action_space.n,)
    train_steps = int(1e6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alg = DQN(obs_dim, a_dim, train_steps, device=device)
    ep_scores = {'DQN': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [0]}
    while alg.train_steps < train_steps:
        # s_t = random_start(env, a_dim)
        s_t = env.reset()
        s_t = s_t / 256.0
        # env.render('human')
        done = False
        total = [0, 0]
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32))
            # s_tp1, r_t, done = step(env, a_t, s_t)
            s_tp1, r_t, done, info = env.step(a_t)
            for _ in range(3):
                s_tp1, r_t, done, info = env.step(a_t)
            s_tp1 = s_tp1 / 256.0
            total[0] += r_t
            total[1] += 1
            r_t = clip_rewards(r_t)
            if alg.train_steps < train_steps:
                buffer.add(s_t, a_t, r_t, s_tp1, done)
                s_t = s_tp1
                alg.train(*transition_to_torch(*buffer.sample(64)[:5]))
                if alg.train_steps % 1000 == 0:
                    ep_scores['DQN'].append(np.mean(total_history['ext'][-10:]))
                    elapsed_time = (int((datetime.now() - start_time).total_seconds() // (60*60)),
                                    int((datetime.now() - start_time).total_seconds() // (60*60) // 60))
                    print('Step:', alg.train_steps, '/', train_steps,
                          'E(G):', np.mean(ep_scores['DQN']),
                          'Eps:', alg.epsilon,
                          'Time elapsed:', str(elapsed_time[0]) + ':' + str(elapsed_time[1]))
                    alg.save(path='saved_objects/DQN.pt')
                    plot_list_in_dict({'DQN': ep_scores['DQN']}, y_low=0,
                                      y_label='Episode extrinsic return',
                                      x_label='Training steps',
                                      title=env_name + ' trained on extrinsic rewards',
                                      x_interval=1000,
                                      path='../plots/' + env_name + '_ext_only.png')
            if done:
                total_history['ext'].append(total[0])
                break
    env.close()


if __name__ == "__main__":
    name = 'Breakout-ram-v0'
    environment = gym.make(name)
    try:
        main(environment, name)
    finally:
        environment.close()
