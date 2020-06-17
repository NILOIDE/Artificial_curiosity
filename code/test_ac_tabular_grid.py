# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import gym
import grid_gym  # Import necessary for GridWorld custom envs
from grid_gym.envs.grid_world import *
import numpy as np
import random
from utils.visualise import Visualise
from datetime import datetime
import matplotlib.pyplot as plt
import os
from param_parse import parse_args
from eval_wm import eval_wm
from modules.algorithms.DQN import TabularQlearning
from modules.world_models.world_model import TabularWorldModel, WorldModelNoEncoder, EncodedWorldModel
from modules.replay_buffers.replay_buffer import DynamicsReplayBuffer


def get_env_instance(env_name):
    if env_name == 'GridWorldBox11x11-v0':
        env = GridWorldBox11x11()
    elif env_name == 'GridWorldSpiral28x28-v0':
        env = GridWorldSpiral28x28()
    elif env_name == 'GridWorldSpiral52x50-v0':
        env = GridWorldSpiral52x50()
    elif env_name == 'GridWorld10x10-v0':
        env = GridWorld10x10()
    elif env_name == 'GridWorld25x25-v0':
        env = GridWorld25x25()
    elif env_name == 'GridWorld42x42-v0':
        env = GridWorld42x42()
    elif env_name == 'GridWorldSubspace50x50-v0':
        env = GridWorldSubspace50x50()
    else:
        raise ValueError('Wrong env_name.')
    return env


def draw_heat_map(visitation_count, t, folder_name):
    folder_name = folder_name + "/heat_maps/"
    os.makedirs(folder_name, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.power(visitation_count, 1 / 1), cmap='jet', vmin=0.0, vmax=0.01)
    ax.set_title(f'Heat map of visitation counts at t={t}')
    fig.colorbar(ax=ax, mappable=im, orientation='vertical')
    plt.savefig(f'{folder_name}{t}.png')
    plt.close()


def main(env, visualise, folder_name, **kwargs):
    obs_dim = tuple(env.observation_space.sample().shape)
    assert len(obs_dim) == 1, f'States should be 1D vector. Received: {obs_dim}'
    a_dim = (env.action_space.n,)
    alg = TabularQlearning(obs_dim, a_dim, kwargs['gamma'], kwargs['eps_min'])
    if kwargs['encoder_type'] == 'tab':
        wm = TabularWorldModel(obs_dim, a_dim, kwargs['wm_lr'])
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if kwargs['encoder_type'] == 'none':
            wm = WorldModelNoEncoder(obs_dim, a_dim, device=device, **kwargs)
        else:
            wm = EncodedWorldModel(obs_dim, a_dim, device=device, **args)
    ep_scores = {'DQN': [0.0], 'Mean intrinsic reward': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [], 'int': []}
    buffer = DynamicsReplayBuffer(get_env_instance(kwargs['env_name']).history_len)
    if kwargs['encoder_type'] == 'cont' and kwargs['gridworld_ns_pool'] == 'uniform':
        for s in get_env_instance(kwargs['env_name']).get_states():
            buffer.add(s, None, None, None)
    pe_map, q_map, walls_map = eval_wm(wm, alg, folder_name, kwargs['env_name'])
    visualise.eval_gridworld_iteration_update(pe_map=pe_map,
                                              q_map=q_map,
                                              walls_map=walls_map)

    while alg.train_steps < kwargs['train_steps']:
        s_t = env.reset()
        done = False
        while not done:
            a_t = alg.act(s_t)
            s_tp1, r_t, done, info = env.step(a_t)
            if kwargs['encoder_type'] == 'cont' and kwargs['gridworld_ns_pool'] == 'visited':
                buffer.add(s_t, None, None, None)
            if alg.train_steps < kwargs['train_steps']:
                r_int_t = wm.train(torch.from_numpy(s_t).to(dtype=torch.float32, device=device),
                                   torch.tensor([a_t], device=device),
                                   torch.from_numpy(s_tp1).to(dtype=torch.float32, device=device).unsqueeze(0),
                                   **{'memories': buffer}).cpu().item()
                alg.train(s_t, a_t, r_int_t, s_tp1)
                total_history['ext'].append(r_t)
                total_history['int'].append(r_int_t)
                if alg.train_steps % kwargs['export_interval'] == 0:
                    ep_scores['DQN'].append(np.mean(total_history['ext'][-500:]))
                    ep_scores['Mean intrinsic reward'].append(np.mean(total_history['int'][-500:]))
                    elapsed_time = (int((datetime.now() - start_time).total_seconds() // (60 * 60)),
                                    int((datetime.now() - start_time).total_seconds() % (60 * 60) // 60),
                                    int((datetime.now() - start_time).total_seconds() % 60))
                    print('--------------------------------------\n',
                          'Step:', alg.train_steps, '/', kwargs['train_steps'], '     ',
                          f'E(G): {ep_scores["DQN"][-1]:.3f}',
                          f'Eps: {alg.epsilon:.3f}',
                          'Time elapsed:', f'{elapsed_time[0]}:{elapsed_time[1]}:{elapsed_time[2]}',
                          'States explored:', len(alg.q_values))

                    visualise.train_iteration_update(ext=ep_scores['DQN'][-1],
                                                     int=ep_scores['Mean intrinsic reward'][-1],
                                                     **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []},
                                                     alg_loss=np.mean(alg.losses[-100:]),
                                                     info=info)
                if alg.train_steps % kwargs['eval_interval'] == 0:
                    print('Evaluating...')
                    if kwargs['env_name'][:9] == 'GridWorld':
                        draw_heat_map(info['density'], alg.train_steps, folder_name)
                        pe_map, q_map, walls_map = eval_wm(wm, alg, folder_name, kwargs['env_name'])
                        visualise.eval_gridworld_iteration_update(density_map=info['density'],
                                                                  pe_map=pe_map,
                                                                  q_map=q_map,
                                                                  walls_map=walls_map)
            s_t = s_tp1

    env.close()
    print('Environment closed.')
    visualise.close()
    print('Tensorboard writer closed.')


if __name__ == "__main__":
    args = parse_args()
    args['env_name'] = 'GridWorld42x42-v0'
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    run_name = f"{args['save_dir']}{args['env_name']}/{args['time_stamp']}_-_{args['name']}_{args['encoder_type']}/"
    print(run_name)
    environment = gym.make(args['env_name'])
    visualise = Visualise(run_name, **args)
    try:
        main(environment, visualise, run_name, **args)
    finally:
        environment.close()
        visualise.close()
