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
import shutil
from param_parse import parse_args
from eval_wm import eval_wm
from modules.algorithms.DQN import TabularQlearning, DQN
from modules.world_models.world_model import TabularWorldModel, WorldModelNoEncoder, EncodedWorldModel
from modules.replay_buffers.replay_buffer import DynamicsReplayBuffer
import matplotlib.pyplot as plt


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


def warmup_enc_all_states(wm, env, start_time, buffer=None, device='cpu', **kwargs):

    def check_neigh_dists(states, neighbours):
        dist = []
        for s, neigh in zip(states, neighbours):
            s = torch.from_numpy(s).to(dtype=torch.float32, device=device).reshape((1, -1))
            for n in neigh:  # States don't necessarily have 4 neighbors (if self is neighbour, it is omitted)
                n = torch.from_numpy(n).to(dtype=torch.float32, device=device).reshape((1, -1))
                dist.append((wm.encode(s) - wm.encode(n)).pow(2).sum().item())
        dist_norm = np.array(dist) / np.mean(dist)
        elapsed_time = (int((datetime.now() - start_time).total_seconds() // (60 * 60)),
                        int((datetime.now() - start_time).total_seconds() % (60 * 60) // 60),
                        int((datetime.now() - start_time).total_seconds() % 60))
        print(np.mean(dist), np.std(dist), np.std(dist_norm), elapsed_time)
        print(t, end='\r')
    states, neighbours = env.get_states_with_neighbours()
    t = -kwargs['wm_warmup_steps'] + 1

    check_neigh_dists(states, neighbours)
    if kwargs['wm_warmup_steps'] == 0:
        return
    print('Warming up world model...')
    while t < 0:
        # for s in states:
        for s, neigh in zip(states, neighbours):
            assert s.sum() == 1.0
            if buffer is None:
                wm.train_contrastive_encoder(torch.from_numpy(s).to(dtype=torch.float32, device=device),
                                             torch.from_numpy(np.array(states)).to(dtype=torch.float32, device=device),
                                             torch.from_numpy(neigh).to(dtype=torch.float32, device=device))
            else:
                wm.train_contrastive_encoder(torch.from_numpy(s).to(dtype=torch.float32, device=device),
                                             buffer,
                                             torch.from_numpy(neigh).to(dtype=torch.float32, device=device))
            t += 1
            if t >= 0:
                break
        check_neigh_dists(states, neighbours)
        if t >= 0:
            break


def main(env, visualise, folder_name, **kwargs):

    shutil.copyfile(os.path.abspath(__file__), folder_name + 'test_ac_tabular_grid.py')
    shutil.copyfile(os.path.dirname(os.path.realpath(__file__)) + '/modules/world_models/world_model.py',
                    folder_name + 'world_model.py')
    obs_dim = tuple(env.observation_space.sample().shape)
    assert len(obs_dim) == 1, f'States should be 1D vector. Received: {obs_dim}'
    a_dim = (env.action_space.n,)
    alg = TabularQlearning(obs_dim, a_dim, kwargs['gamma'], kwargs['eps_min'])
    if kwargs['encoder_type'] == 'tab':
        wm = TabularWorldModel(obs_dim, a_dim, kwargs['wm_lr'], **kwargs)
        device = 'cpu'
    else:
        device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
        if kwargs['encoder_type'] == 'none':
            wm = WorldModelNoEncoder(obs_dim, a_dim, device=device, **kwargs)
        else:
            wm = EncodedWorldModel(obs_dim, a_dim, device=device, **args)
    ep_scores = {'DQN': [0.0], 'Mean intrinsic reward': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [], 'int': []}
    cont_buffer = DynamicsReplayBuffer(get_env_instance(kwargs['env_name']).history_len)
    cont_visited = False if not kwargs['gridworld_ns_pool'] == 'visited' else True
    cont_visited_uniform = False if not kwargs['gridworld_ns_pool'] == 'visited_uniform' else True
    if kwargs['encoder_type'] == 'cont' and kwargs['gridworld_ns_pool'] == 'uniform':
        # cont_buffer = torch.from_numpy(get_env_instance(kwargs['env_name']).get_states()).to(dtype=torch.float32)
        for s in get_env_instance(kwargs['env_name']).get_states():
            cont_buffer.add(s, None, None, None)
    wm.save(folder_name + 'saved_objects/')
    if kwargs['encoder_type'] == 'cont':
        # wm.load_encoder('final_results/GridWorld42x42-v0/2020-07-12_14-15-38-185893_-_zdim16_hdim64_eps01_envLoopAround_encPretrain2M_noEncTrainlr-3_test_cont_1/' + 'saved_objects/trained_encoder.pt')
        # warmup_enc(env, alg, wm, cont_buffer, visualise, device, **kwargs)
        warmup_enc_all_states(wm, env, start_time, buffer=cont_buffer, device=device, **kwargs)
        wm.save_encoder(folder_name + 'saved_objects/')
    pe_map, q_map, walls_map = eval_wm(wm, alg, folder_name, kwargs['env_name'])
    visualise.eval_gridworld_iteration_update(pe_map=pe_map,
                                              q_map=q_map,
                                              walls_map=walls_map)
    while alg.train_steps < kwargs['train_steps']:
        s_t = env.reset()
        done = False
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32, device=device))
            s_tp1, r_t, done, info = env.step(a_t)
            if kwargs['encoder_type'] == 'cont' and cont_visited:
                cont_buffer.add(s_t, None, None, None)
            elif kwargs['encoder_type'] == 'cont' and cont_visited_uniform:
                cont_buffer = torch.from_numpy(info['visited_states']).to(dtype=torch.float32)
            if alg.train_steps < kwargs['train_steps']:
                r_int_t = wm.train(torch.from_numpy(s_t).to(dtype=torch.float32, device=device),
                                   torch.tensor([a_t], device=device),
                                   torch.from_numpy(s_tp1).to(dtype=torch.float32, device=device).unsqueeze(0),
                                   **{'memories': cont_buffer, 'distance': info['distance']}).cpu().item()
                # r_int_t = wm.train_contrastive_fm(torch.from_numpy(s_t).to(dtype=torch.float32, device=device),
                #                                   torch.tensor([a_t], device=device),
                #                                   torch.from_numpy(s_tp1).to(dtype=torch.float32,
                #                                                              device=device).unsqueeze(0),
                #                                   **{'distance': info['distance']}).cpu().item()
                alg.train(s_t, a_t, r_int_t, s_tp1, False)
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
                          'Explored:', len(alg.q_values))

                    visualise.train_iteration_update(ext=ep_scores['DQN'][-1],
                                                     int=ep_scores['Mean intrinsic reward'][-1],
                                                     **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []},
                                                     alg_loss=np.mean(alg.losses[-100:]),
                                                     info=info)
                if alg.train_steps % kwargs['eval_interval'] == 0:
                    if kwargs['env_name'][:9] == 'GridWorld':
                        draw_heat_map(info['density'], alg.train_steps, folder_name)
                        pe_map, q_map, walls_map = eval_wm(wm, alg, folder_name, kwargs['env_name'])
                        visualise.eval_gridworld_iteration_update(density_map=info['density'],
                                                                  pe_map=pe_map,
                                                                  q_map=q_map,
                                                                  walls_map=walls_map)
                    wm.save(folder_name + 'saved_objects/')
            s_t = s_tp1

            if alg.train_steps % 50000 == 0:
                for d in range(2, 20):
                    to_plot = []
                    for i, li in enumerate(wm.state_wise_loss_diff):
                        if wm.state_wise_loss[li]['d'] != d:
                            continue
                        if len(wm.state_wise_loss[li]['list']) < 10:
                            continue
                        to_plot.append(wm.state_wise_loss[li]['list'][3:500])
                    if to_plot:
                        plt.figure()
                        for li in to_plot:
                            plt.plot(li)
                            plt.ylim(bottom=0.0)
                        plt.title(kwargs['env_name'] + "_" + kwargs['encoder_type'] + '_distance:' + str(d))
                        plt.savefig(run_name + kwargs['encoder_type'] + '_distance' + str(d) + '.png')
                        plt.close()


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
    run_name = f"{args['save_dir']}{args['env_name']}/{args['time_stamp']}_-_{args['name']}_{args['encoder_type']}_{args['seed']}/"
    print(run_name)
    environment = gym.make(args['env_name'])
    visualise = Visualise(run_name, **args)
    try:
        main(environment, visualise, run_name, **args)
    finally:
        environment.close()
        visualise.close()
