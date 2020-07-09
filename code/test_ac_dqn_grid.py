# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import gym
import grid_gym  # Import necessary for GridWorld custom envs
import numpy as np
from modules.replay_buffers.replay_buffer import DynamicsReplayBuffer, ReplayBuffer
from modules.algorithms.DQN import DQN
from modules.world_models.world_model import EncodedWorldModel, WorldModelNoEncoder
from utils.utils import transition_to_torch_no_r, CONV_LAYERS2014
from utils.visualise import Visualise
from datetime import datetime
import matplotlib.pyplot as plt
import os
from param_parse import parse_args
from eval_wm import eval_wm
np.set_printoptions(linewidth=400)




def set_intr_rew_norm_type(type):
    """"
    Rather than having a string to compare against (O(n)) every time we want to check which normalization
    type we should follow, have a dict of booleans (O(1)).
    """
    if type == 'none':
        return None
    d = {'max': False, 'whiten': False, 'history': False}
    if type == 'max':
        d['max'] = True
    elif type == 'max_history':
        d['max'] = True
        d['history'] = True
    elif type == 'whiten':
        d['whiten'] = True
    elif type == 'whiten_history':
        d['whiten'] = True
        d['history'] = True
    else:
        raise ValueError('Reward normalization type not recognized')
    return d


def intr_reward_bookkeeping(r_int_t, history, intr_rew_norm, n):
    def update_mean(new_value, d, n):
        if len(d['list']) >= n:
            d['running_mean'] -= d['list'][-n] / n
        d['list'].append(new_value)
        d['running_mean'] += new_value / n

    update_mean(r_int_t.mean().item(), history['int']['mean'], n)
    if intr_rew_norm is not None:
        if intr_rew_norm['history']:
            if intr_rew_norm['max']:
                update_mean(r_int_t.max().item(), history['int']['max'], n)
                update_mean(r_int_t.min().item(), history['int']['min'], n)
            elif intr_rew_norm['whiten']:
                update_mean(r_int_t.std().item(), history['int']['std'], n)


def normalize_rewards(r_int_t, history, norm_type):
    if norm_type['max']:
        if norm_type['history']:
            r_min, r_max = history['int']['min']['running_mean'], history['int']['max']['running_mean']
        else:
            r_min, r_max = r_int_t.min(), r_int_t.max()
        r_range = r_max - r_min + 1e-10
        return (r_int_t - r_min) / r_range
    if norm_type['whiten']:
        if norm_type['history']:
            r_mean = history['int']['mean']['running_mean']
            r_std = (history['int']['std']['running_mean'] + 1e-8) if len(history['int']['std']['list']) > 1 else 1.0
        else:
            r_mean = history['int']['mean']['list'][-1]  # Mean is already calculated during book keeping
            r_std = r_int_t.std()
        return (r_int_t - r_mean) / r_std


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


def fill_buffer(env, alg, buffer, **kwargs):
    print('Filling buffer')
    while len(buffer) < kwargs['buffer_size']:
        s_t = env.reset()
        done = False
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32), eps=1.0).item()
            s_tp1, r_t, done, info = env.step(a_t)
            buffer.add(s_t, a_t, s_tp1, done)
            if len(buffer) >= kwargs['buffer_size']:
                break
            s_t = s_tp1


def warmup_env(env, alg, wm, total_history):
    """"
    Used to warmup the environment visitation history.
    """
    print('Warming up...')
    for _ in range(1):
        s_t = env.reset()
        s_t = torch.from_numpy(s_t).reshape((1, -1)).to(dtype=torch.float32)
        done = False
        while not done:
            a_t = alg.act(s_t).item()
            s_tp1, r_t, done, info = env.step(a_t)
            a_t = torch.tensor([a_t, ]).reshape((1, -1)).to(dtype=torch.long)
            s_tp1 = torch.from_numpy(s_tp1).reshape((1, -1)).to(dtype=torch.float32)
            r_int_t = wm.forward(s_t, a_t, s_tp1)
            total_history['ext'].append(r_t)
            total_history['int'].append(r_int_t.mean().item())
            s_t = s_tp1
    print(info['counts'])


def warmup_wm(alg, wm, buffer, visualise, **kwargs):
    """"
    Used to warmup the environment visitation history.
    """
    print('Warming up world model...')
    for i in range(-kwargs['wm_warmup_steps'] + 1, 0 + 1):
        batch = buffer.sample(kwargs['batch_size'])
        obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch_no_r(*batch)
        r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
        if alg.train_steps % kwargs['interval'] == 0:
            visualise.eval_wm_warmup(i, **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []})


def main(env, visualise, folder_name, **kwargs):
    buffer = DynamicsReplayBuffer(kwargs['buffer_size'])
    obs_dim = tuple(env.observation_space.sample().shape)
    assert len(obs_dim) == 1, f'States should be 1D vector. Received: {obs_dim}'
    a_dim = (env.action_space.n,)
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    alg = DQN(obs_dim, a_dim, device=device, **kwargs)
    if kwargs['encoder_type'] == 'none':
        wm = WorldModelNoEncoder(obs_dim, a_dim, device=device, **kwargs)
    else:
        wm = EncodedWorldModel(obs_dim, a_dim, device=device, **args)
    intr_rew_norm = set_intr_rew_norm_type(kwargs['intr_rew_norm_type'])
    ep_scores = {'DQN': [0.0], 'Mean intrinsic reward': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [0.0], 'len': [], 'int': {'mean': {'list': [], 'running_mean': 0.0},
                                                      'std': {'list': [], 'running_mean': 0.0},
                                                      'min': {'list': [], 'running_mean': 0.0},
                                                      'max': {'list': [], 'running_mean': 0.0}}}
    fill_buffer(env, alg, buffer, **kwargs)
    if kwargs['wm_warmup_steps'] > 0:
        warmup_wm(alg, wm, buffer, visualise, **kwargs)
    while alg.train_steps < kwargs['train_steps']:
        s_t = env.reset()
        done = False
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32)).item()
            s_tp1, r_t, done, info = env.step(a_t)
            buffer.add(s_t, a_t, s_tp1, done)
            if alg.train_steps < kwargs['train_steps']:
                batch = buffer.sample(kwargs['batch_size'])
                obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch_no_r(*batch)
                r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
                intr_reward_bookkeeping(r_int_t, total_history, intr_rew_norm, kwargs['intr_rew_mean_n'])
                if intr_rew_norm is not None:
                    r_int_t = normalize_rewards(r_int_t, total_history, intr_rew_norm)
                alg.train(obs_t_batch, a_t_batch, r_int_t, obs_tp1_batch, dones_batch)
                total_history['ext'].append(r_t)
            s_t = s_tp1
            if alg.train_steps % kwargs['export_interval'] == 0:
                ep_scores['DQN'].append(np.mean(total_history['ext'][-500:]))
                ep_scores['Mean intrinsic reward'].append(total_history['int']['mean']['running_mean'])
                elapsed_time = (int((datetime.now() - start_time).total_seconds() // (60 * 60)),
                                int((datetime.now() - start_time).total_seconds() % (60 * 60) // 60),
                                int((datetime.now() - start_time).total_seconds() % 60))
                print('--------------------------------------\n',
                      'Step:', alg.train_steps, '/', kwargs['train_steps'], '     ',
                      f'E(G): {ep_scores["DQN"][-1]:.3f}',
                      f'Eps: {alg.epsilon:.3f}',
                      'Time elapsed:', f'{elapsed_time[0]}:{elapsed_time[1]}:{elapsed_time[2]}')

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
    env.close()
    print('Environment closed.')
    visualise.close()
    print('Tensorboard writer closed.')


if __name__ == "__main__":
    args = parse_args()
    args['env_name'] = 'GridWorld42x42-v0'
    np.random.seed(args['seed'])
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

