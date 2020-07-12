# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import gym
import numpy as np
from modules.replay_buffers.replay_buffer import DynamicsReplayBuffer, ReplayBuffer
from modules.algorithms.DQN import DQN
from modules.world_models.world_model import EncodedWorldModel, WorldModelNoEncoder
from utils.utils import standardize_state, transition_to_torch_no_r
from utils.visualise import Visualise
from datetime import datetime
import os
from param_parse import parse_args
import shutil

np.set_printoptions(linewidth=400)
np.seterr(all='raise')


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


def step(env, a_t, s_t, obs_dim):
    """"
    Add the new frame to the top of the stack.
    """
    new_frame, r_t, done, info = env.step(a_t)
    new_frame = standardize_state(new_frame, obs_dim, grayscale=True)
    s_tp1 = np.concatenate((s_t[1:], new_frame), axis=0)  # Oldest frame is at index 0
    return s_tp1, r_t, done, info


def reset(env, obs_dim):
    new_frame = env.reset()
    new_frame = standardize_state(new_frame, obs_dim, grayscale=True)
    s_t = np.zeros(obs_dim)
    s_t[-1] = new_frame
    return s_t


def evaluate(env_name, alg, wm, obs_dim, n=3):
    print('Evaluating...', end='\r')
    eval_start = datetime.now()
    env = gym.make(env_name)
    returns = {'ext': [], 'int': [], 'len': []}
    for i in range(n):
        s_t = reset(env, obs_dim)
        s_t = torch.from_numpy(s_t).to(dtype=torch.float32)
        done = False
        total = {'ext': 0.0, 'int': 0.0, 'len': 0}
        while not done:
            a_t = alg.act(s_t, eval=True).item()
            s_tp1, r_ext_t, done, info = step(env, a_t, s_t, obs_dim)
            s_tp1 = torch.from_numpy(s_tp1).to(dtype=torch.float32)
            r_int_t = wm.forward(s_t, torch.tensor([a_t, ]), s_tp1)
            s_t = s_tp1
            total['ext'] += r_ext_t
            total['int'] += r_int_t.item()
            total['len'] += 1
        returns['ext'].append(total['ext'])
        returns['int'].append(total['int'])
        returns['len'].append(total['len'])
    env.close()
    print(f'Evaluation:  Returns: {returns["ext"]}\n',
          f'Int_returns: {returns["int"]}\n',
          f'Lengths: {returns["len"]}\n',
          f'Eval time: {(datetime.now() - eval_start).total_seconds()}s.\n',
          f'Num episodes: {n}')
    return {'ext': np.mean(returns['ext']), 'int': np.mean(returns['int'])}


def fill_buffer(env, alg, buffer, obs_dim, **kwargs):
    print('Filling buffer')
    eval_start = datetime.now()
    while len(buffer) < kwargs['buffer_size']:
        s_t = reset(env, obs_dim)
        done = False
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32), eps=1.0).item()
            s_tp1, r_t, done, info = step(env, a_t, s_t, obs_dim)
            buffer.add(s_t, a_t, s_tp1, done)
            if len(buffer) >= kwargs['buffer_size']:
                break
            s_t = s_tp1
    print(f'Buffer fill time: {(datetime.now() - eval_start).total_seconds()}s.')


def main(env, visualise, folder_name, **kwargs):
    shutil.copyfile(os.path.abspath(__file__), folder_name + 'test_ac_tabular_grid.py')
    shutil.copyfile(os.path.dirname(os.path.realpath(__file__)) + '/modules/world_models/world_model.py',
                    folder_name + 'world_model.py')
    buffer = DynamicsReplayBuffer(kwargs['buffer_size'])
    obs_dim = (kwargs['frame_stack'] if kwargs['grayscale'] else 3 * kwargs['frame_stack'], *kwargs['resize_dim'])
    # obs_dim = env.observation_space.sample().shape
    assert len(obs_dim) == 3, 'States should be image (C, W, H).'
    assert (obs_dim[0] == kwargs['frame_stack'] and kwargs['grayscale']) or \
           (obs_dim[0] == 3 * kwargs['frame_stack'] and not kwargs['grayscale']), \
        f'Expected channels first. Received: {obs_dim}'
    a_dim = (env.action_space.n,)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    alg = DQN(obs_dim, a_dim, device=device, **kwargs)
    if kwargs['encoder_type'] == 'none':
        raise NotImplementedError
        wm = WorldModelNoEncoder(obs_dim, a_dim, device=device, **kwargs)
    else:
        wm = EncodedWorldModel(obs_dim, a_dim, device=device, **kwargs)
    intr_rew_norm = set_intr_rew_norm_type(kwargs['intr_rew_norm_type'])
    ep_scores = {'DQN': [0.0], 'Mean intrinsic reward': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [0.0], 'len': [], 'int': {'mean': {'list': [], 'running_mean': 0.0},
                                                      'std': {'list': [], 'running_mean': 0.0},
                                                      'min': {'list': [], 'running_mean': 0.0},
                                                      'max': {'list': [], 'running_mean': 0.0}}}
    fill_buffer(env, alg, buffer, obs_dim, **kwargs)
    visualise.eval_iteration_update(**evaluate(kwargs['env_name'], alg, wm, obs_dim))
    print('Training...')
    while alg.train_steps < kwargs['train_steps']:
        s_t = reset(env, obs_dim)
        done = False
        total = [0, 0]
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32)).item()
            s_tp1, r_t, done, info = step(env, a_t, s_t, obs_dim)
            total[0] += r_t
            total[1] += 1
            buffer.add(s_t, a_t, s_tp1, done)
            if alg.train_steps < kwargs['train_steps']:
                batch = buffer.sample(kwargs['batch_size'])
                obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch_no_r(*batch)
                r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
                intr_reward_bookkeeping(r_int_t, total_history, intr_rew_norm, kwargs['intr_rew_mean_n'])
                if intr_rew_norm is not None:
                    r_int_t = normalize_rewards(r_int_t, total_history, intr_rew_norm)
                alg.train(obs_t_batch, a_t_batch, r_int_t, obs_tp1_batch, dones_batch)

                if done:
                    total_history['ext'].append(total[0])
                    total_history['len'].append(total[1])
                if alg.train_steps % kwargs['export_interval'] == 0:
                    ep_scores['DQN'].append(np.mean(total_history['ext'][-10:]))
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
                                                     alg_loss=np.mean(alg.losses[-100:]))
                if alg.train_steps % kwargs['eval_interval'] == 0:
                    wm.save(f'{folder_name}saved_objects/')
                    alg.save(f'{folder_name}saved_objects/')
                    visualise.eval_iteration_update(**evaluate(kwargs['env_name'], alg, wm, obs_dim))
            s_t = s_tp1

    env.close()
    print('Environment closed.')
    visualise.close()
    print('Tensorboard writer closed.')


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args['seed'])
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
