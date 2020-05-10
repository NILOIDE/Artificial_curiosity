# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import gym
import numpy as np
from modules.replay_buffers.replay_buffer import DynamicsReplayBuffer, ReplayBuffer
from modules.algorithms.DQN import DQN
from modules.world_models.world_model import EncodedWorldModel, WorldModelNoEncoder
from utils.utils import standardize_state, transition_to_torch
from utils.visualise import Visualise
from datetime import datetime
import os
from param_parse import parse_args
np.set_printoptions(linewidth=400)
np.seterr(all='raise')


def step(env, a_t, s_t, obs_dim):
    """"
    Add the new frame to the top of the stack.
    """
    new_frame, r_t, done, _ = env.step(a_t)
    new_frame = standardize_state(new_frame, obs_dim, grayscale=True)
    s_tp1 = np.concatenate((s_t[1:], new_frame), axis=0)  # Oldest frame is at index 0
    return s_tp1, r_t, done


def reset(env, obs_dim):
    new_frame = env.reset()
    new_frame = standardize_state(new_frame, obs_dim, grayscale=True)
    s_t = np.zeros(obs_dim)
    s_t[-1] = new_frame
    return s_t


def evaluate(env_name, alg, wm, obs_dim, n=3):
    print('Evaluating...', end='\r')
    env = gym.make(env_name)
    returns = {'ext': [], 'int': [], 'len': []}
    eval_start = datetime.now()
    for i in range(n):
        s_t = env.reset()
        s_t = torch.from_numpy(s_t).to(dtype=torch.float32)
        s_t /= 256.0
        done = False
        total = {'ext': 0.0, 'int': 0.0, 'len': 0}
        while not done:
            print(s_t.shape)
            a_t = alg.act(s_t, eval=True).item()
            print(s_t.shape)
            s_tp1, r_ext_t, done, _ = env.step(a_t)
            s_tp1 = torch.from_numpy(s_tp1).to(dtype=torch.float32)
            s_tp1 /= 256.0
            # r_int_t = wm.forward(s_t, torch.tensor([a_t,]), s_tp1)
            s_t = s_tp1
            total['ext'] += r_ext_t
            # total['int'] += r_int_t.item()
            total['len'] += 1
        returns['ext'].append(total['ext'])
        returns['int'].append(total['int'])
    env.close()
    print(f'Evaluation:  Returns: {returns["ext"]}\n',
          f'Int_returns: {returns["int"]}\n',
          f'Lengths: {returns["len"]}\n',
          f'Eval time: {int((datetime.now() - eval_start).total_seconds())}s.',
          f'Num episodes: {n}')
    return {'ext': np.mean(returns['ext']), 'int': np.mean(returns['int'])}


def fill_buffer(env, alg, buffer, obs_dim, **kwargs):
    print('Filling buffer')
    while len(buffer) < kwargs['buffer_size']:
        s_t = env.reset()
        s_t /= 256.0
        done = False
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32), eps=1.0).item()
            s_tp1, r_t, done, _ = env.step(a_t)
            s_tp1 /= 256.0
            buffer.add(s_t, a_t, r_t, s_tp1, done)
            if len(buffer) >= kwargs['buffer_size']:
                break
            s_t = s_tp1


def warmup_wm(alg, wm, buffer, visualise, **kwargs):
    """"
    Used to warmup the environment visitation history.
    """
    print('Warming up world model...')
    for i in range(-kwargs['wm_warmup_steps'] + 1, 0 + 1):
        batch = buffer.sample(kwargs['batch_size'])
        obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch(*batch)
        r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
        if alg.train_steps % kwargs['interval'] == 0:
            visualise.eval_wm_warmup(i, **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []})


def main(env, visualise, folder_name, **kwargs):
    buffer = ReplayBuffer(kwargs['buffer_size'])
    obs_dim = env.observation_space.sample().shape
    a_dim = (env.action_space.n,)
    device = 'cpu'
    print('Device:', device)
    alg = DQN(obs_dim, a_dim, device=device, **kwargs)
    if kwargs['encoder_type'] == 'none':
        wm = WorldModelNoEncoder(obs_dim, a_dim, device=device, **kwargs)
    else:
        wm = EncodedWorldModel(obs_dim, a_dim, device=device, **kwargs)
    ep_scores = {'DQN': [0.0], 'Mean intrinsic reward': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [0.0], 'int': {'mean': [], 'std': [], 'min': [], 'max': []}, 'len': []}
    fill_buffer(env, alg, buffer, obs_dim, **kwargs)
    visualise.eval_iteration_update(**evaluate(kwargs['env_name'], alg, wm, obs_dim))
    if kwargs['wm_warmup_steps'] > 0:
        warmup_wm(alg, wm, buffer, visualise, **kwargs)
    print('Training...')
    while alg.train_steps < kwargs['train_steps']:
        s_t = env.reset()
        # s_t /= 256.0
        # env.render('human')
        done = False
        total = [0, 0]
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32)).item()
            s_tp1, r_t, done, _ = env.step(a_t)
            # s_tp1 /= 256.0
            total[0] += r_t
            total[1] += 1
            buffer.add(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1
            # env.render('human')
            if alg.train_steps < kwargs['train_steps']:
                batch = buffer.sample(kwargs['batch_size'])
                obs_t_batch, a_t_batch, r_t_batch, obs_tp1_batch, dones_batch = transition_to_torch(*batch)
                r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
                total_history['int']['mean'].append(r_int_t.mean().item())

                alg.train(obs_t_batch, a_t_batch, r_t_batch, obs_tp1_batch, dones_batch)

                if done:
                    total_history['ext'].append(total[0])
                    total_history['len'].append(total[1])
                if alg.train_steps % kwargs['export_interval'] == 0:
                    ep_scores['DQN'].append(np.mean(total_history['ext'][-10:]))
                    ep_scores['Mean intrinsic reward'].append(np.mean(total_history['int']['mean'][-1000:]))
                    elapsed_time = (int((datetime.now() - start_time).total_seconds() // (60 * 60)),
                                    int((datetime.now() - start_time).total_seconds() % (60 * 60) // 60),
                                    int((datetime.now() - start_time).total_seconds() % 60))
                    print('--------------------------------------\n',
                          'Step:', alg.train_steps, '/', kwargs['train_steps'], '     ',
                          f'E(G): {ep_scores["DQN"][-1]:.3f}',
                          f'Eps: {alg.epsilon:.3f}',
                          'Time elapsed:', f'{elapsed_time[0]}:{elapsed_time[1]}:{elapsed_time[2]}')

                    os.makedirs(folder_name + 'objects/', exist_ok=True)
                    wm.save(path=f'{folder_name}objects/WM.pt')
                    alg.save(path=f'{folder_name}objects/DQN.pt')
                    visualise.train_iteration_update(ext=ep_scores['DQN'][-1],
                                                     int=ep_scores['Mean intrinsic reward'][-1],
                                                     **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []},
                                                     alg_loss=np.mean(alg.losses[-100:]))
                if alg.train_steps % kwargs['eval_interval'] == 0:
                    visualise.eval_iteration_update(**evaluate(kwargs['env_name'], alg, wm, obs_dim))

    print('Environment closed.')
    visualise.close()
    print('Tensorboard writer closed.')


if __name__ == "__main__":
    args = parse_args()
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
