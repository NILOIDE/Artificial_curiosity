# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import gym
import numpy as np
from modules.replay_buffers.replay_buffer import DynamicsReplayBuffer, ReplayBuffer
from modules.algorithms.DQN import DQN
from modules.world_models.world_model import EncodedWorldModel, WorldModelNoEncoder
from utils.utils import standardize_state, transition_to_torch_no_r, CONV_LAYERS2014
from utils.visualise import Visualise
from datetime import datetime
import os
from param_parse import parse_args
np.set_printoptions(linewidth=400)


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
    for i in range(1):
        s_t = env.reset()
        done = False
        total = 0.0
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32), eval=True)
            s_tp1, r_t, done, info = env.step(a_t)
            total += r_t
        returns.append(total)
    env.close()
    return np.mean(returns)


def fill_buffer(env, alg, buffer, **kwargs):
    print('Filling buffer')
    while len(buffer) < kwargs['buffer_size']:
        s_t = env.reset()
        # s_t = torch.from_numpy(s_t).reshape((1, -1)).to(dtype=torch.float32)
        done = False
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32), eps=1.0).item()
            s_tp1, r_t, done, info = env.step(a_t)
            buffer.add(s_t, a_t, s_tp1, done)
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
        obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch_no_r(*batch)
        r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
        if alg.train_steps % kwargs['interval'] == 0:
            visualise.eval_wm_warmup(i, **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []})


def main(env, visualise, folder_name, **kwargs):
    buffer = DynamicsReplayBuffer(kwargs['buffer_size'])
    # buffer = ReplayBuffer(kwargs['buffer_size'])
    # obs_dim = INPUT_DIM
    obs_dim = env.observation_space.sample().shape
    assert len(obs_dim) == 1, 'States should be 1D vector.'
    a_dim = (env.action_space.n,)
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    alg = DQN(obs_dim, a_dim, device=device, **kwargs)
    if kwargs['encoder_type'] == 'none':
        wm = WorldModelNoEncoder(obs_dim, a_dim, device=device, **kwargs)
    else:
        wm = EncodedWorldModel(obs_dim, a_dim, device=device, **args)
    ep_scores = {'DQN': [0.0], 'Mean intrinsic reward': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [0.0], 'int': {'mean': [], 'std': [], 'min': [], 'max': []}, 'len': []}
    fill_buffer(env, alg, buffer, **kwargs)
    if kwargs['wm_warmup_steps'] > 0:
        warmup_wm(alg, wm, buffer, visualise, **kwargs)
    s_tp1, a_t = None, None
    while alg.train_steps < kwargs['train_steps']:
        s_t = env.reset()
        # s_t = standardize_state(s_t, grayscale=True)
        s_t = s_t / 256.0
        # env.render('human')
        done = False
        # if s_tp1 is not None:
        #     buffer.add(s_tp1, a_t, s_t, done)
        total = [0, 0]
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32)).item()
            s_tp1, r_t, done, info = env.step(a_t)
            # s_tp1 = standardize_state(s_tp1, grayscale=True)
            s_tp1 = s_tp1 / 256.0
            total[0] += r_t
            total[1] += 1
            buffer.add(s_t, a_t, s_tp1, done)
            s_t = s_tp1
            # env.render('human')
            if alg.train_steps < kwargs['train_steps']:
                batch = buffer.sample(kwargs['batch_size'])
                obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch_no_r(*batch)
                r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
                total_history['int']['mean'].append(r_int_t.mean().item())
                # total_history['int']['std'].append(r_int_t.std().item())
                # r_mean = np.mean(total_history['int']['mean'][-100:])
                # r_std = np.mean(total_history['int']['std'][-100:])+1e-8 if len(total_history['int']['std'])>1 else 1.0
                # r_mean = total_history['int']['mean'][-1]
                # r_std = r_int_t.std()
                r_min, r_max = r_int_t.min(), r_int_t.max()
                # total_history['int']['min'].append(r_min)
                # total_history['int']['max'].append(r_max)
                # r_min = np.min(total_history['int']['min'][-1000:])
                # r_max = np.max(total_history['int']['max'][-1000:])
                r_range = r_max - r_min + 1e-10
                alg.train(obs_t_batch, a_t_batch, (r_int_t - r_min) / r_range, obs_tp1_batch, dones_batch)

                if done:
                    total_history['ext'].append(total[0])
                    total_history['len'].append(total[1])
                if alg.train_steps % kwargs['eval_interval'] == 0:
                    ep_scores['DQN'].append(np.mean(total_history['ext'][-10:]))
                    ep_scores['Mean intrinsic reward'].append(np.mean(total_history['int']['mean'][-1000:]))
                    elapsed_time = (int((datetime.now() - start_time).total_seconds() // (60 * 60)),
                                    int((datetime.now() - start_time).total_seconds() % (60 * 60) // 60),
                                    int((datetime.now() - start_time).total_seconds() % 60))
                    print('--------------------------------------\n',
                          'Step:', alg.train_steps, '/', kwargs['train_steps'], '     ',
                          'E(G):', ep_scores['DQN'][-1],
                          'Eps:', alg.epsilon,
                          'Time elapsed:', f'{elapsed_time[0]}:{elapsed_time[1]}:{elapsed_time[2]}')

                    os.makedirs(folder_name + 'objects/', exist_ok=True)
                    wm.save(path=f'{folder_name}objects/WM.pt')
                    alg.save(path=f'{folder_name}objects/DQN.pt')
                    visualise.train_iteration_update(ext=ep_scores['DQN'][-1],
                                                     int=ep_scores['Mean intrinsic reward'][-1],
                                                     **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []},
                                                     alg_loss=np.mean(alg.losses[-100:]))
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
