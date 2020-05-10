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
from eval_wm import eval_wm
np.set_printoptions(linewidth=400)


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


def warmup_wm(env, alg, wm, buffer, total_history, visualise, **kwargs):
    """"
    Used to warmup the environment visitation history.
    """
    print('Warming up world model...')
    while len(buffer) < kwargs['buffer_size']:
        s_t = env.reset()
        # s_t = torch.from_numpy(s_t).reshape((1, -1)).to(dtype=torch.float32)
        done = False
        while not done:
            a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32), eps=1.0).item()
            s_tp1, r_t, done, info = env.step(a_t)
            buffer.add(s_t, a_t, s_tp1, done)
            s_t = s_tp1
    for i in range(-kwargs['wm_warmup_steps'] + 1, 0 + 1):
        batch = buffer.sample(kwargs['batch_size'])
        obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch_no_r(*batch)
        r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
        total_history['int'].append(r_int_t.mean().item())
        if alg.train_steps % kwargs['interval'] == 0:
            visualise.eval_wm_warmup(i, **{k: np.mean(i[-100:]) for k, i in wm.losses.items() if i != []})


def main(env, visualise, folder_name, **kwargs):
    buffer = DynamicsReplayBuffer(kwargs['buffer_size'])
    # buffer = ReplayBuffer(kwargs['buffer_size'])
    obs_dim = (len(env.observation_space.sample()),)
    a_dim = (env.action_space.n,)
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    alg = DQN(obs_dim, a_dim, device=device, **kwargs)
    if kwargs['encoder_type'] == 'none':
        wm = WorldModelNoEncoder(obs_dim, a_dim, device=device, **kwargs)
    else:
        wm = EncodedWorldModel(obs_dim, a_dim, device=device, **args)
    ep_scores = {'DQN': [0.0], 'Mean intrinsic reward': [0.0]}
    start_time = datetime.now()
    total_history = {'ext': [], 'int': []}
    warmup_wm(env, alg, wm, buffer, total_history, visualise, **kwargs)
    while alg.train_steps < kwargs['train_steps']:
        s_t = env.reset()
        done = False
        while not done:
            for i in range(1):
                a_t = alg.act(torch.from_numpy(s_t).to(dtype=torch.float32)).item()
                s_tp1, r_t, done, info = env.step(a_t)
                buffer.add(s_t, a_t, s_tp1, done)
                s_t = s_tp1
            if alg.train_steps < kwargs['train_steps']:
                batch = buffer.sample(kwargs['batch_size'])
                obs_t_batch, a_t_batch, obs_tp1_batch, dones_batch = transition_to_torch_no_r(*batch)
                r_int_t = wm.train(obs_t_batch, a_t_batch, obs_tp1_batch, **{'memories': buffer})
                r_mean = np.mean(total_history['int'][-1000:]) if len(total_history['int'])>10 else 0.0
                r_std = np.std(total_history['int'][-1000:])+1e-8 if len(total_history['int'])>10 else 1.0
                alg.train(obs_t_batch, a_t_batch, (r_int_t - r_mean) / r_std, obs_tp1_batch, dones_batch)
                total_history['ext'].append(r_t)
                total_history['int'].append(r_int_t.mean().item())
                if alg.train_steps % kwargs['interval'] == 0:
                    ep_scores['DQN'].append(np.mean(total_history['ext'][-500:]))
                    ep_scores['Mean intrinsic reward'].append(np.mean(total_history['int'][-500:]))
                    # total_history = {'ext': [], 'int': []}
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
                    if env_name[:9] == 'GridWorld':
                        draw_heat_map(info['density'], alg.train_steps, folder_name)
                        pe_map, ape_map, walls_map = eval_wm(wm, folder_name, env_name)
                        visualise.eval_gridworld_iteration_update(density_map=info['density'],
                                                                  pe_map=pe_map,
                                                                  ape_map=ape_map,
                                                                  walls_map=walls_map)
    env.close()
    print('Environment closed.')
    visualise.close()
    print('Tensorboard writer closed.')


if __name__ == "__main__":
    env_name = 'GridWorldSpiral28x28-v0'
    args = {'save_dir': 'results_test/',
            'env_name': env_name,
            'name': 'alglr001_wmlr00001_batch256_noDetach_tnetAlg100',
            'time_stamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
            'seed': 1,
            'interval': 500,
            'buffer_size': int(5e3),
            'train_steps': int(10e5),
            'gamma': 0.99,
            'eps_static': True,
            'eps_half': 0.08,
            'eps_min': 0.1,
            'batch_size': 256,
            'alg_target_net_steps': 100,
            'alg_soft_target': False,
            'alg_optimizer': torch.optim.Adam,
            'alg_lr': 0.01,
            'z_dim': (32,),
            'wm_target_net_steps': 0,
            'wm_soft_target': False,
            'wm_tau': 0.01,
            'wm_optimizer': torch.optim.Adam,
            'wm_lr': 0.0001,
            'wm_warmup_steps': 0,
            'encoder_type': 'cont',
            'conv_layers': CONV_LAYERS2014,
            'stochastic_latent': False,
            'encoder_batchnorm': False,
            'neg_samples': 3,
            'hinge_value': 1.0}
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    run_name = f"{args['save_dir']}{args['env_name']}/{args['time_stamp']}_-_{args['name']}_{args['encoder_type']}/"
    print(run_name)
    environment = gym.make(env_name)
    visualise = Visualise(run_name, **args)
    try:
        main(environment, visualise, run_name, **args)
    finally:
        environment.close()
        visualise.close()
