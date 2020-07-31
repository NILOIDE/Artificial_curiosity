import torch
import numpy as np
np.set_printoptions(linewidth=400)
import gym
import grid_gym
from grid_gym.envs.grid_world import *
from modules.world_models.world_model import EncodedWorldModel, WorldModelNoEncoder, TabularWorldModel, WorldModelContrastive
import os
import matplotlib.pyplot as plt
plt.ioff()


def draw_heat_map(array, path, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(array, cmap='jet', vmin=0.0)
    ax.set_title(f'Heat map of {name}')
    fig.colorbar(ax=ax, mappable=im, orientation='vertical')
    plt.savefig(path)
    plt.close()


def size_from_env_name(env_name: str) -> tuple:
    i = 0
    for i in range(len(env_name) - 1, -1, -1):
        if env_name[i] == 'x':
            break
    return (int(env_name[i+1-3:-len('-v0')-3]), int(env_name[i+1:-len('-v0')]))


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
    elif env_name == 'GridWorldRandFeatures42x42-v0':
        env = GridWorldRandFeatures42x42()
    elif env_name == 'GridWorldSubspace50x50-v0':
        env = GridWorldSubspace50x50()
    else:
        raise ValueError('Wrong env_name.')
    return env


def eval_wm(wm, q_values, folder_name, env_name, separate_enc=None, save_name=None):
    print('Evaluating...')
    size = size_from_env_name(env_name)
    env = get_env_instance(env_name)
    a_dim = env.action_space.n
    prediction_error = np.zeros(size)
    # Map where Max Q-values will be plotted into
    q_grid = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if hasattr(env, 'map') and env.map[i, j] == 1:
                continue
            s = torch.zeros(size)
            s[i, j] = 1
            for a in range(a_dim):
                env.pos = [i, j]
                ns = env.step(a)[0]
                # Map Q-values
                q = q_values.forward(s.reshape((-1,)).numpy(), a)
                if q > q_grid[i, j]:
                    q_grid[i, j] = q
                # Map pred error
                if isinstance(wm, EncodedWorldModel) or isinstance(wm, WorldModelContrastive):
                    z_tp1_p = wm.next(s.reshape((-1,)), torch.tensor([a]))
                    z_tp1 = wm.encode(torch.from_numpy(ns).type(torch.float))
                    prediction_error[i, j] += (z_tp1_p - z_tp1).abs().sum().item() / 2 / a_dim
                elif separate_enc is not None:
                    z_t = separate_enc.encode(s.reshape((-1,)))
                    z_tp1_p = wm.next(z_t, torch.tensor([a]))
                    z_tp1 = separate_enc.encode(torch.from_numpy(ns).type(torch.float))
                    prediction_error[i, j] += (z_tp1_p - z_tp1).abs().sum().item() / 2 / a_dim
                else:
                    pns = wm.next(s.reshape((-1,)), torch.tensor([a])).numpy()
                    prediction_error[i, j] += np.sum(np.abs(ns - pns))/2/a_dim
    if save_name is not None:
        os.makedirs(folder_name + "/heat_maps/", exist_ok=True)
        draw_heat_map(prediction_error, folder_name + "/heat_maps/pred_error.png", 'prediction error')
        draw_heat_map(q_grid, folder_name + "/heat_maps/q_values.png", 'Q-values')
    walls_map = None
    if hasattr(env, 'map'):
        walls_map = env.map
    return prediction_error, q_grid, walls_map


if __name__ == "__main__":
    # PATH = 'r/2020-03-24_19-41-40-546547_-_GridWorld25x25-v0eps0_1_target100_lr001_Adam_buff2k/'
    PATH = 'r/2020-04-02_00-32-10-160532_-_GridWorldBox11x11-v0-eps01_Adam_alglr001_wmlr00001_soft_buf10k/'

    env_name = 'GridWorldBox11x11-v0'
    env = gym.make(env_name)
    obs_dim = (len(env.observation_space.sample()),)
    a_dim = (env.action_space.n,)
    wm = WorldModelNoEncoder(obs_dim, a_dim, **{'wm_optimizer': torch.optim.Adam, 'wm_lr': 0.001})
    wm.load(PATH + 'objects/WM.pt')
    eval_wm(wm, PATH, env_name, PATH)
