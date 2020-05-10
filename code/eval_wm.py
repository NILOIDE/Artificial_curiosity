import torch
import numpy as np
np.set_printoptions(linewidth=400)
import gym
import grid_gym
from grid_gym.envs.grid_world import *
from modules.world_models.world_model import EncodedWorldModel, WorldModelNoEncoder
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


def size_from_env_name(env_name: str) -> int:
    i = 0
    for i in range(len(env_name) - 1, -1, -1):
        if env_name[i] == 'x':
            break
    return int(env_name[i+1:-len('-v0')])


def get_env_instance(env_name):
    if env_name == 'GridWorldBox11x11-v0':
        env = GridWorldBox11x11()
    elif env_name == 'GridWorldSpiral28x28-v0':
        env = GridWorldSpiral28x28()
    elif env_name == 'GridWorld10x10-v0':
        env = GridWorld10x10()
    elif env_name == 'GridWorld25x25-v0':
        env = GridWorld25x25()
    elif env_name == 'GridWorld40x40-v0':
        env = GridWorld40x40()
    else:
        raise ValueError('Wrong env_name.')
    return env


def eval_wm(wm, folder_name, env_name, save_name=None):
    size = size_from_env_name(env_name)
    env = get_env_instance(env_name)
    a_dim = env.action_space.n
    prediction_error = np.zeros((size, size))
    argmax_prediction_error = None
    if not isinstance(wm, EncodedWorldModel):
        argmax_prediction_error = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # if env.map[i, j] == 1:
            #     continue
            s = torch.zeros((size, size))
            s[i, j] = 1
            for a in range(a_dim):
                env.pos = [i, j]
                ns = env.step(a)[0]#.reshape((size, size))
                if isinstance(wm, EncodedWorldModel):
                    z_tp1_p = wm.next(s.reshape((1,-1)), torch.tensor([a]))
                    z_tp1 = wm.encode(torch.from_numpy(ns).type(torch.float))
                    prediction_error[i, j] += (z_tp1_p - z_tp1).abs().sum().item() / 2 / a_dim
                else:
                    pns = wm.next(s.reshape((1, -1,)), torch.tensor([a])).numpy()#.reshape((size, size))
                    argmax_prediction_error[i, j] += np.sum(np.abs(ns - (pns == pns.max()))) / 2 / a_dim
                    prediction_error[i, j] += np.sum(np.abs(ns - pns))/2/a_dim
    if save_name is not None:
        os.makedirs(folder_name + "/heat_maps/", exist_ok=True)
        draw_heat_map(prediction_error, folder_name + "/heat_maps/pred_error.png", 'prediction error')
        draw_heat_map(argmax_prediction_error, folder_name + "/heat_maps/argmax_pred_error.png", 'argmax prediction error')
    walls_map = None
    if hasattr(env, 'map'):
        walls_map = env.map
    return prediction_error, argmax_prediction_error, walls_map


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
