import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import cv2

name_map = {'eps1_tab': 'Random policy ($\epsilon = 1.0$)',
            'count': 'Count-based',
            'tab': 'Tabular PE',
            'none': 'NN PE (No encoder)',
            'random': 'Random encoder',
            'uniform_cont': 'Contrastive (uniform sample)',
            'uniform_x10_cont': 'Contrastive (uniform sample), 10:1 updates',
            'visited_cont': 'Contrastive (visited sample)',
            'visited_x10_cont': 'Contrastive (visited sample), 10:1 updates',
            'pretrain_cont': 'Contrastive Pretrained',
            'eps1': 'Random policy ($\epsilon = 1.0$)',
            'cont': 'Contrastive encoder',
            'vae': 'Variational Autoencoder',
            'idf': 'Inverse Dynamics encoder (ICM)'}


def gather_data_tensorboard(directory):
    folder = directory + '2020-08-07_15-26-15-186475_-__count_none_4/events.out.tfevents.1596806775.LAPTOP-NST0I3T8'
    print(folder)
    event_acc = EventAccumulator(folder, size_guidance={'images':100}, compression_bps='images')
    event_acc.Reload()
    value_name = 'Evaluation/Argmax_Q-value_map'
    print(event_acc.Tags())
    # if folder[-4:] == 'eps1':
    #     value_name = 'Training/Mean_ep_extrinsic_rewards'
    # # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    if not value_name in event_acc.Tags()['images']:
        print('nope')
        quit()
    img = event_acc.Images(value_name)
    a = img[2]
    lis = []
    print(len(img))
    def decode(p):
        print(p)
        tf_img = np.frombuffer(img[p-1].encoded_image_string, dtype=np.uint8)
        tf_img = cv2.imdecode(tf_img, cv2.IMREAD_COLOR)  # [H, W, C]
        lis.append(tf_img)
        fig = plt.figure(p)
        ax = plt.subplot(111)

        ax.imshow(tf_img)

    decode(100000//20000)
    decode(250000//20000)
    decode(500000//20000)
    decode(1000000//20000)
    decode(2000000//20000)
    decode(3000000//20000)


def plot(save_path, data):
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, y in enumerate(data):
        x = np.arange(0, len(y['mean'])) *500*4
        x = x.tolist()
        # if i == len(data) -2:
        #     ax.plot(x, y['mean'], 'c', label=y['name'])
        # elif i == len(data) - 1:
        #     ax.plot(x, y['mean'], 'y', label=y['name'])
        # else:
        ax.plot(x, y['mean'], label=y['name'])

        ax.fill_between(x, y['mean'] - y['std'], y['mean'] + y['std'], alpha=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Training steps')
    # ax.set_ylabel('Unique visited states')
    # ax.set_ylabel('Mean extrinsic return')
    ax.set_ylabel('Visitation probability difference')
    # ax.set_title('Breakout-v0\nMean extrinsic return during evaluation ($\epsilon$ = 0.0)')
    # ax.set_title('Number of unique visited states in recent history (last ~35K steps)')
    # ax.set_title('Difference between policy and uniform visitation probability\nin recent history (last ~35K steps) across all states')
    ax.set_title('Difference between policy and uniform visitation probability\nin recent history (last ~35K steps) across visited states')
    # print(data[2]['name'])
    # ax.set_xlim(-100000, int(6e6)+100000)

if __name__ == "__main__":
    path = 'final_results/GridWorldRandFeatures42x42-v0/'
    data_list = gather_data_tensorboard(path)
