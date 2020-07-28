import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


name_map = {'eps1_tab': 'Epsilon 1.0',
            'count': 'Count-based',
            'tab': 'Tabular PE',
            'none': 'No encoder',
            'random': 'Random encoder',
            'uniform_cont': 'Contrastive (uniform sample)',
            'visited_cont': 'Contrastive (visited sample)',
            'pretrain_cont': 'Contrastive Pretrained',
            'cont': 'Contrastive encoder',
            'vae': 'Variational Autoencoder',
            'idf': 'Inverse Dynamics encoder (ICM)'}


def gather_data_tensorboard(directory):
    data = []
    for i in os.listdir(directory):
        for j in os.listdir(os.path.join(directory, i)):
            if os.path.isfile(os.path.join(directory, i + '/' +  j)) and j[:len('events')] == 'events':
                folder = os.path.join(directory, i)
                print(folder)
                event_acc = EventAccumulator(folder)
                event_acc.Reload()
                value_name = 'Training/Unique_states_visited'
                # value_name = 'Training/Policy-Uniform_difference'
                # value_name = 'Training/Policy-Uniform_difference_visited_states'
                # value_name = 'Evaluation/Mean_ep_extrinsic_rewards'
                # # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
                if not value_name in event_acc.Tags()['scalars']:
                    print('nope')
                    continue
                w_times, step_nums, vals = zip(*event_acc.Scalars(value_name))

                name = ''
                # Remove time stamp (everything before the - )
                for c in reversed(i):
                    if c == '-':
                        break
                    name += c
                name = name[::-1]  # Undo reverse
                name = name[1:]  # Remove initial underscore

                d = {'name': name,
                     'list': np.array(list(vals)[::20])}
                print(len(d['list']))
                data.append(d)

    data_clean = []

    def take_average(name, lists):
        if not lists:
            return None
        if name == 'Epsilon 1.0':
            d = {'name': name,
                 'mean': np.ones((int(3e6 / 500/20))) * np.mean(np.array(lists)),
                 'std': np.zeros(np.array(lists).shape[0])
                 }
        else:
            max_len = 0
            for i in lists:
                if len(i) > max_len:
                    max_len = len(i)
            mean = []
            std = []
            for i in range(max_len):
                elements = []
                for l in lists:
                    if i >= len(l):
                        continue
                    elements.append(l[i])
                mean.append(np.mean(elements))
                std.append(np.std(elements))

            d = {'name': name,
                 'mean': np.array(mean),
                 'std': np.array(std)
                 }
            assert len(d['mean']) == max_len
            assert len(d['std']) == max_len
        return d
    print(len(data))

    l = [d['list'] for d in data if d['name'] == 'eps1_tab']
    if l:
        data_clean.append(take_average(name_map['eps1_tab'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'count']
    if l:
        data_clean.append(take_average(name_map['count'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'tab']
    if l:
        data_clean.append(take_average(name_map['tab'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'none']
    if l:
        data_clean.append(take_average(name_map['none'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'random']
    if l:
        data_clean.append(take_average(name_map['random'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'pretrain_cont']
    if l:
        data_clean.append(take_average(name_map['pretrain_cont'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'uniform_cont']
    if l:
        data_clean.append(take_average(name_map['uniform_cont'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'visited_cont']
    if l:
        data_clean.append(take_average(name_map['visited_cont'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'vae']
    if l:
        data_clean.append(take_average(name_map['vae'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'idf']
    if l:
        data_clean.append(take_average(name_map['idf'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'cont']
    if l:
        data_clean.append(take_average(name_map['cont'], l))

    print(len(data_clean))
    return data_clean


def plot(save_path, data):
    fig = plt.figure()
    ax = plt.subplot(111)
    for y in data:
        x = np.arange(0, len(y['mean'])) * 500*20
        x = x.tolist()
        ax.plot(x, y['mean'], label=y['name'])
        ax.fill_between(x, y['mean'] - y['std'], y['mean'] + y['std'], alpha=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Number of visited states')
    # ax.set_ylabel('Visitation probability difference')
    # ax.set_title('Breakout-v0\nMean extrensic return during evaluation ($\epsilon$ = 0.0)')
    ax.set_title('Unique visited states in recent history (last 35280 steps)')
    # ax.set_title('Difference between policy and uniform visitation probability\nin recent history (last 35280 steps) across all states')
    # ax.set_title('Difference between policy and uniform visitation probability\nin recent history (last 35280 steps) across visited states')


if __name__ == "__main__":
    path = 'final_results/GridWorldSpiral28x28-v0/'
    data_list = gather_data_tensorboard(path)
    plot(path, data_list)
