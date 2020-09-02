import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
    data = []
    for i in os.listdir(directory):
        for j in os.listdir(os.path.join(directory, i)):
            if os.path.isfile(os.path.join(directory, i + '/' + j)) and j[:len('events')] == 'events':
                folder = os.path.join(directory, i)
                print(folder)
                event_acc = EventAccumulator(folder)
                event_acc.Reload()
                value_name = 'Training/Unique_states_visited'
                # value_name = 'Training/Policy-Uniform_difference'
                # value_name = 'Training/Policy-Uniform_difference_visited_states'
                # value_name = 'Evaluation/Mean_ep_extrinsic_rewards'
                # if folder[-4:] == 'eps1':
                #     value_name = 'Training/Mean_ep_extrinsic_rewards'
                # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
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
                     'list': np.array(list(vals)[:])}
                print(len(d['list']))
                data.append(d)

    data_clean = []

    def take_average(name, lists):
        if not lists:
            return None
        if name == 'Random policy ($\epsilon = 1.0$)':
            d = {'name': name,
                 'mean': np.ones(1500) * np.mean(np.array(lists)),
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
                for j in lists:
                    if i >= len(j):
                            continue
                    elements.append(j[i])
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
    l = [d['list'][:307] for d in data if d['name'] == 'eps1']
    if l:
        data_clean.append(take_average(name_map['eps1'], l))
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
    l = [d['list'] for d in data if d['name'][:-2] == 'uniform_x10_cont']
    if l:
        data_clean.append(take_average(name_map['uniform_x10_cont'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'visited_x10_cont']
    if l:
        data_clean.append(take_average(name_map['visited_x10_cont'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'vae']
    if l:
        data_clean.append(take_average(name_map['vae'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'idf']
    if l:
        data_clean.append(take_average(name_map['idf'], l))
    l = [d['list'] for d in data if d['name'][:-2] == 'cont']
    if l:
        data_clean.append(take_average(name_map['cont'], l))
    return data_clean


def plot(save_path, data):
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, y in enumerate(data):
        x = np.arange(0, len(y['mean'])) *500*4
        x = x.tolist()
        cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if y['name'] == 'Tabular PE':
            color = cmap[2]
        elif y['name'] == 'NN PE (No encoder)':
            color = cmap[3]
        elif y['name'] == 'Random encoder':
            color = cmap[4]
        elif y['name'] == 'Contrastive Pretrained':
            color = cmap[5]
        elif y['name'] == 'Contrastive (uniform sample)':
            color = cmap[6]
        elif y['name'] == 'Contrastive (visited sample)':
            color = cmap[7]
        elif y['name'] == 'Contrastive (uniform sample), 10:1 updates':
            color = cmap[0]
        elif y['name'] == 'Contrastive (visited sample), 10:1 updates':
            color = cmap[1]
        else:
            color = cmap[i]
        ax.plot(x, y['mean'], color=color, label=y['name'])
        ax.fill_between(x, y['mean'] - y['std'], y['mean'] + y['std'], color=color, alpha=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Training steps')
    # ax.set_ylabel('Unique visited states')
    # ax.set_ylabel('Mean extrinsic return')
    ax.set_ylabel('Visitation probability difference')
    # ax.set_title('Breakout-v0\nMean extrinsic return during evaluation ($\epsilon$ = 0.0)')
    ax.set_title('Number of unique visited states in recent history (last ~35K steps)')
    # ax.set_title('Difference between policy and uniform visitation probability\nin recent history (last ~35K steps) across all states')
    # ax.set_title('Difference between policy and uniform visitation probability\nin recent history (last ~35K steps) across visited states')
    # print(data[2]['name'])
    # ax.set_xlim(-100000, int(6e6)+100000)

if __name__ == "__main__":
    path = 'final_results/GridWorldRandFeatures42x42-v0_appendix/'
    data_list = gather_data_tensorboard(path)
    plot(path, data_list)
