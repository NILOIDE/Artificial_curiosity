import numpy as np

name = 'eps1.out'
with open(name) as f:
    lines = f.readlines()
    stds = []
    for line in lines:
        if line[:len('Evaluation:  Returns: ')] == 'Evaluation:  Returns: ':
            string = line[len('Evaluation:  Returns: ['):-2]
            a = [float(s) for s in string.split(',')]
            std = np.std(a)
            stds.append(std)

    print(stds)
