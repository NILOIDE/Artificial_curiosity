import gym
import numpy as np
from gym import spaces
import copy
import os
np.set_printoptions(linewidth=150)


class SimpleGridWorld(gym.Env):
    metadata = {'render.modes': ['human', 'agent']}

    def __init__(self, size: tuple, **kwargs):
        """"
        Actions are given as indices. 0 is stay still, indices 1 and 2
        are -1 and +1 in dim 0 (left and right), indices 2 and 3 are
        -1 and +1 in dim 1 (up and down).
        """
        self.n_dims = len(size)
        if self.n_dims < 1:
            raise ValueError('Number of dimensions should be larger than zero.')
        self.size = size
        self.observation_space = spaces.Tuple((spaces.Discrete(2) for _ in range(int(np.prod(size)))))
        # self.observation_space = spaces.Box(0, 1, [*size, 1])
        self.action_space = spaces.Discrete(self.n_dims * 2 + 1)
        self.pos = None
        self.start_pos = [self.size[i] // 2 for i in range(self.n_dims)]
        self.last_state = None
        self.visitation_count = np.zeros(self.size, dtype=np.int)
        self.t = 0
        self.history_len = np.prod(size) * 20
        self.visitation_history = []

    def step(self, a):
        self.t += 1
        info = {'steps': self.t,
                'counts': self.visitation_count,
                'density': self.visitation_count/len(self.visitation_history)}
        dim = (a - 1) // 2
        assert 0 <= a < self.n_dims*2 + 1, 'Invalid action: 0 <= a < '+str(self.n_dims*2 + 1)
        direction = (a - 1) % 2
        if a != 0:
            if direction == 0:
                if self.pos[dim] > 0:
                    self.pos[dim] -= 1
            else:
                if self.pos[dim] < self.size[dim] - 1:
                    self.pos[dim] += 1
        s = self._index_to_grid(self.pos).reshape((-1,))
        self.last_state = s
        self.update_visitation_counts()
        return s, self._reward(), False, info

    def update_visitation_counts(self):
        self.visitation_count[tuple(self.pos)] += 1
        self.visitation_history.append(copy.copy(self.pos))
        if len(self.visitation_history) > self.history_len:
            pos = self.visitation_history.pop(0)
            self.visitation_count[tuple(pos)] -= 1
            assert self.visitation_count[tuple(pos)] >= 0, self.visitation_count

    def _reward(self):
        return np.abs(np.array(self.pos) - np.array(self.start_pos)).sum()

    def reset(self):
        self.t = 0
        self.pos = copy.copy(self.start_pos)
        s = self._index_to_grid(self.pos)
        self.last_state = s
        self.update_visitation_counts()
        return s.reshape((-1,))

    def _index_to_grid(self, pos):
        s = np.zeros(self.size)
        s[tuple(pos)] = 1
        return s

    def render(self, mode='human'):
        if mode == 'human':
            print(self.pos)
        elif mode == 'agent':
            print(self.last_state)


class GridWorldLoad(SimpleGridWorld):
    def __init__(self, map_path: str = 'maps/Box.txt', **kwargs):
        self.map = self.load_map(map_path)
        super().__init__(tuple(self.map.shape))

    def step(self, a):
        self.t += 1
        info = {'steps': self.t,
                'counts': self.visitation_count,
                'density': self.visitation_count / len(self.visitation_history) if len(self.visitation_history) else None}
        dim = (a - 1) // 2
        assert 0 <= a < self.n_dims * 2 + 1, f'Invalid action: 0 <= a < {self.n_dims * 2 + 1}'
        direction = (a - 1) % 2
        if a != 0:
            if not self.check_collision(dim, direction):
                if direction == 0:
                    self.pos[dim] -= 1
                else:
                    self.pos[dim] += 1
        s = self._index_to_grid(self.pos).reshape((-1,))
        self.last_state = s
        self.update_visitation_counts()
        return s, self._reward(), False, info

    def check_collision(self, dim: int, direction: int) -> bool:
        """"
        Given a position, check whether there is a collision one step away along the
        given dimension in the given direction.
        """
        pos = copy.copy(self.pos)
        if direction == 0:
            if pos[dim] == 0:
                return True
            pos[dim] -= 1
        elif direction == 1:
            if pos[dim] == self.size[dim] - 1:
                return True
            pos[dim] += 1
        return self.map[tuple(pos)] == 1  # Return True if there is a wall in intended move position

    @staticmethod
    def load_map(map_path: str) -> np.ndarray:
        path = f'{os.path.dirname(os.path.realpath(__file__))}/{map_path}'
        with open(path, 'r') as f:
            lines = f.readlines()
            assert lines[-1][-1] == '\n', 'Map file must end with empty new line.'
            width = len(lines[0]) - 1  # New line character (\n) doesnt count
            height = len(lines)
            assert lines[width//2][height//2] != '#', "There shouldn't be a wall at the starting position."
            map = np.zeros((width, height))
            for y, line in enumerate(lines):
                assert len(line) - 1 == width, f'Map width is inconsistent at row {y}'
                for x, c in enumerate(line):
                    if c == '.':
                        pass
                    elif c == '#':
                        map[y, x] = 1  # Yes, indexing is correct.
                    elif c == '\n':
                        assert x + 1 == len(line), f'Row contains characters after new-line character.'
                        break
                    else:
                        raise ValueError(f'Invalid character {c} at: row {x}, column {y}.')
        return map


class GridWorld10x10(SimpleGridWorld):
    def __init__(self):
        super(GridWorld10x10, self).__init__(size=(10, 10))


class GridWorld25x25(SimpleGridWorld):
    def __init__(self):
        super(GridWorld25x25, self).__init__(size=(25, 25))


class GridWorld40x40(SimpleGridWorld):
    def __init__(self):
        super(GridWorld40x40, self).__init__(size=(40, 40))


class GridWorldBox11x11(GridWorldLoad):
    def __init__(self):
        super(GridWorldBox11x11, self).__init__(map_path='maps/Box.txt')


class GridWorldSpiral28x28(GridWorldLoad):
    def __init__(self):
        super(GridWorldSpiral28x28, self).__init__(map_path='maps/Spiral.txt')


if __name__ == "__main__":
    env = GridWorldBox11x11()
