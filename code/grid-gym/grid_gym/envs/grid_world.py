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
        self.observation_space = spaces.MultiBinary(n=int(np.prod(size)))
        self.action_space = spaces.Discrete(self.n_dims * 2 + 1)
        self.pos = None
        self.start_pos = [self.size[i] // 2 for i in range(self.n_dims)]
        self.last_state = None
        self.visitation_count = np.zeros(self.size, dtype=np.int)
        self.visited_states = np.zeros(self.size, dtype=np.int)
        self.t = 0
        self.history_len = np.prod(size) * 20
        self.visitation_history = []
        self.num_accesible_states = np.prod(self.size)
        self.uniform_prob_map = np.ones(self.size) / self.num_accesible_states

    def _create_info_dict(self):
        info = {'steps': self.t,
                'distance': abs(self.pos[0] - self.start_pos[0]) + abs(self.pos[1] - self.start_pos[1]),
                'counts': self.visitation_count,
                'density': self.visitation_count / len(self.visitation_history)
                if len(self.visitation_history) > 0 else self.visitation_count}
        # Count of unique states visited
        self.visited_states = (self.visitation_count > 0)
        visited_sum = self.visited_states.sum()
        # info['visited_states'] = np.array(self.get_unique_visited_states())
        info['unique_states'] = visited_sum if visited_sum > 0.0 else 1.0
        info['uniform_diff'] = np.abs(self.uniform_prob_map - info['density']).sum()
        info['uniform_diff_visited'] = (np.abs(self.visited_states / info['unique_states']
                                               - info['density']) * self.visited_states).sum()
        return info

    def step(self, a: int):
        self.t += 1
        info = self._create_info_dict()
        dim = (a - 1) // 2
        assert 0 <= a < self.n_dims * 2 + 1, 'Invalid action: 0 <= a < ' + str(self.n_dims * 2 + 1)
        direction = (a - 1) % 2
        if a != 0:
            if direction == 0:
                self.pos[dim] = (self.pos[dim] - 1) % self.size[dim]
            else:
                self.pos[dim] = (self.pos[dim] + 1) % self.size[dim]
        s = self._index_to_grid(self.pos).reshape((-1,))
        self.last_state = s
        self.update_visitation_counts()
        return s, info['uniform_diff'], False, info

    def update_visitation_counts(self):
        self.visitation_count[tuple(self.pos)] += 1
        self.visitation_history.append(copy.copy(self.pos))
        if len(self.visitation_history) > self.history_len:
            pos = self.visitation_history.pop(0)
            self.visitation_count[tuple(pos)] -= 1
            assert self.visitation_count[tuple(pos)] >= 0, self.visitation_count

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

    def get_states(self):
        states = []
        size = np.prod(self.size)
        for i in range(int(size)):
            s = np.zeros((size,))
            s[i] = 1
            states.append(s)
        return states

    def get_states_with_neighbours(self):
        states = []
        neighbours = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                neigh = []
                # If self is neighbour (due to wall), that neighbour is omitted
                # if i > 0:
                neigh.append(self._index_to_grid([(i - 1)%self.size[0], j]).reshape((-1,)))
                # if i < self.size[0] - 1:
                neigh.append(self._index_to_grid([(i + 1)%self.size[0], j]).reshape((-1,)))
                # if j > 0:
                neigh.append(self._index_to_grid([i, (j - 1)%self.size[1]]).reshape((-1,)))
                # if j < self.size[1] - 1:
                neigh.append(self._index_to_grid([i, (j + 1)%self.size[1]]).reshape((-1,)))
                neighbours.append(np.array(neigh))
                s = self._index_to_grid([i, j]).reshape((-1,))
                states.append(s)
        return states, neighbours

    def get_unique_visited_states(self):
        states = []
        size = np.prod(self.size)
        visited = self.visited_states.reshape((-1,))
        for i in range(int(size)):
            if visited[i] == 0:
                continue
            s = np.zeros((size,))
            s[i] = 1
            states.append(s)
        return np.array(states)

    def get_visited_states(self):
        states = []
        size = np.prod(self.size)
        count = self.visitation_count.reshape((-1,))
        visited = self.visited_states.reshape((-1,))
        for i in range(int(size)):
            if visited[i] == 1:
                continue
            for _ in range(int(count[i])):
                s = np.zeros((size,))
                s[i] = 1
                states.append(s)
        return states


class GridWorldRandFeatures(SimpleGridWorld):
    def __init__(self, size: tuple, **kwargs):
        print('Random Feature Gridworld!')
        super().__init__(size=size, **kwargs)
        self.feature_size = np.prod(size)
        self.state_features = np.random.uniform(low=0.0, high=1.0, size=(*size, self.feature_size)).round()
        self.observation_space = spaces.MultiBinary(n=int(self.feature_size))

    def reset(self):
        self.t = 0
        self.pos = copy.copy(self.start_pos)
        s = self.state_features[tuple(self.pos)]
        assert s.shape[0] == self.feature_size, s.shape
        self.last_state = s
        self.update_visitation_counts()
        return s

    def step(self, a: int):
        self.t += 1
        info = self._create_info_dict()
        dim = (a - 1) // 2
        assert 0 <= a < self.n_dims * 2 + 1, 'Invalid action: 0 <= a < ' + str(self.n_dims * 2 + 1)
        direction = (a - 1) % 2
        if a != 0:
            if direction == 0:
                self.pos[dim] = (self.pos[dim] - 1) % self.size[dim]
            else:
                self.pos[dim] = (self.pos[dim] + 1) % self.size[dim]
        s = self.state_features[tuple(self.pos)]
        self.last_state = s
        self.update_visitation_counts()
        return s, info['uniform_diff'], False, info

    def get_states(self):
        states = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                s = self.state_features[i, j]
                states.append(s)
        return states

    def get_states_with_neighbours(self):
        states = []
        neighbours = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                neigh = []
                # If self is neighbour (due to wall), that neighbour is omitted
                # if i > 0:
                neigh.append(self.state_features[(i - 1)%self.size[0], j])
                # if i < self.size[0] - 1:
                neigh.append(self.state_features[(i + 1)%self.size[0], j])
                # if j > 0:
                neigh.append(self.state_features[i, (j - 1)%self.size[1]])
                # if j < self.size[1] - 1:
                neigh.append(self.state_features[i, (j + 1)%self.size[1]])
                neighbours.append(np.array(neigh))
                assert len(neighbours[-1].shape) == 2
                assert neighbours[-1].shape[0] == 4
                assert neighbours[-1].shape[1] == self.feature_size
                s = self.state_features[i, j]
                assert s.shape[0] == self.feature_size
                assert len(s.shape) == 1
                states.append(s)
        return states, neighbours

    def get_visited_states(self):
        states = []
        count = self.visitation_count.reshape((-1,))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.visited_states[i, j] == 1:
                    continue
                for _ in range(int(count[i])):
                    s = self.state_features[i, j]
                    states.append(s)
        return states


class GridWorldSubspaces(SimpleGridWorld):
    def __init__(self, size: tuple, **kwargs):
        super().__init__(size=size, **kwargs)
        self.subspace_size = (10, 10)
        self.subspace_feature_length = 10
        self.subspace_features = np.random.uniform(low=0.0, high=1.0, size=(*size, self.subspace_feature_length))
        # How many super spaces per grid dimension
        self.superspace_size = tuple(i // j + 1 for i, j in zip(size, self.subspace_size))
        self.observation_space = spaces.MultiBinary(n=int(self.subspace_feature_length + np.prod(self.superspace_size)))

    def reset(self):
        self.t = 0
        self.pos = copy.copy(self.start_pos)
        s = np.concatenate((self._superspace_feature(self.pos), self._subspace_feature(self.pos)))
        self.last_state = s
        self.update_visitation_counts()
        return s.reshape((-1,))

    def _create_info_dict(self):
        info = {'steps': self.t,
                'counts': self.visitation_count,
                'density': self.visitation_count / len(self.visitation_history)
                if len(self.visitation_history) > 0 else self.visitation_count}
        # Count of unique states visited
        self.visited_states = (self.visitation_count > 0)
        visited_sum = self.visited_states.sum()
        info['unique_states'] = visited_sum if visited_sum > 0.0 else 1.0
        # Policy-Uniform difference
        info['uniform_diff'] = np.abs(self.uniform_prob_map - info['density']).sum()
        info['uniform_diff_visited'] = (np.abs(self.visited_states / info['unique_states']
                                               - info['density']) * self.visited_states).sum()
        return info

    def step(self, a):
        self.t += 1
        info = self._create_info_dict()
        dim = (a - 1) // 2
        assert 0 <= a < self.n_dims * 2 + 1, 'Invalid action: 0 <= a < ' + str(self.n_dims * 2 + 1)
        direction = (a - 1) % 2
        if a != 0:
            if direction == 0:
                if self.pos[dim] > 0:
                    self.pos[dim] -= 1
            else:
                if self.pos[dim] < self.size[dim] - 1:
                    self.pos[dim] += 1
        s = np.concatenate((self._superspace_feature(self.pos), self._subspace_feature(self.pos)))
        assert len(s.shape) == 1
        self.last_state = s
        self.update_visitation_counts()
        return s, info['uniform_diff'], False, info

    def _superspace_feature(self, pos):
        superspace = []
        for i in range(len(pos)):
            superspace.append(pos[i] // self.subspace_size[i])
        feature = np.zeros((np.prod(self.superspace_size, )))
        # Only works for 2D gridworld
        idx = superspace[0] + superspace[1] * self.superspace_size[0]
        feature[idx] = 1.0
        return feature

    def _subspace_feature(self, pos):
        feature = self.subspace_features[pos[0], pos[1], :]
        return feature

    def get_states(self):
        states = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                pos = [i, j]
                s = np.concatenate((self._superspace_feature(pos), self._subspace_feature(pos)))
                states.append(s)
        return states


class GridWorldLoad(SimpleGridWorld):
    def __init__(self, map_path: str = 'maps/Box.txt', **kwargs):
        self.map = self.load_map(map_path)
        super().__init__(tuple(self.map.shape))
        # Uniform visitation probability for accessible positions
        self.accesible_states = (self.map == 0)
        self.num_accesible_states = self.accesible_states.sum()
        self.uniform_prob_map = np.ones(self.size) / self.num_accesible_states

    def _create_info_dict(self):
        info = {'steps': self.t,
                'counts': self.visitation_count,
                'density': self.visitation_count / len(self.visitation_history)
                if len(self.visitation_history) > 0 else self.visitation_count}
        # Count of unique states visited
        self.visited_states = (self.visitation_count > 0).astype(np.float)
        visited_sum = self.visited_states.sum()
        info['unique_states'] = visited_sum if visited_sum > 0.0 else 1.0
        # Policy-Uniform difference
        info['uniform_diff'] = np.abs(self.uniform_prob_map - info['density']).sum()
        info['uniform_diff_visited'] = (np.abs(self.visited_states / info['unique_states']
                                               - info['density']) * self.visited_states).sum()
        return info

    def step(self, a):
        self.t += 1
        info = self._create_info_dict()
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
        return s, info['uniform_diff'], False, info

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
            assert lines[width // 2][height // 2] != '#', "There shouldn't be a wall at the starting position."
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

    def get_states(self):
        states = []
        size = np.prod(self.size)
        map1D = self.map.reshape((-1,))
        for i in range(int(size)):
            if map1D[i] == 1:
                continue
            s = np.zeros((size,))
            s[i] = 1
            states.append(s)
        return states

    def get_unique_visited_states(self):
        states = []
        size = np.prod(self.size)
        visited = self.visited_states.reshape((-1,))
        for i in range(int(size)):
            if visited[i] == 1:
                continue
            s = np.zeros((size,))
            s[i] = 1
            states.append(s)
        return states

    def get_visited_states(self):
        states = []
        size = np.prod(self.size)
        count = self.visitation_count.reshape((-1,))
        visited = self.visited_states.reshape((-1,))
        for i in range(int(size)):
            if visited[i] == 1:
                continue
            for _ in range(int(count[i])):
                s = np.zeros((size,))
                s[i] = 1
                states.append(s)
        return states

    def get_states_with_neighbours(self):
        states = []
        neighbours = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if not self.accesible_states[i, j]:  # If grid element is a wall, skip
                    continue
                neigh = []
                # If self is neighbour (due to wall), that neighbour is omitted
                if i - 1 >= 0 and self.accesible_states[i - 1, j]:
                    neigh.append(self._index_to_grid([i - 1, j]).reshape((-1,)))
                if i + 1 < self.size[0] and self.accesible_states[i + 1, j]:
                    neigh.append(self._index_to_grid([i + 1, j]).reshape((-1,)))
                if j - 1 >= 0 and self.accesible_states[i, j - 1]:
                    neigh.append(self._index_to_grid([i, j - 1]).reshape((-1,)))
                if j + 1 < self.size[1] and self.accesible_states[i, j + 1]:
                    neigh.append(self._index_to_grid([i, j + 1]).reshape((-1,)))
                neighbours.append(np.array(neigh))
                s = self._index_to_grid([i, j]).reshape((-1,))
                states.append(s)
        return states, neighbours


class GridWorld10x10(SimpleGridWorld):
    def __init__(self):
        super(GridWorld10x10, self).__init__(size=(10, 10))


class GridWorld25x25(SimpleGridWorld):
    def __init__(self):
        super(GridWorld25x25, self).__init__(size=(25, 25))


class GridWorld42x42(SimpleGridWorld):
    def __init__(self):
        super(GridWorld42x42, self).__init__(size=(42, 42))


class GridWorldRandFeatures42x42(GridWorldRandFeatures):
    def __init__(self):
        super(GridWorldRandFeatures42x42, self).__init__(size=(42, 42))


class GridWorldSubspace50x50(GridWorldSubspaces):
    def __init__(self):
        super(GridWorldSubspace50x50, self).__init__(size=(50, 50))


class GridWorldBox11x11(GridWorldLoad):
    def __init__(self):
        super(GridWorldBox11x11, self).__init__(map_path='maps/Box.txt')


class GridWorldSpiral28x28(GridWorldLoad):
    def __init__(self):
        super(GridWorldSpiral28x28, self).__init__(map_path='maps/Spiral.txt')


class GridWorldSpiral52x50(GridWorldLoad):
    def __init__(self):
        super(GridWorldSpiral52x50, self).__init__(map_path='maps/Spiral_large.txt')
