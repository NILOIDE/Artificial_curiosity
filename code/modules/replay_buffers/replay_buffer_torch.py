import torch
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def update_rewards(self, rewards, idxes):
        """"
        Custom function added by me. Replaces reward value in gives transition indices.
        """
        assert type(rewards) == torch.tensor or type(rewards) == list
        for i, new_r in zip(idxes, rewards):
            (obs_t, action, _, obs_tp1, done) = self._storage[i]
            self._storage[i] = (obs_t, action, new_r, obs_tp1, done)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(torch.as_tensor(obs_t))
            actions.append(torch.as_tensor(action))
            rewards.append(reward)
            obses_tp1.append(torch.as_tensor(obs_tp1))
            dones.append(done)
        return torch.stack(obses_t), torch.stack(actions), torch.stack(rewards), torch.stack(obses_tp1), torch.stack(dones)

    def sample_states(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        obses_t = []
        for i in idxes:
            obs_t, *_ = self._storage[i]
            obses_t.append(obs_t)
        return torch.stack(obses_t)

    def return_all(self):
        return self._encode_sample(torch.tensor(len(self._storage)))


class DynamicsReplayBuffer(object):
    def __init__(self, size, device):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self.device = device
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, obs_tp1, done):
        data = (obs_t, action, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, obs_tp1, done = data
            obses_t.append(torch.as_tensor(obs_t))
            actions.append(torch.as_tensor(action))
            obses_tp1.append(torch.as_tensor(obs_tp1))
            dones.append(done)
        return torch.tensor(obses_t), torch.tensor(actions), torch.tensor(obses_tp1), torch.tensor(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_states(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        obses_t = []
        for i in idxes:
            obs_t, *_ = self._storage[i]
            obses_t.append(obs_t)
        return torch.stack(obses_t)

    def return_all(self):
        return self._encode_sample(torch.arange(len(self._storage)))
