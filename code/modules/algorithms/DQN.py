import torch
import numpy as np
import copy
import math
from modules.algorithms.network import Network1D, Network2D
import os

class DQN:
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.gamma = kwargs['gamma']
        self.epsilon_static = kwargs['eps_static']
        self.epsilon_min = kwargs['eps_min']
        if self.epsilon_static:
            self.epsilon = self.epsilon_min
            self.epsilon_tau = None
        else:
            self.epsilon = 1.0
            self.epsilon_tau = -kwargs['train_steps'] / 2 / math.log(kwargs['eps_half'])
            print('Epsilon at 20%:', math.exp(-0.2 * kwargs['train_steps'] / self.epsilon_tau))
            print('Epsilon at 80%:', math.exp(-0.8 * kwargs['train_steps'] / self.epsilon_tau))

        self.device = device
        if len(x_dim) == 1:
            self.network = Network1D(x_dim, a_dim, device=device)  # type: Network1D
        else:
            self.network = Network2D(x_dim, a_dim, device=device, **kwargs)  # type: Network2D
        self.loss_func = torch.nn.SmoothL1Loss().to(device)
        # self.loss_func = torch.nn.MSELoss.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=kwargs['alg_lr'])
        self.soft_target = kwargs['alg_soft_target']
        self.target_network_steps = kwargs['alg_target_net_steps']
        self.target_network = None
        if self.soft_target or self.target_network_steps != 0:
            self.target_network = copy.deepcopy(self.network).to(device)
        self.losses = []
        self.train_steps = 0
        print('DQN Architecture:')
        print(self.network)

    def act(self, s_t, eval=False, eps=None):
        # type: (torch.Tensor, bool, float) -> torch.tensor
        """"
        Return argmax over all actions for given states.
        """
        if eps is None:
            eps = self.epsilon
        if eval or torch.rand(1)[0] >= eps:
            with torch.no_grad():
                return torch.argmax(self.network(s_t), dim=1)
        else:
            if len(s_t.shape) == len(self.x_dim):
                batch_size = 1
            else:
                batch_size = s_t.shape[0]
            return torch.randint(self.a_dim[0], (batch_size,))

    def train(self, s_t, a_t, r_t, s_tp1, d_t):
        # type: (torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.IntTensor) -> None
        """"
        Tune Q-values for the given actions in s_t so as to be closer to the observed r_t.
        """
        s_t = s_t.to(device=self.device)
        a_t = a_t.to(device=self.device)
        s_tp1 = s_tp1.to(device=self.device)
        r_t = r_t.to(device=self.device)
        d_t = d_t.to(device=self.device)
        assert len(tuple(a_t.shape)) == 1
        assert len(tuple(r_t.shape)) == 1
        assert len(tuple(d_t.shape)) == 1
        assert torch.isnan(r_t).sum() == 0, r_t
        self.network.zero_grad()
        idx = torch.arange(a_t.shape[0]).long()
        q_t = self.network(s_t)
        bootstrap_network = self.target_network if self.target_network is not None else self.network
        with torch.no_grad():
            q_t_target = q_t.clone().detach()
            t = r_t + (-d_t + 1).float() * self.gamma * bootstrap_network(s_tp1).detach().max(dim=1).values
            assert len(t.shape) == 1
            assert t.shape[0] == q_t.shape[0]
            q_t_target[idx, a_t] = t
        loss = self.loss_func(q_t, q_t_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.train_steps += 1
        self.update_target_network()
        self.update_epsilon()

    def forward(self, s_t, a_t):
        # type: (torch.Tensor, torch.LongTensor) -> torch.Tensor
        if isinstance(s_t, np.ndarray):
            s_t = torch.from_numpy(s_t)
        if isinstance(a_t, np.ndarray):
            a_t = torch.from_numpy(a_t).to(dtype=torch.long)
        if isinstance(a_t, int):
            a_t = torch.tensor([a_t]).to(dtype=torch.long)
        s_t = s_t.to(device=self.device)
        a_t = a_t.to(device=self.device)
        # TODO: This is not general, made to work for eval
        q_t = self.network(s_t)[0][a_t]
        return q_t.item()

    def update_target_network(self):
        if self.target_network is not None:
            if self.soft_target:
                for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.01) + param.data * 0.01)
            else:
                assert self.target_network_steps != 0
                if self.train_steps % self.target_network_steps == 0:
                    self.target_network = copy.deepcopy(self.network)
        else:
            assert not self.soft_target or self.target_network_steps == 0

    def update_epsilon(self):
        if self.epsilon_static:
            pass
        else:
            self.epsilon = max(self.epsilon_min, math.exp(-self.train_steps/self.epsilon_tau))

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save({'train_steps': self.train_steps,
                    'network': self.network,
                    'optimizer': self.optimizer,
                    'losses': self.losses
                    }, folder_path + 'DQN_items.pt')

    def load(self, path):
        checkpoint = torch.load(path)
        self.train_steps = checkpoint['train_steps']
        self.network = checkpoint['network']
        self.optimizer = checkpoint['optimizer']
        self.losses = checkpoint['losses']


class TabularQlearning:
    def __init__(self, obs_dim, a_dim, gamma, epsilon, lr=0.001):
        self.obs_dim = obs_dim
        self.a_dim = a_dim
        self.q_values = {}
        self.lr = 1.0
        self.gamma = gamma
        self.epsilon = epsilon
        self.train_steps = 0
        self.losses = []

    def act(self, s_t, eval=False, eps=None):
        # type: (np.ndarray, bool, float) -> int
        if eps is None:
            eps = self.epsilon
        key = str(s_t.tolist())
        if key not in self.q_values:  # If state has never been visited before, take random action
            a_t = torch.randint(self.a_dim[0] - 1, (1,))[0] + 1
            return int(a_t)
        if eval:
            a_t = np.argmax(self.q_values[key])
            return int(a_t)
        if torch.rand(1).item() >= eps:
            a_t = np.argmax(self.q_values[key])
            return int(a_t)
        else:
            a_t = torch.randint(self.a_dim[0] -1, (1,))[0]+1
            return int(a_t)

    def train(self, s_t, a_t, r_t, s_tp1, d_t):
        # type: (np.ndarray, int, float, np.ndarray, bool) -> None
        s_t_key = str(s_t.tolist())
        s_tp1_key = str(s_tp1.tolist())
        if s_t_key not in self.q_values:
            self.q_values[s_t_key] = [0.0 for _ in range(self.a_dim[0])]
        # print('--------')
        # print(np.argmax(s_t))
        # print(self.q_values[s_t_key])
        q_t = self.q_values[s_t_key]
        if s_tp1_key not in self.q_values:
            self.q_values[s_tp1_key] = [0.0 for _ in range(self.a_dim[0])]
        q_tp1 = self.q_values[s_tp1_key]
        new_q = r_t + self.gamma * np.max(q_tp1)
        self.losses.append(np.abs(new_q - q_t[a_t]))
        q_t[a_t] += self.lr * (new_q - q_t[a_t])
        # print(self.q_values[s_t_key], a_t)
        self.train_steps += 1

    def forward(self, s_t, a_t):
        # type: (np.ndarray, int) -> float
        assert len(s_t.shape) == 1
        key = str(s_t.tolist())
        if key in self.q_values:
            return self.q_values[key][a_t]
        else:
            return 0.0

    def save(self, path='saved_objects/dqn_items.pt'):
        torch.save({'train_steps': self.train_steps,
                    'losses': self.losses
                    }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.train_steps = checkpoint['train_steps']
        self.losses = checkpoint['losses']

