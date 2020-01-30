import torch
import torch.nn as nn
import copy
import random


class Network(nn.Module):

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)

    STANDARD_CONV = ({'channel_num': 32, 'kernel_size': 8, 'stride': 4, 'padding': 0},
                     {'channel_num': 64, 'kernel_size': 4, 'stride': 2, 'padding': 0},
                     {'channel_num': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0})

    def apply_tensor_constraints(self, x: torch.Tensor) -> torch.Tensor:
        assert type(x) == torch.Tensor
        if len(tuple(x.shape)) != 3 and len(tuple(x.shape)) != 4:
            raise ValueError("Encoder input tensor should be 3D (single image) or 4D (batch).")
        if len(tuple(x.shape)) == 3:  # Add batch dimension to 1D tensor
            x = x.unsqueeze(0)
        if self.cuda and not x.is_cuda:
            x = x.to(self.device)
        assert tuple(x.shape[-3:]) == self.x_dim
        return x

    def get_z_dim(self) -> tuple:
        return self.z_dim

    def get_x_dim(self) -> tuple:
        return self.x_dim

    def __init__(self, x_dim, y_dim, conv_layers=None, fc_dim=512, device='cpu'):
        # type: (tuple, tuple, tuple, int, str) -> None
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.device = device
        self.conv_layers = self.STANDARD_CONV if conv_layers is None else conv_layers
        self.layers = []
        prev_channels = x_dim[0]
        prev_dim_x = x_dim[1]
        prev_dim_y = x_dim[2]
        for layer in self.conv_layers:
            self.layers.append(nn.Conv2d(prev_channels,
                                         layer['channel_num'],
                                         kernel_size=layer['kernel_size'],
                                         stride=layer['stride'],
                                         padding=layer['padding']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_dim, y_dim[0]))
        self.model = nn.Sequential(*self.layers).to(device)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        s_t = self.apply_tensor_constraints(s_t)
        return self.model(s_t)


class DQN:

    def __init__(self, x_dim, a_dim, gamma=0.95, target_net_steps=500, device='cuda'):
        # type: (tuple, tuple, float, int, str) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.device = device
        self.network = Network(x_dim, a_dim, device=device)
        self.loss_func = nn.MSELoss(reduction='none').to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.target_network = copy.deepcopy(self.network).to(device)
        self.target_network_steps = target_net_steps
        self.losses = []
        self.train_steps = 0

    def act(self, s_t, eval=False):
        # type: (torch.Tensor, bool) -> int
        """"
        Return argmax over all actions for given states.
        """
        if eval or random.random() >= self.epsilon:
            with torch.no_grad():
                return torch.argmax(self.network(s_t), dim=1).item()
        else:
            return torch.randint(self.a_dim[0], (1,)).item()

    def train(self, s_t, a_t, r_t, s_tp1, d_t):
        # type: (torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor) -> None
        """"
        Tune Q-values for the given actions in s_t so as to be closer to the observed r_t.
        """
        s_t = s_t.to(device=self.device)
        a_t = a_t.to(device=self.device)
        s_tp1 = s_tp1.to(device=self.device)
        r_t = r_t.to(device=self.device)
        d_t = d_t.to(device=self.device)
        self.network.zero_grad()
        idx = torch.arange(a_t.shape[0])
        q_t = self.network(s_t)[idx, a_t[0]]
        with torch.no_grad():
            q_t_target = r_t + (1.0 - d_t) * self.gamma * torch.max(self.target_network(s_tp1), dim=1)[0]
        assert len(tuple(q_t.shape)) == 1
        assert len(tuple(q_t_target.shape)) == 1
        loss = torch.sum(self.loss_func(q_t, q_t_target))
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.train_steps += 1
        if self.train_steps % self.target_network_steps == 0:
            self.update_target_network()
        self.update_epsilon()

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.network)

    def update_epsilon(self):
        if self.epsilon > 0.02:
            self.epsilon -= 0.0001

