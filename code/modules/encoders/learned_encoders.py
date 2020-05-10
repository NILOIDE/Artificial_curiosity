import torch
import torch.nn as nn
from modules.encoders.base_encoder import BaseEncoder


class Encoder_2D_Sigma(BaseEncoder):

    def __init__(self, x_dim, z_dim, conv_layers=None, fc_dim=512, batch_norm=True, device='cpu'):
        # type: (tuple, tuple, tuple, int, bool, str) -> None
        """"
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
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
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(layer['channel_num']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        # if batch_norm:
        #     self.layers.append(nn.BatchNorm1d(fc_dim))  # Causes issues if batch size is 1
        self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim)).to(self.device)
        self.sigma_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim)).to(self.device)

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and variance with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        # x = self.apply_tensor_constraints(x)
        output = self.model(x)
        mu = self.mu_head(output)
        log_sigma = self.sigma_head(output)
        return mu, log_sigma


class Encoder_1D_Sigma(BaseEncoder):

    def __init__(self, x_dim, z_dim, hidden_dim=(64,), batch_norm=False, device='cpu'):
        # type: (tuple, tuple, tuple, bool, str) -> None
        """"
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        h_dim_prev = self.x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.model = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim[-1], *self.z_dim)).to(self.device)
        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim[-1], *self.z_dim)).to(self.device)

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and variance with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        # x = self.apply_tensor_constraints(x)
        output = self.model(x)
        mu = self.mu_head(output)
        log_sigma = self.sigma_head(output)
        return mu, log_sigma


class Encoder_1D(BaseEncoder):

    def __init__(self, x_dim, z_dim, hidden_dim=(64,), batch_norm=True, device='cpu'):
        # type: (tuple, tuple, tuple, bool, str) -> None
        """"
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        h_dim_prev = self.x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(hidden_dim[-1], *self.z_dim))
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and variance with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        # x = self.apply_tensor_constraints(x)
        return self.model(x)
