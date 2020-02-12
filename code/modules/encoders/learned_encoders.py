import torch
import torch.nn as nn
from modules.encoders.base_encoder import BaseEncoder


class Encoder_2D(BaseEncoder):

    def __init__(self, x_dim=(3, 84, 84), conv_layers=None, fc_dim=512, z_dim=(20,), batch_norm=True, device='cpu'):
        # type: (tuple, tuple, int, tuple, bool, str) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
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
        x = self.apply_tensor_constraints(x)
        output = self.model(x)
        mu = self.mu_head(output)
        log_sigma = self.sigma_head(output)
        return mu, log_sigma