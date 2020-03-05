import torch
import torch.nn as nn


class Network1D(nn.Module):

    def __init__(self, x_dim, y_dim, hidden_dim=(256,), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.layers = []
        h_dim_prev = x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, y_dim[0]))
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        s_t = self.apply_tensor_constraints(s_t)
        return self.model(s_t)

    def apply_tensor_constraints(self, x: torch.Tensor) -> torch.Tensor:
        # assert type(x) == torch.Tensor
        # if len(tuple(x.shape)) != 1 and len(tuple(x.shape)) != 2:
        #     raise ValueError("Encoder input tensor should be 1D (single example) or 2D (batch). Received "+str(x.shape))
        if len(tuple(x.shape)) == 1:  # Add batch dimension to 1D tensor
            x = x.unsqueeze(0)
        # if self.cuda and not x.is_cuda:
        x = x.to(self.device)
        # assert tuple(x.shape[1:]) == self.x_dim
        return x

    def get_z_dim(self) -> tuple:
        return self.z_dim

    def get_x_dim(self) -> tuple:
        return self.x_dim


class Network2D(nn.Module):

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)

    STANDARD_CONV = ({'channel_num': 32, 'kernel_size': 8, 'stride': 4, 'padding': 0},
                     {'channel_num': 64, 'kernel_size': 4, 'stride': 2, 'padding': 0},
                     {'channel_num': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0})

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
        # s_t = self.apply_tensor_constraints(s_t)
        return self.model(s_t)

    def apply_tensor_constraints(self, x: torch.Tensor) -> torch.Tensor:
        assert type(x) == torch.Tensor
        if len(tuple(x.shape)) != 3 and len(tuple(x.shape)) != 4:
            raise ValueError("Encoder input tensor should be 3D (single image) or 4D (batch).")
        if len(tuple(x.shape)) == 3:  # Add batch dimension to 3D tensor
            x = x.unsqueeze(0)
        if self.cuda and not x.is_cuda:
            x = x.to(self.device)
        assert tuple(x.shape[1:]) == self.x_dim
        return x

    def get_z_dim(self) -> tuple:
        return self.z_dim

    def get_x_dim(self) -> tuple:
        return self.x_dim
