import torch
import torch.nn as nn


class BaseDecoder(nn.Module):

    class View2D(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return x.view(x.shape[0], *self.dim)

    STANDARD_CONV = ({'channel_num': 64, 'kernel_size': 8, 'stride': 4, 'padding': 0},
                     {'channel_num': 64, 'kernel_size': 4, 'stride': 2, 'padding': 0},
                     {'channel_num': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0})
    
    def __init__(self, z_dim, x_dim, device='cpu'):
        # type: (tuple, tuple, str) -> None
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        print('Decoder has dimensions:', z_dim, '->', self.x_dim)
        if device in {'cuda', 'cpu'}:
            self.device = device
            self.cuda = True if device == 'cuda' else False

    def apply_tensor_constraints(self, z: torch.Tensor) -> torch.Tensor:
        assert type(z) == torch.Tensor
        assert z.shape[-1] == self.z_dim[0]
        if len(tuple(z.shape)) != 1 and len(tuple(z.shape)) != 2:
            raise ValueError("Decoder input tensor should be 1D (single example) or 2D (batch).")
        if len(tuple(z.shape)) == 1:  # Add batch dimension to 1D tensor
            z = z.unsqueeze(0)
        if self.cuda and not z.is_cuda:
            z = z.to(self.device)
        return z

    def determine_upconv_dims(self, conv_layers: list) -> list:
        prev_channels = self.x_dim[0]
        prev_dim_x = self.x_dim[1]
        prev_dim_y = self.x_dim[2]
        dims = [(prev_channels, prev_dim_x, prev_dim_y)]
        for layer in conv_layers:
            prev_channels = layer['channel_num']
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
            if prev_dim_x == 0 or prev_dim_y == 0 or prev_channels == 0:
                raise ValueError("Upconv dimensions cannot be zero: " + str((prev_channels, prev_dim_x, prev_dim_y)))
            dims.append((prev_channels, prev_dim_x, prev_dim_y))
        # dims.append((self.x_dim[0], self.x_dim[1], self.x_dim[2]))
        return dims


class Decoder_1D(BaseDecoder):

    def __init__(self, z_dim, x_dim, hidden_dim=(64,), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        super().__init__(z_dim, x_dim, device)
        self.y_dim = x_dim
        h_dim_prev = z_dim[0]
        self.layers = []
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, self.y_dim[0]))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """
        z = self.apply_tensor_constraints(z)
        return self.model(z)


class Decoder_2D(BaseDecoder):

    def __init__(self, z_dim, x_dim, hidden_dim=(128, 128), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        super().__init__(z_dim, x_dim, device)
        self.y_dim = x_dim[0] * x_dim[1] * x_dim[2]
        h_dim_prev = z_dim[0]
        self.layers = []
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, self.y_dim))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """
        z = self.apply_tensor_constraints(z)
        y = self.model(z)
        x_prime = y.reshape((z.shape[0], *self.x_dim))
        assert x_prime.shape[-3:] == self.x_dim
        return x_prime


class Decoder_2D_conv(BaseDecoder):

    def __init__(self, z_dim, x_dim, conv_layers=None, fc_dim=512, device='cpu'):
        # type: (tuple, tuple, tuple, int, str) -> None
        super().__init__(z_dim, x_dim, device)
        self.conv_layers = list(self.STANDARD_CONV if conv_layers is None else conv_layers)
        self.layers = []
        self.upconv_dims = self.determine_upconv_dims(self.conv_layers)
        self.conv_layers.reverse()
        self.upconv_dims.reverse()
        conv_input_size = self.upconv_dims[0][0] * self.upconv_dims[0][1] * self.upconv_dims[0][2]
        self.layers.append(nn.Linear(self.z_dim[0], fc_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_dim, conv_input_size))
        self.layers.append(self.View2D(self.upconv_dims[0]))
        # self.layers.append(nn.BatchNorm2d(self.upconv_dims[0][0]))
        for prev_layer_dims, layer in zip(self.upconv_dims[1:], self.conv_layers):
            next_channels, next_dim_x, next_dim_y = prev_layer_dims[0], prev_layer_dims[1], prev_layer_dims[2]
            self.layers.append(nn.ReLU())
            self.layers.append(nn.ConvTranspose2d(layer['channel_num'],
                                                  next_channels,
                                                  kernel_size=layer['kernel_size'],
                                                  stride=layer['stride'],
                                                  padding=layer['padding']))
            # self.layers.append(nn.BatchNorm2d(next_channels))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """
        z = self.apply_tensor_constraints(z)
        y = self.model(z)
        x_prime = y.reshape((z.shape[0], *self.x_dim))
        assert x_prime.shape[-3:] == self.x_dim
        return x_prime