import torch
import torch.nn as nn


class BaseDecoder(nn.Module):

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


class Decoder(BaseDecoder):

    def __init__(self, z_dim, x_dim, hidden_dim=(512, 512), device='cpu'):
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
