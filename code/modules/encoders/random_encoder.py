import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.shared_layers = nn.Sequential(nn.Linear(x_dim, hidden_dim),
                              nn.ReLU())
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, z_dim))
        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim, z_dim))


