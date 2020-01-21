import argparse
import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.shared_layers = nn.Sequential(nn.Linear(x_dim, hidden_dim),
                              nn.ReLU())
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, z_dim))
        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim, z_dim))

    def forward(self, x):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        shared_output = self.shared_layers(x)
        mean = self.mu_head(shared_output)
        log_std = self.sigma_head(shared_output)

        return mean, log_std


class Decoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.layers = [nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                       nn.Linear(hidden_dim, x_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Perform forward pass of encoder.
        Returns mean with shape [batch_size, 784].
        """

        mean = self.model(x)

        return mean


class VAE(nn.Module):

    def __init__(self, x_dims, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.x_dims = x_dims
        self.encoder = Encoder(x_dim=x_dims[-2]*x_dims[-1], hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(x_dim=x_dims[-2]*x_dims[-1], hidden_dim=hidden_dim, z_dim=z_dim)
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, x):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, log_std = self.encoder(x)
        noise = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        z = mean + log_std.exp() * noise
        out = self.decoder(z)

        l_reg = 0.5 * torch.sum(log_std.exp() + mean**2 - log_std - 1, dim=1)
        l_recon = torch.sum(self.loss(out, x), dim=1)
        average_negative_elbo = torch.mean(l_recon + l_reg, dim=0)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        with torch.no_grad():
            z = torch.normal(torch.zeros(n_samples, self.z_dim), torch.ones(n_samples, self.z_dim))
            sampled_imgs = self.decoder(z).view(n_samples, *self.x_dims[1:])
            im_means = sampled_imgs.mean(dim=0)

        return sampled_imgs, im_means
