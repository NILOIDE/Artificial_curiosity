import torch
import torch.nn as nn
from modules.encoders.learned_encoders import Encoder_2D_Sigma, Encoder_1D_Sigma
from modules.decoders.decoder import Decoder_2D, Decoder_2D_conv, Decoder_1D


class VAE(nn.Module):

    def __init__(self, x_dim, device='cpu', **kwargs):
        # type: (tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.avgdims = tuple(range(1, len(x_dim)+1))
        self.encoder = self._encoder(len(x_dim), **kwargs)
        self.decoder = self._decoder(len(x_dim), **kwargs)
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mu, log_sigma = self.encoder(x)
        noise = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
        z = mu + log_sigma.exp() * noise
        im_recon = self.decoder(z)

        l_reg = 0.5 * torch.sum(log_sigma.exp() + mu ** 2 - log_sigma - 1, dim=1)
        target = self.encoder.apply_tensor_constraints(x)
        l_recon = torch.sum(self.loss(im_recon, target), dim=self.avgdims)
        average_negative_elbo = torch.mean(l_recon + l_reg, dim=0)

        return average_negative_elbo, z, im_recon, mu, log_sigma

    def _encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return Encoder_1D_Sigma(x_dim=self.x_dim,
                                    z_dim=self.z_dim,
                                    batch_norm=kwargs['encoder_batchnorm'],
                                    device=self.device)  # type: Encoder_1D_Sigma
        elif input_dim == 3:
            return Encoder_2D_Sigma(x_dim=self.x_dim,
                                    z_dim=self.z_dim,
                                    conv_layers=kwargs['conv_layers'],
                                    device=self.device)  # type: Encoder_2D_Sigma
        else:
            raise AssertionError(f'Input dim should be 1 (vector) or 3 (image), but received: {input_dim}')

    def _decoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return Decoder_1D(x_dim=self.x_dim,
                              z_dim=self.z_dim,
                              device=self.device)  # type: Decoder_2D
        elif input_dim == 3:
            return Decoder_2D_conv(x_dim=self.x_dim,
                                   z_dim=self.z_dim,
                                   conv_layers=kwargs['conv_layers'],
                                   device=self.device)  # type: Decoder_2D_conv
        else:
            raise AssertionError(f'Input dim should be 1 (vector) or 3 (image), but received: {input_dim}')

    def sample(self, n_samples):
        # type: (int) ->  [torch.Tensor, torch.Tensor]
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        with torch.no_grad():
            z = torch.normal(torch.zeros(n_samples, self.z_dim[0]), torch.ones(n_samples, self.z_dim[0]))
            sampled_imgs = self.decoder(z).view(n_samples, *self.x_dims[1:])
            im_means = sampled_imgs.mean(dim=0)

        return sampled_imgs, im_means

    def forward_no_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """"
        Forward pass for whenever we are interested in seeing the reconstruction
        of the mean of the latent distribution.
        """
        with torch.no_grad():
            mu, _ = self.encoder(x)
            y = self.decoder(mu)
        return y

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.decoder(z)
        return y

    def encode(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    def get_x_dim(self) -> tuple:
        return self.x_dim

    def get_z_dim(self) -> tuple:
        return self.z_dim


if __name__ == "__main__":

    #################### MNIST PERFORMANCE TEST ###############################

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torchvision
    import os


    def create_conv_layer_dict(params: tuple) -> dict:
        return {'channel_num': params[0],
                'kernel_size': params[1],
                'stride': params[2],
                'padding': params[3]}


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    root_path = os.getcwd().partition("Adversarial_curiosity\code")
    data_path = root_path[0] + root_path[1] + "\data"
    mnist_trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    train_data = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)
    ex = mnist_testset[0][0]
    x_dim = tuple(ex.shape)
    conv_layers = (create_conv_layer_dict((32, 8, 4, 0)),)
    vae = VAE(x_dim, conv_layers=conv_layers, device=device, **{'z_dim': (20,)})  # type: VAE
    optimizer = torch.optim.Adam(vae.parameters())
    for i, (batch, target) in enumerate(train_data):
        batch = batch.to(device)
        vae.zero_grad()
        loss, _, _, _, _ = vae(batch)
        if i % 200 == 0:
            print(i, loss.item())
        loss.backward()
        optimizer.step()
        if i == 1000:
            break
    import matplotlib.pyplot as plt

    test_data = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=True)

    batch = next(iter(test_data))
    for i, d in enumerate(batch[0]):
        fig = plt.figure(i)
        ims = vae.forward_mean_only(d.to(device))
        ims = ims.detach().cpu()
        plt.imshow(torchvision.utils.make_grid(ims).permute(1, 2, 0))
        plt.imshow(ims.reshape((x_dim[-2], x_dim[-1])))
        if i > 9:
            break
