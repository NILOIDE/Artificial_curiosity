import torch
import torch.nn as nn

class RandomEncoder_1D(nn.Module):
    def __init__(self, x_dim, hidden_dim=(256,), z_dim=20):
        # type: (int, tuple, int) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        The latent representation is assumed to be Gaussian.
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        print('Encoder has input dimension:', x_dim)
        print('Encoder has ouput dimension:', z_dim)
        self.layers = []
        h_dim_prev = x_dim
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, z_dim))
        self.model = nn.Sequential(*self.layers)
        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        assert type(x) == torch.Tensor
        if len(tuple(x.shape)) != 1 and len(tuple(x.shape)) != 2:
            raise ValueError("Input tensor should be 1D (single example) or 2D (batch)")
        assert x.shape[-1] == self.x_dim
        out = self.model(x)
        return out


class RandomEncoder_1D_sigma(nn.Module):
    def __init__(self, x_dim, hidden_dim=(256,), z_dim=20):
        # type: (int, tuple, int) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        The latent representation is assumed to be Gaussian.
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        print('Encoder has input dimension:', x_dim)
        print('Encoder has ouput dimension:', z_dim)
        self.layers = []
        h_dim_prev = x_dim
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.model = nn.Sequential(*self.layers)
        for param in self.network.parameters():
            param.requires_grad = False
        self.mu_head = nn.Sequential(nn.Linear(h_dim_prev, z_dim))
        for param in self.mu_head.parameters():
            param.requires_grad = False
        self.sigma_head = nn.Sequential(nn.Linear(h_dim_prev, z_dim))
        for param in self.sigma_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        assert type(x) == torch.Tensor
        if len(tuple(x.shape)) != 1 and len(tuple(x.shape)) != 2:
            raise ValueError("Input tensor should be 1D (single example) or 2D (batch)")
        assert x.shape[-1] == self.x_dim
        output = self.model(x)
        mean = self.mu_head(output)
        log_std = self.sigma_head(output)
        return mean, log_std


class RandomEncoder_2D(nn.Module):

    def __init__(self, x_dim=(3, 84, 84), conv_layers=(), fc_dim=512, z_dim=(20,)):
        # type: (tuple, tuple, int, int) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__()

        class Flatten(torch.nn.Module):
            def forward(self, x):
                return x.view(x.size()[0], -1)

        self.x_dim = x_dim
        self.z_dim = z_dim
        print('Encoder has input dimension:', x_dim)
        print('Encoder has ouput dimension:', z_dim)
        self.layers = []
        prev_channels = x_dim[0]
        prev_dim_x = x_dim[1]
        prev_dim_y = x_dim[2]
        for layer in conv_layers:
            self.layers.append(nn.Conv2d(prev_channels,
                                         layer['channel_num'],
                                         kernel_size=layer['kernel_size'],
                                         stride=layer['stride'],
                                         padding=layer['padding']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            # TODO: Check that these calculations are correct
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - (layer['kernel_size'] - 1)) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - (layer['kernel_size'] - 1)) // layer['stride'] + 1
        self.layers.append(Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_dim, *z_dim))
        self.model = nn.Sequential(*self.layers)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        assert type(x) == torch.Tensor
        if len(tuple(x.shape)) != 3 and len(tuple(x.shape)) != 4:
            raise ValueError("Input tensor should be 3D (single example) or 4D (batch)")
        assert tuple(x.shape[-3:]) == self.x_dim
        out = self.model(x)
        return out


class RandomEncoder_2D_sigma(nn.Module):

    def __init__(self, x_dim=(3, 84, 84), conv_layers=(), fc_dim=512, z_dim=(20,)):
        # type: (tuple, tuple, int, int) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__()

        class Flatten(torch.nn.Module):
            def forward(self, x):
                return x.view(x.size()[0], -1)

        self.x_dim = x_dim
        self.z_dim = z_dim
        print('Encoder has input dimension:', x_dim)
        print('Encoder has ouput dimension:', z_dim)
        self.layers = []
        prev_channels = x_dim[0]
        prev_dim_x = x_dim[1]
        prev_dim_y = x_dim[2]
        for layer in conv_layers:
            self.layers.append(nn.Conv2d(prev_channels,
                                         layer['channel_num'],
                                         kernel_size=layer['kernel_size'],
                                         stride=layer['stride'],
                                         padding=layer['padding']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            # TODO: Check that these calculations are correct
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - (layer['kernel_size'] - 1)) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - (layer['kernel_size'] - 1)) // layer['stride'] + 1
        self.layers.append(Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)
        for param in self.model.parameters():
            param.requires_grad = False
        self.mu_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim))
        for param in self.mu_head.parameters():
            param.requires_grad = False
        self.sigma_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim))
        for param in self.sigma_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        assert type(x) == torch.Tensor
        if len(tuple(x.shape)) != 3 and len(tuple(x.shape)) != 4:
            raise ValueError("Input tensor should be 3D (single example) or 4D (batch)")
        assert tuple(x.shape[-3:]) == self.x_dim
        output = self.model(x)
        mean = self.mu_head(output)
        log_std = self.sigma_head(output)
        return mean, log_std


class Decoder(nn.Module):

    def __init__(self, x_dim, z_dim, hidden_dim=(512, 512)):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = x_dim[-2] * x_dim[-1]
        print('Encoder has input dimension:', z_dim)
        print('Encoder has ouput dimension:', self.y_dim, '(viewed as', self.x_dim, ')')
        h_dim_prev = z_dim[0]
        self.layers = []
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, self.y_dim))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """
        assert type(x) == torch.Tensor
        out = self.model(x)
        if len(list(x.shape)) == 1:
            out = out.view(self.x_dim)
        elif len(list(x.shape)) == 2:
            out = out.view((-1, *self.x_dim))
        return out


def create_conv_layer_dict(params: tuple) -> dict:
    return {'channel_num': params[0],
            'kernel_size': params[1],
            'stride': params[2],
            'padding': params[3]}


if __name__ == "__main__":
    import argparse
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torchvision

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--max_norm', default=10, type=float,
                        help='gradient clip value')
    parser.add_argument('--num_rows', default=10, type=int,
                        help='number of rows in sample image')
    parser.add_argument('--manifold_rows', default=20, type=int,
                        help='number of rows in sample image')

    ARGS = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_data = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)
    ex = mnist_testset[0][0]
    x_dim = tuple(ex.shape)
    conv_layers = (create_conv_layer_dict((32, 8, 4, 0)),)
    en = RandomEncoder_2D(x_dim=x_dim, conv_layers=conv_layers).to(device)
    de = Decoder(x_dim=x_dim, z_dim=en.z_dim).to(device)
    loss_func = nn.BCELoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(de.parameters())
    for i, (batch, target) in enumerate(train_data):
        batch = batch.to(device)
        de.zero_grad()
        z = en(batch).detach()
        out = de(z)
        loss = torch.sum(loss_func(out, batch))
        if i % 200 == 0:
            print(i, loss.item())
        loss.backward()
        optimizer.step()
        if i == 1000:
            break
    import matplotlib.pyplot as plt

    test_data = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=True)

    batch = next(iter(test_data))
    en = en.cpu()
    de = de.cpu()
    for i, d in enumerate(batch[0]):
        fig = plt.figure(i)
        z = en(d.unsqueeze(0)).detach()
        out = de(z).detach()#.reshape((1, 1, x_dim[-2], x_dim[-1]))
        plt.imshow(torchvision.utils.make_grid(out).permute(1, 2, 0))
        plt.imshow(out.reshape(( x_dim[-2], x_dim[-1])))
        if i > 9:
            break

