import torch
import torch.nn as nn

STANDARD_CONV = ({'channel_num': 32, 'kernel_size': 8, 'stride': 4, 'padding': 0},
                 {'channel_num': 64, 'kernel_size': 4, 'stride': 2, 'padding': 0},
                 {'channel_num': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0})


class BaseEncoder(nn.Module):

    class Flatten(torch.nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)

    def __init__(self, x_dim, z_dim, device='cpu'):
        # type: (tuple, tuple, str) -> None
        """"
        This is the base class for the encoders below. Contains all the required shared variables and functions.
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        print('Encoder has dimensions:', x_dim, '->', z_dim)
        if device in {'cuda', 'cpu'}:
            self.device = device
            self.cuda = True if device == 'cuda' else False

    def apply_tensor_constraints(self, x):
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


class RandomEncoder_1D(BaseEncoder):
    def __init__(self, x_dim, hidden_dim=(256,), z_dim=(20,), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        h_dim_prev = x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, z_dim[0]))
        self.model = nn.Sequential(*self.layers).to(self.device)
        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        return self.model(x)


class RandomEncoder_1D_sigma(BaseEncoder):
    def __init__(self, x_dim, hidden_dim=(256,), z_dim=(20,), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        h_dim_prev = x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.model = nn.Sequential(*self.layers).to(self.device)
        for param in self.network.parameters():
            param.requires_grad = False
        self.mu_head = nn.Sequential(nn.Linear(h_dim_prev, z_dim[0])).to(self.device)
        for param in self.mu_head.parameters():
            param.requires_grad = False
        self.sigma_head = nn.Sequential(nn.Linear(h_dim_prev, z_dim[0])).to(self.device)
        for param in self.sigma_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and variance with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        output = self.model(x)
        mean = self.mu_head(output)
        log_sigma = self.sigma_head(output)
        return mean, log_sigma


class RandomEncoder_2D(BaseEncoder):

    def __init__(self, x_dim=(3, 84, 84), conv_layers=STANDARD_CONV, fc_dim=512, z_dim=(20,), device='cpu'):
        # type: (tuple, tuple, int, tuple, str) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        prev_channels = self.x_dim[0]
        prev_dim_x = self.x_dim[1]
        prev_dim_y = self.x_dim[2]
        for layer in conv_layers:
            self.layers.append(nn.Conv2d(prev_channels,
                                         layer['channel_num'],
                                         kernel_size=layer['kernel_size'],
                                         stride=layer['stride'],
                                         padding=layer['padding']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            # TODO: Check that these calculations are correct
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - (layer['kernel_size'])) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - (layer['kernel_size'])) // layer['stride'] + 1
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_dim, *self.z_dim))
        self.model = nn.Sequential(*self.layers).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        return self.model(x)


class RandomEncoder_2D_sigma(BaseEncoder):

    def __init__(self, x_dim=(3, 84, 84), conv_layers=STANDARD_CONV, fc_dim=512, z_dim=(20,), device='cpu'):
        # type: (tuple, tuple, int, tuple, str) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
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
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.mu_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim)).to(self.device)
        for param in self.mu_head.parameters():
            param.requires_grad = False
        self.sigma_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim)).to(self.device)
        for param in self.sigma_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and variance with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        output = self.model(x)
        mean = self.mu_head(output)
        log_sigma = self.sigma_head(output)
        return mean, log_sigma


def create_conv_layer_dict(params: tuple) -> dict:
    return {'channel_num': params[0],
            'kernel_size': params[1],
            'stride': params[2],
            'padding': params[3]}


if __name__ == "__main__":
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torchvision
    from modules.decoders.decoder import Decoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_data = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)
    ex = mnist_testset[0][0]
    x_dim = tuple(ex.shape)
    conv_layers = (create_conv_layer_dict((32, 8, 4, 0)),)
    en = RandomEncoder_2D(x_dim=x_dim, conv_layers=conv_layers).to(device)
    de = Decoder(z_dim=en.z_dim, x_dim=x_dim, device=device)
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
        out = de(z).detach()
        plt.imshow(torchvision.utils.make_grid(out).permute(1, 2, 0))
        plt.imshow(out.reshape(( x_dim[-2], x_dim[-1])))
        if i > 9:
            break

