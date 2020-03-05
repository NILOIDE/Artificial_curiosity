import torch
import torch.nn as nn
from modules.world_models.forward_model import ForwardModel_SigmaInputOutput, ForwardModel_SigmaOutputOnly, ForwardModel
from modules.encoders.learned_encoders import Encoder_2D_Sigma, Encoder_1D_Sigma, Encoder_1D
from modules.decoders.decoder import Decoder_1D, Decoder_2D
from torch.distributions import kl, Normal, MultivariateNormal
import copy


class StochasticEncodedFM(nn.Module):
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.encoder = self.create_encoder(len(x_dim), **kwargs)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.tau = kwargs['wm_tau']
        self.world_model = ForwardModel(x_dim=self.encoder.get_z_dim(),
                                        a_dim=self.a_dim,
                                        device=self.device)  # type: ForwardModel
        # self.loss_func_stochastic = kl.kl_divergence
        self.loss_func = torch.nn.SmoothL1Loss(reduction='none').to(device)
        self.steps = 0

    def forward(self, x_t, a_t, x_tp1, n=32):
        mu_t, log_sigma_t = self.encoder(x_t)
        noise = Normal(torch.zeros_like(mu_t), torch.ones_like(log_sigma_t)).sample((n,))  ##
        z_t = mu_t + log_sigma_t.exp() * noise  ##
        z_t = z_t.view((-1, self.z_dim[0]))  # Samples of same dist are every batch_len elements
        a_t = a_t.repeat((n,))  # Samples of same dist are every batch_len elements
        # mu_tp1_prime, log_sigma_tp1_prime = self.world_model.forward(z_t, a_t)
        # TODO: Does average of multiple samples help?
        z_tp1 = self.world_model.forward(z_t, a_t)
        mu_tp1, log_sigma_tp1 = self.target_encode(x_tp1)
        mu_tp1, log_sigma_tp1 = mu_tp1.detach().repeat((n, 1)), log_sigma_tp1.detach().repeat((n, 1))
        # distribution_tp1 = MultivariateNormal(mu_tp1_prime, torch.diag_embed((log_sigma_tp1_prime.exp())))
        # distribution_target = MultivariateNormal(mu_tp1, torch.diag_embed((log_sigma_tp1.exp())))
        # loss_rec = - distribution_target.log_prob(z_tp1).view((-1, n)).exp()
        loss_rec = ((z_tp1 - mu_tp1) / (log_sigma_tp1.exp() + 1e-8)) ** 2
        loss_rec = loss_rec.view((-1, self.z_dim[0], n)).sum(dim=1)
        assert loss_rec.shape == (x_t.shape[0], n)
        loss_rec = loss_rec.mean(dim=1)
        assert loss_rec.shape[0] == x_t.shape[0]
        # loss_reg = 0.5 * torch.sum(log_sigma_t.exp() + mu_t ** 2 - log_sigma_t - 1, dim=1)
        # loss_recon = self.loss_func(distribution_target, distribution_tp1)
        loss = loss_rec  # + 0.1 * loss_reg
        if torch.isnan(loss).sum() != 0:
            # print(torch.isnan(loss), z_tp1, distribution_target.log_prob(z_tp1))
            # print(mu_t, log_sigma_t.exp())
            # print(mu_tp1, log_sigma_tp1.exp())
            # print(log_sigma_tp1.min(), log_sigma_tp1.max(), log_sigma_tp1_prime.min(), log_sigma_tp1_prime.max())
            raise ValueError('Nan found in WM forward loss.')

        self.steps += 1
        if self.steps % self.target_steps == 0:
            self.update_target()
        # self.soft_update_target(self.tau)
        return loss

    def update_target(self):
        self.target_encoder = copy.deepcopy(self.encoder)

    def soft_update_target(self, tau):
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            encoder = Encoder_1D_Sigma(x_dim=self.x_dim,
                                       z_dim=self.z_dim,
                                       batch_norm=kwargs['encoder_batchnorm'],
                                       device=self.device)  # type: Encoder_1D_Sigma
        else:
            encoder = Encoder_2D_Sigma(x_dim=self.x_dim,
                                       z_dim=self.z_dim,
                                       batch_norm=kwargs['encoder_batchnorm'],
                                       device=self.device)  # type: Encoder_2D_Sigma
        return encoder

    def encode(self, x):
        return self.encoder(x)

    def target_encode(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def next_z(self, z_t, a_t):
        with torch.no_grad():
            mu_tp1, log_sigma_tp1 = self.world_model(z_t, a_t)
        return mu_tp1, log_sigma_tp1

    def next_z_from_x(self, x_t, a_t):
        with torch.no_grad():
            mu_t, log_sigma_t = self.encode(x_t)
            mu_tp1, log_sigma_tp1 = self.world_model(mu_t, a_t)
        return mu_tp1, log_sigma_tp1


class DeterministicEncodedFM(nn.Module):

    class HingeLoss(torch.nn.Module):

        def __init__(self, hinge: float):
            super().__init__()

            self.hinge = hinge
            self.d = nn.MSELoss(reduction='none')

        def forward(self, output, neg_samples):
            # type: (torch.tensor, torch.tensor) -> torch.tensor
            assert output.shape == neg_samples.shape[:-1]
            output = output.unsqueeze(2).repeat((1, 1, neg_samples.shape[-1]))
            hinge_loss = -self.d(output, neg_samples) + self.hinge
            hinge_loss[hinge_loss < 0] = 0
            hinge_loss = hinge_loss.sum(dim=1)
            hinge_loss = hinge_loss.mean(dim=1)
            assert len(hinge_loss.shape) == 1
            return hinge_loss

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.encoder = self.create_encoder(len(x_dim), **kwargs)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.tau = kwargs['wm_tau']
        self.neg_samples = kwargs['neg_samples']
        self.world_model = ForwardModel(x_dim=self.encoder.get_z_dim(),
                                        a_dim=self.a_dim,
                                        device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.loss_func_neg_sampling = self.HingeLoss(kwargs['hinge_value']).to(device)
        self.steps = 0

    def forward(self, x_t, a_t, x_tp1, experiences):
        z_t = self.encoder(x_t)
        z_diff = self.world_model(z_t, a_t)
        z_tp1 = self.encoder(x_tp1)
        # z_tp1 = self.target_encode(x_tp1)
        loss_trans = self.loss_func_distance(z_t + z_diff, z_tp1).mean(dim=1)

        batch_size = x_t.shape[0]
        neg_samples = torch.from_numpy(experiences.sample_states(self.neg_samples * batch_size)).to(dtype=torch.float32)
        neg_samples_z = self.encoder(neg_samples)
        neg_samples_z = neg_samples_z.view((batch_size, -1, self.neg_samples))
        loss_ns = self.loss_func_neg_sampling(z_t, neg_samples_z)

        self.steps += 1
        # self.update_target()
        return loss_trans, loss_ns

    def update_target(self):
        if self.soft_target:
            self.soft_target_update(self.tau)
        else:
            if self.steps % self.target_steps == 0:
                self.target_encoder = copy.deepcopy(self.encoder)

    def soft_target_update(self, tau):
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            encoder = Encoder_1D(x_dim=self.x_dim,
                                 z_dim=self.z_dim,
                                 batch_norm=kwargs['encoder_batchnorm'],
                                 device=self.device)  # type: Encoder_1D
        else:
            raise NotImplementedError
        return encoder

    def encode(self, x):
        return self.encoder(x)

    def target_encode(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def next_z(self, z_t, a_t):
        with torch.no_grad():
            mu_tp1, log_sigma_tp1 = self.world_model(z_t, a_t)
        return mu_tp1, log_sigma_tp1

    def next_z_from_x(self, x_t, a_t):
        with torch.no_grad():
            mu_t, log_sigma_t = self.encode(x_t)
            mu_tp1, log_sigma_tp1 = self.world_model(mu_t, a_t)
        return mu_tp1, log_sigma_tp1


class EncodedWorldModel:
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.device = device
        print('Observation space:', self.x_dim)
        if kwargs['stochastic_latent']:
            self.model = StochasticEncodedFM(x_dim, a_dim, device=self.device, **kwargs)  # type: StochasticEncodedFM
        else:
            self.model = DeterministicEncodedFM(x_dim, a_dim, device=self.device, **kwargs)  # type: DeterministicEncodedFM
        self.optimizer_wm = kwargs['wm_optimizer'](self.model.parameters(), lr=kwargs['wm_lr'])
        self.decoder, self.loss_func_d, self.optimizer_d = None, None, None
        if len(x_dim) != 1:
            self.decoder = Decoder_2D(z_dim=self.model.z_dim,
                                      x_dim=x_dim,
                                      device=self.device)  # type: Decoder_2D

            self.loss_func_d = nn.MSELoss().to(self.device)
            self.optimizer_d = torch.optim.Adam(self.decoder.parameters())
        self.losses = {'wm': [], 'wm_trans': [], 'wm_ns': [], 'decoder': []}

    def forward(self, x_t, a_t, x_tp1, memories):
        x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[1:]) == self.x_dim
        with torch.no_grad():
            loss_trans, loss_ns = self.model.forward(x_t, a_t, x_tp1, memories)
        return loss_trans

    def train(self, x_t, a_t, x_tp1, memories):
        assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[1:]) == self.x_dim

        self.model.zero_grad()
        loss_trans, loss_ns = self.model.forward(x_t, a_t, x_tp1, memories)
        loss = torch.mean(loss_trans + loss_ns)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        self.losses['wm'].append(loss.item())
        self.losses['wm_trans'].append(loss_trans.mean().item())
        self.losses['wm_ns'].append(loss_ns.mean().item())
        return loss_trans.detach()

    def train_d(self, x_t):
        x_t = x_t.to(self.device)
        self.decoder.zero_grad()
        mu_t, log_sigma_t = self.encode(x_t)
        x_t_prime = self.decoder(mu_t.detach())
        loss_d = self.loss_func_d(x_t_prime, x_t)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
        self.optimizer_d.step()
        self.losses['decoder'].append(loss_d.item())

    def encode(self, x):
        with torch.no_grad():
            mu, sigma = self.model.encode(x)
        return mu.detach(), sigma.detach()

    def decode(self, z):
        with torch.no_grad():
            return self.decoder(z).detach()

    def next_z(self, mu_t, a_t):
        mu_tp1, log_sigma_tp1 = self.model.next_z(mu_t, a_t)
        return mu_tp1.detach(), log_sigma_tp1.detach()

    def next_z_from_x(self, x_t, a_t):
        mu_tp1, log_sigma_tp1 = self.model.next_z_from_x(x_t, a_t)
        return mu_tp1.detach(), log_sigma_tp1.detach()

    def predict_next_obs(self, x_t, a_t):
        with torch.no_grad():
            mu_tp1, _ = self.model.next_z_from_x(x_t, a_t)
            x_tp1 = self.decode(mu_tp1).detach()
            return x_tp1

    def get_losses(self):
        return self.losses

    def save(self, path='saved_objects/wm_encoder.pt'):
        torch.save({'model': self.model,
                    'optimizer_wm': self.optimizer_wm,
                    'losses': self.losses,
                    'decoder': self.decoder,
                    'optimizer_d': self.optimizer_d,
                    }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer_wm = checkpoint['optimizer_wm']
        self.decoder = checkpoint['decoder']
        self.optimizer_d = checkpoint['optimizer_d']
        self.losses['world_model'] = checkpoint['losses']


class WorldModelNoEncoder:
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.device = device
        print('Observation space:', self.x_dim)
        self.model = ForwardModel(x_dim, a_dim, device=self.device)  # type: ForwardModel
        self.optimizer = kwargs['wm_optimizer'](self.model.parameters(), lr=kwargs['wm_lr'])
        self.loss_func = torch.nn.SmoothL1Loss(reduction='none').to(self.device)
        self.losses = {'wm': [], 'decoder': []}

    def train(self, x_t, a_t, x_tp1):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[1:]) == self.x_dim

        self.model.zero_grad()
        x_tp1_prime = self.model(x_t, a_t)
        loss_vector = self.loss_func(x_tp1_prime, x_tp1).sum(dim=1)
        loss = loss_vector.mean()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.losses['wm'].append(loss.item())
        return loss_vector.detach()

    def next(self, mu_t, a_t):
        mu_tp1, log_sigma_tp1 = self.model.next_z(mu_t, a_t)
        return mu_tp1.detach(), log_sigma_tp1.detach()

    def get_losses(self):
        return self.losses

    def save(self, path='saved_objects/wm_encoder.pt'):
        torch.save({'model': self.model,
                    'optimizer_wm': self.optimizer,
                    'losses': self.losses,
                    }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer = checkpoint['optimizer_wm']
        self.losses['world_model'] = checkpoint['losses']
