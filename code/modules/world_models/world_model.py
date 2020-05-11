import torch
import torch.nn as nn
from modules.world_models.forward_model import ForwardModel_SigmaInputOutput, ForwardModel_SigmaOutputOnly, ForwardModel
from modules.encoders.learned_encoders import Encoder_2D_Sigma, Encoder_1D_Sigma, Encoder_1D
from modules.encoders.random_encoder import RandomEncoder_1D, RandomEncoder_2D
from modules.encoders.vae import VAE
from modules.decoders.decoder import Decoder_2D, Decoder_2D_conv
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


class VAEFM(nn.Module):

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.vae = VAE(x_dim, device=device, **kwargs)  # type: VAE
        self.forward_model = ForwardModel(x_dim=kwargs['z_dim'],
                                          a_dim=self.a_dim,
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.steps = 0

    def forward(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> [torch.Tensor, list, dict]
        """
        Forward pass of the world model. Returns processed intrinsic reward and keyworded
        loss components alongside the loss. Keyworded loss components are meant for bookkeeping
        and should be given as floats.
        """
        vae_loss, *_ = self.vae(x_t)
        z_t = self.vae.encode(x_t).detach()
        z_tp1= self.vae.encode(x_tp1).detach()
        z_diff = self.forward_model(z_t, a_t)
        assert not z_t.requires_grad
        assert not z_tp1.requires_grad
        loss_wm_vector = self.loss_func_distance(z_t + z_diff, z_tp1).mean(dim=1)
        loss_wm = loss_wm_vector.mean(dim=0)
        self.steps += 1
        loss = vae_loss + loss_wm
        loss_dict = {'wm_loss': loss.detach().mean().item(),
                     'wm_trans_loss': loss_wm.detach().mean().item(),
                     'wm_vae_loss': vae_loss.detach().mean().item()}
        return loss_wm_vector.detach(), loss, loss_dict

    def encode(self, x):
        with torch.no_grad():
            return self.vae.encode(x)

    def next_z_from_z(self, z_t, a_t):
        with torch.no_grad():
            return self.forward_model(z_t, a_t)

    def next(self, x_t, a_t):
        with torch.no_grad():
            mu_t = self.vae.encode(x_t)
            mu_tp1 = mu_t + self.forward_model(mu_t, a_t)
        return mu_tp1


class DeterministicCRandomEncodedFM(nn.Module):

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.encoder = self.create_encoder(len(x_dim), **kwargs)
        self.forward_model = ForwardModel(x_dim=self.encoder.get_z_dim(),
                                          a_dim=self.a_dim,
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.steps = 0

    def forward(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> [torch.Tensor, list, dict]
        """
        Forward pass of the world model. Returns processed intrinsic reward and keyworded
        loss components alongside the loss. Keyworded loss components are meant for bookkeeping
        and should be given as floats.
        """
        z_t = self.encoder(x_t)
        z_diff = self.forward_model(z_t, a_t)
        z_tp1 = self.encoder(x_tp1)
        assert not z_t.requires_grad
        assert not z_tp1.requires_grad
        loss_vector = self.loss_func_distance(z_t + z_diff, z_tp1).mean(dim=1)
        loss = loss_vector.mean(dim=0)
        # loss = self.loss_func_distance(z_diff, z_tp1.detach()).mean(dim=1)
        self.steps += 1
        return loss_vector.detach(), loss, {'wm_loss': loss.detach().item()}

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return RandomEncoder_1D(x_dim=self.x_dim,
                                    z_dim=self.z_dim,
                                    batch_norm=kwargs['encoder_batchnorm'],
                                    device=self.device)  # type: RandomEncoder_1D
        else:
            return RandomEncoder_2D(x_dim=self.x_dim,
                                    conv_layers=kwargs['conv_layers'],
                                    z_dim=self.z_dim,
                                    device=self.device)  # type: RandomEncoder_2D

    def encode(self, x):
        return self.encoder(x)

    def next_z_from_z(self, z_t, a_t):
        with torch.no_grad():
            return self.forward_model(z_t, a_t)

    def next(self, x_t, a_t):
        with torch.no_grad():
            z_t = self.encode(x_t)
            z_tp1 = z_t + self.forward_model(z_t, a_t)
        return z_tp1


class DeterministicContrastiveEncodedFM(nn.Module):
    class HingeLoss(torch.nn.Module):

        def __init__(self, hinge: float):
            super().__init__()

            self.hinge = hinge

        def forward(self, output, neg_samples):
            # type: (torch.tensor, torch.tensor) -> torch.tensor
            """
            Return the hinge loss between the output vector and its negative sample encodings.
            Expected dimensions of output is (batch, zdim) which gets repeated into(batch, zdim, negsamples).
            Negative sample encodings have expected sahpe (batch*negsamples, zdim).
            Implementation from https://github.com/tkipf/c-swm/blob/master/modules.py.
            """
            assert output.shape == neg_samples.shape[:-1]
            output = output.unsqueeze(2).repeat((1, 1, neg_samples.shape[-1]))
            diff = output - neg_samples
            energy = diff.pow(2).sum(dim=1).sqrt()
            energy = energy.mean(dim=1)
            hinge_loss = -energy + self.hinge
            hinge_loss = hinge_loss.clamp(min=0.0)
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
        self.target_encoder_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.target_encoder = None
        if self.soft_target or self.target_encoder_steps != 0:
            self.target_encoder = copy.deepcopy(self.encoder).to(device)
        self.tau = kwargs['wm_tau']
        self.neg_samples = kwargs['neg_samples']
        self.forward_model = ForwardModel(x_dim=self.encoder.get_z_dim(),
                                          a_dim=self.a_dim,
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.loss_func_neg_sampling = self.HingeLoss(kwargs['hinge_value']).to(device)
        self.train_steps = 0

    def forward(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> [torch.Tensor, list, dict]
        z_t = self.encoder(x_t)
        z_diff = self.forward_model(z_t, a_t)
        if self.target_encoder is not None:
            with torch.no_grad():
                z_tp1 = self.target_encoder(x_tp1)
        else:
            z_tp1 = self.encoder(x_tp1)
        loss_trans = self.loss_func_distance(z_t + z_diff, z_tp1).mean(dim=1)

        batch_size = x_t.shape[0]
        if self.neg_samples > 0:
            neg_samples = torch.from_numpy(kwargs['memories'].sample_states(self.neg_samples * batch_size)).to(
                dtype=torch.float32)

            neg_samples_z = self.encoder(neg_samples)

            neg_samples_z = neg_samples_z.view((batch_size, -1, self.neg_samples))
            loss_ns = self.loss_func_neg_sampling(z_t, neg_samples_z)
        else:
            loss_ns = torch.zeros((batch_size,))
        loss = (loss_trans + loss_ns).mean()
        self.train_steps += 1
        self.update_target_encoder()
        loss_dict = {'wm_loss': loss.detach().item(),
                     'wm_trans_loss': loss_trans.detach().mean().item(),
                     'wm_ns_loss': loss_ns.detach().mean().item()}
        return loss_trans.detach(), loss, loss_dict

    def update_target_encoder(self):
        if self.target_encoder is not None:
            if self.soft_target:
                for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.01) + param.data * 0.01)
            else:
                assert self.target_encoder_steps != 0
                if self.train_steps % self.target_encoder_steps == 0:
                    self.target_encoder = copy.deepcopy(self.encoder)
        else:
            assert not self.soft_target or self.target_encoder_steps == 0

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return Encoder_1D(x_dim=self.x_dim,
                              z_dim=self.z_dim,
                              batch_norm=kwargs['encoder_batchnorm'],
                              device=self.device)  # type: Encoder_1D
        else:
            raise NotImplementedError

    def encode(self, x):
        return self.encoder(x)

    def target_encode(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def next_z_from_z(self, z_t, a_t):
        with torch.no_grad():
            return self.forward_model(z_t, a_t)

    def next(self, x_t, a_t):
        with torch.no_grad():
            z_t = self.encode(x_t)
            z_diff = self.forward_model(z_t, a_t)
            return z_t + z_diff


class DeterministicInvDynFeatFM(nn.Module):
    class InverseModel(torch.nn.Module):
        def __init__(self, phi_dim, a_dim, hidden_dim=(256,), batch_norm=False, device='cpu'):
            # type: (tuple, tuple, tuple, bool, str) -> None
            """"
            Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
            Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
            """
            super().__init__()
            self.phi_dim = phi_dim
            self.a_dim = a_dim
            self.device = device
            self.layers = []
            h_dim_prev = self.phi_dim[0] * 2
            for h_dim in hidden_dim:
                self.layers.append(nn.Linear(h_dim_prev, h_dim))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(h_dim))
                self.layers.append(nn.ReLU())
                h_dim_prev = h_dim
            self.layers.append(nn.Linear(hidden_dim[-1], self.a_dim[0]))
            self.layers.append(nn.Softmax())
            self.model = nn.Sequential(*self.layers).to(self.device)

        def forward(self, phi_t, phi_tp1):
            # type: (torch.tensor, torch.tensor) -> torch.tensor
            assert len(phi_t.shape) > 1, 'Do not forget batch dim'
            x = torch.cat((phi_t, phi_tp1), dim=1)
            assert x.shape[0] == phi_t.shape[0]
            assert len(x.shape) == len(phi_t.shape)
            a_pred = self.model(x)
            assert a_pred.shape[1:] == self.a_dim
            return a_pred

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.encoder = self.create_encoder(len(x_dim), **kwargs)
        self.target_encoder_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.target_encoder = None
        if self.soft_target or self.target_encoder_steps != 0:
            self.target_encoder = copy.deepcopy(self.encoder).to(device)
        self.tau = kwargs['wm_tau']
        self.inverse_model = self.InverseModel(self.encoder.get_z_dim(), self.a_dim, device=device)
        self.forward_model = ForwardModel(x_dim=self.encoder.get_z_dim(),
                                          a_dim=self.a_dim,
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.optimizer_wm = kwargs['wm_optimizer'](self.inverse_model.parameters(), lr=kwargs['wm_lr'])
        self.loss_func_inverse = nn.BCELoss().to(device)
        self.train_steps = 0

    def forward(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> [torch.Tensor, torch.Tensor, dict]
        phi_t = self.encoder(x_t)
        phi_tp1 = self.encoder(x_tp1)
        a_t_pred = self.inverse_model(phi_t, phi_tp1)
        inverse_loss = self.loss_func_inverse(a_t_pred, self.action_index_to_onehot(a_t))
        phi_diff = self.forward_model(phi_t.detach(), a_t)
        if self.target_encoder is not None:
            with torch.no_grad():
                phi_t = self.target_encoder(x_t)
                phi_tp1 = self.target_encoder(x_tp1)
        loss_trans = self.loss_func_distance(phi_t.detach() + phi_diff, phi_tp1.detach()).mean(dim=1)
        loss_trans_mean = loss_trans.mean()
        self.train_steps += 1
        self.update_target_encoder()
        loss = loss_trans_mean + inverse_loss
        loss_dict = {'wm_loss': loss.detach().item(),
                     'wm_trans_loss': loss_trans_mean.detach().item(),
                     'wm_inv_loss': inverse_loss.detach().item()}
        return loss_trans.detach(), loss, loss_dict

    def action_index_to_onehot(self, a_i):
        # type: (torch.Tensor) -> torch.Tensor
        """"
        Given an action's index, return a one-hot vector where the action in question has the value 1.
        """
        assert type(a_i) == torch.Tensor
        batch_size = a_i.shape[0]
        # Add batch dimension to 1D tensor
        if len(tuple(a_i.shape)) == 1:
            if batch_size == self.a_dim:  # If there is no batch dim and entire vector belongs to 1 single action
                a_i = a_i.unsqueeze(0)
            else:
                a_i = a_i.unsqueeze(1)
        # Create 1-hot
        one_hot = torch.zeros((batch_size, self.a_dim[0]), device=self.device, dtype=torch.float32)
        batch_dim = torch.arange(batch_size, device=self.device)
        one_hot[batch_dim, a_i[:, 0].to(dtype=torch.long)] += 1.0  # Place ones into taken action indices
        return one_hot

    def update_target_encoder(self):
        if self.target_encoder is not None:
            if self.soft_target:
                for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.01) + param.data * 0.01)
            else:
                assert self.target_encoder_steps != 0
                if self.train_steps % self.target_encoder_steps == 0:
                    self.target_encoder = copy.deepcopy(self.encoder)
        else:
            assert not self.soft_target or self.target_encoder_steps == 0

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return Encoder_1D(x_dim=self.x_dim,
                              z_dim=self.z_dim,
                              batch_norm=kwargs['encoder_batchnorm'],
                              device=self.device)  # type: Encoder_1D
        else:
            raise NotImplementedError

    def encode(self, x):
        return self.encoder(x)

    def target_encode(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def next_z_from_z(self, z_t, a_t):
        with torch.no_grad():
            return self.forward_model(z_t, a_t)

    def next(self, x_t, a_t):
        with torch.no_grad():
            z_t = self.encode(x_t)
            return self.forward_model(z_t, a_t)


class EncodedWorldModel:
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.device = device
        print('Observation space:', self.x_dim)
        assert kwargs['encoder_type'] in {'random', 'cont', 'vae', 'idf'}, 'Unknown encoder type.'
        self.enc_is_vae = False
        if kwargs['encoder_type'] == 'cont':
            if kwargs['stochastic_latent']:
                self.model = StochasticEncodedFM(x_dim, a_dim, device=self.device,
                                                 **kwargs)  # type: StochasticEncodedFM
            else:
                self.model = DeterministicContrastiveEncodedFM(x_dim, a_dim, device=self.device,
                                                               **kwargs)  # type: DeterministicContrastiveEncodedFM
        elif kwargs['encoder_type'] == 'random':
            self.model = DeterministicCRandomEncodedFM(x_dim, a_dim, device=self.device,
                                                       **kwargs)  # type: DeterministicCRandomEncodedFM
        elif kwargs['encoder_type'] == 'idf':
            self.model = DeterministicInvDynFeatFM(x_dim, a_dim, device=self.device,
                                                   **kwargs)  # type: DeterministicInvDynFeatFM
        elif kwargs['encoder_type'] == 'vae':
            self.model = VAEFM(x_dim, a_dim, device=self.device, **kwargs)  # type: VAEFM
            self.enc_is_vae = True
        else:
            raise NotImplementedError
        self.optimizer_wm = torch.optim.Adam(self.model.parameters(), lr=kwargs['wm_lr'])
        self.decoder, self.loss_func_d, self.optimizer_d = None, None, None
        if kwargs['decoder']:
            if len(x_dim) != 1 and not self.enc_is_vae:
                self.decoder = Decoder_2D_conv(z_dim=self.model.z_dim,
                                               x_dim=x_dim,
                                               device=self.device)  # type: Decoder_2D_conv

                self.loss_func_d = nn.BCELoss().to(self.device)
                self.optimizer_d = torch.optim.Adam(self.decoder.parameters())
            elif self.enc_is_vae:
                self.decoder = self.model.vae.decoder
        self.losses = {'wm_loss': [], 'wm_trans_loss': [], 'wm_ns_loss': [],
                       'wm_inv_loss': [], 'wm_vae_loss': [], 'decoder_loss': []}

    def forward(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> torch.Tensor
        # x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        # assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[-3:]) == self.x_dim, f'Received: {tuple(x_t.shape[1:])} {self.x_dim}'
        with torch.no_grad():
            intr_reward, *_ = self.model.forward(x_t, a_t, x_tp1, **kwargs)
        return intr_reward

    def train(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> torch.Tensor
        # assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[1:]) == self.x_dim, f'Received: {tuple(x_t.shape[1:])} and {self.x_dim}'
        self.model.zero_grad()
        intr_reward, loss, loss_items = self.model.forward(x_t, a_t, x_tp1, **kwargs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        for key in loss_items:
            self.losses[key].append(loss_items[key])
        return intr_reward

    def train_d(self, x_t: torch.Tensor):
        x_t = x_t.to(self.device)
        if self.enc_is_vae:
            return
        self.decoder.zero_grad()
        z_t = self.encode(x_t)
        x_t_prime = self.decoder.forward(z_t)
        loss_d = self.loss_func_d(x_t_prime, x_t)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
        self.optimizer_d.step()
        self.losses['decoder_loss'].append(loss_d.item())

    def encode(self, x):
        with torch.no_grad():
            return self.model.encode(x).detach()

    def decode(self, z):
        with torch.no_grad():
            return self.decoder(z).detach()

    def next_z_from_z(self, z_t, a_t):
        return self.model.next_z_from_z(z_t, a_t).detach()

    def next(self, x_t, a_t):
        return self.model.next(x_t, a_t).detach()

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs['wm_lr'])
        # self.loss_func = torch.nn.MSELoss(reduction='none').to(self.device)
        self.loss_func = torch.nn.SmoothL1Loss(reduction='none').to(self.device)
        # self.loss_func = torch.nn.BCELoss(reduction='none').to(self.device)
        self.losses = {'wm_loss': [], 'decoder': []}

    def forward(self, x_t, a_t, x_tp1):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        with torch.no_grad():
            x_tp1_prime = self.model(x_t, a_t)
            # x_tp1_prime = torch.nn.functional.softmax(x_tp1_prime, dim=1)
            loss_vector = self.loss_func(x_tp1_prime, x_tp1).mean(dim=1)
        return loss_vector.detach()

    def train(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> torch.Tensor
        self.model.zero_grad()
        x_tp1_prime = self.model(x_t, a_t)
        # x_tp1_prime = torch.nn.functional.softmax(x_tp1_prime, dim=1)
        loss_vector = self.loss_func(x_tp1_prime, x_tp1).mean(dim=1)
        loss = loss_vector.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.losses['wm_loss'].append(loss.item())
        return loss_vector.detach()

    def next(self, x_t, a_t):
        with torch.no_grad():
            x_tp1 = self.model(x_t, a_t)
            x_tp1 = torch.nn.functional.softmax(x_tp1, dim=1)
        return x_tp1

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
