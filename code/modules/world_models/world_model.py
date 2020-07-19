import torch
import torch.nn as nn
import numpy as np
from modules.world_models.forward_model import ForwardModel
from modules.encoders.learned_encoders import Encoder_1D, Encoder_2D
from modules.encoders.random_encoder import RandomEncoder_1D, RandomEncoder_2D
from modules.encoders.vae import VAE
from modules.decoders.decoder import Decoder_2D, Decoder_2D_conv
import copy
import os


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
                                          hidden_dim=kwargs['wm_h_dim'],
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.train_steps = 0

    def forward(self, x_t, a_t, x_tp1, eval=False, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> [torch.Tensor, list, dict]
        """
        Forward pass of the world model. Returns processed intrinsic reward and keyworded
        loss components alongside the loss. Keyworded loss components are meant for bookkeeping
        and should be given as floats.
        """
        # Section necessary for training and eval (Calculate batch-wise translation error in latent space)
        z_t = self.vae.encode(x_t)
        z_tp1 = self.vae.encode(x_tp1)
        z_diff = self.forward_model(z_t, a_t)
        assert not z_t.requires_grad
        assert not z_tp1.requires_grad
        loss_wm_vector = self.loss_func_distance(z_t + z_diff, z_tp1).sum(dim=1)
        loss, loss_dict = None, None
        # Section necessary only for training (Calculate VAE loss and overall loss)
        if not eval:
            vae_loss, *_ = self.vae(x_t)
            loss_wm = loss_wm_vector.mean()
            self.train_steps += 1
            loss = vae_loss + loss_wm
            loss_dict = {'wm_loss': loss.detach().mean().item(),
                         'wm_trans_loss': loss_wm.detach().item(),
                         'wm_vae_loss': vae_loss.detach().item()}
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

    def save_encoder(self, path):
        torch.save(self.vae.state_dict(), path)

    def load_encoder(self, path):
        self.vae.load_state_dict(torch.load(path))


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
                                          hidden_dim=kwargs['wm_h_dim'],
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.trains_steps = 0

    def forward(self, x_t, a_t, x_tp1, eval=False, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> [torch.Tensor, list, dict]
        """
        Forward pass of the world model. Returns processed intrinsic reward and keyworded
        loss components alongside the loss. Keyworded loss components are meant for bookkeeping
        and should be given as floats.
        """
        # Section necessary for training and eval (Calculate batch-wise translation error in latent space)
        with torch.no_grad():
            z_t = self.encoder(x_t)
            z_tp1 = self.encoder(x_tp1)
        z_diff = self.forward_model(z_t, a_t)
        assert not z_t.requires_grad
        assert not z_tp1.requires_grad
        loss_vector = self.loss_func_distance(z_t + z_diff, z_tp1).sum(dim=1)
        loss, loss_dict = None, None
        # Section necessary only for training (Calculate overall loss)
        if not eval:
            loss = loss_vector.mean()
            # loss = self.loss_func_distance(z_diff, z_tp1.detach()).mean(dim=1)
            self.trains_steps += 1
            loss_dict = {'wm_loss': loss.detach().item()}
        return loss_vector.detach(), loss, loss_dict

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

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load_encoder(self, path):
        self.encoder.load_state_dict(torch.load(path))


class DeterministicContrastiveEncodedFM(nn.Module):
    class HingeLoss(torch.nn.Module):

        def __init__(self, hinge: float):
            super().__init__()
            self.hinge = hinge

        def apply_tensor_constraints(self, x, required_dim, name=''):
            # type: (torch.Tensor, tuple, str) -> torch.Tensor
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(dtype=torch.float32)
            assert isinstance(x, torch.Tensor), type(x)
            if len(tuple(x.shape)) == 1:  # Add batch dimension to 1D tensor
                if x.shape[0] == required_dim[0]:
                    x = x.unsqueeze(0)
                else:
                    x = x.unsqueeze(1)
            x = x.to(self.device)
            return x

        def forward(self, output, neg_samples):
            # type: (torch.tensor, torch.tensor) -> torch.tensor
            """
            Return the hinge loss between the output vector and its negative sample encodings.
            Expected dimensions of output is (batch, zdim) which gets repeated into(batch, negsamples, zdim).
            Negative sample encodings have expected shape (batch*negsamples, zdim).
            Implementation from https://github.com/tkipf/c-swm/blob/master/modules.py.
            """
            assert output.shape[0] == neg_samples.shape[0] and output.shape[1] == neg_samples.shape[2], \
                f'Received: {tuple(output.shape)} {neg_samples.shape}'
            output = output.unsqueeze(1).repeat((1, neg_samples.shape[1], 1))
            diff = output - neg_samples
            energy = diff.pow(2).sum(dim=2)
            # energy = energy.mean(dim=1)
            hinge_loss = -energy + self.hinge
            hinge_loss = hinge_loss.clamp(min=0.0)
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
        self.target_encoder_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.target_encoder = None
        if self.soft_target or self.target_encoder_steps != 0:
            self.target_encoder = copy.deepcopy(self.encoder).to(device)
        self.tau = kwargs['wm_tau']
        self.neg_samples = kwargs['neg_samples']
        self.forward_model = ForwardModel(x_dim=self.encoder.get_z_dim(),
                                          a_dim=self.a_dim,
                                          hidden_dim=kwargs['wm_h_dim'],
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.loss_func_neg_sampling = self.HingeLoss(kwargs['hinge_value']).to(device)
        self.train_steps = 0

    def forward(self, x_t, a_t, x_tp1, eval=False, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> [torch.Tensor, list, dict]
        if len(tuple(x_t.shape)) == 1:  # Add batch dimension to 1D tensor
            x_t = x_t.unsqueeze(0)
        if len(tuple(x_tp1.shape)) == 1:  # Add batch dimension to 1D tensor
            x_tp1 = x_tp1.unsqueeze(0)
        # Section necessary for training and eval (Calculate batch-wise translation error in latent space)
        z_t = self.encoder(x_t)
        z_diff = self.forward_model(z_t, a_t)
        if self.target_encoder is not None:
            with torch.no_grad():
                z_tp1 = self.target_encoder(x_tp1)
        else:
            z_tp1 = self.encoder(x_tp1)
        loss_trans = self.loss_func_distance(z_t + z_diff, z_tp1).sum(dim=1)
        # Section necessary only for training (Calculate negative sampling error and overall loss)
        loss_ns, loss, loss_dict = None, None, None
        if not eval:
            if self.neg_samples > 0:
                if not isinstance(kwargs['memories'], torch.Tensor):
                    neg_samples = torch.from_numpy(kwargs['memories'].sample_states(self.neg_samples)).to(
                        dtype=torch.float32, device=self.device)
                    # neg_samples = kwargs['memories'].sample_states(self.neg_samples)
                else:
                    neg_samples = kwargs['memories']
                loss_ns = self.calculate_contrastive_loss(neg_samples, z_t=z_t, pos_examples_z=z_tp1)
                # loss_ns = self.calculate_neg_example_loss(neg_samples, z_t=z_t)
                loss = (loss_trans + loss_ns).mean()
            else:
                loss = loss_trans.mean()
            self.train_steps += 1
            self.update_target_encoder()
            loss_dict = {'wm_loss': loss.detach().item(),
                         'wm_trans_loss': loss_trans.detach().mean().item(),
                         'wm_ns_loss': loss_ns.detach().mean().item()}
        return loss_trans.detach(), loss, loss_dict

    def calculate_neg_example_loss(self, neg_examples, x_t=None, z_t=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> [torch.Tensor]
        if z_t is None:
            assert x_t is not None, "Either x_t or z_t should be None."
            z_t = self.encoder(x_t)
        else:
            assert x_t is None, "Either x_t or z_t should be None."
        if len(tuple(z_t.shape)) == 1:  # Add batch dimension to 1D tensor
            z_t = z_t.unsqueeze(0)
        assert len(tuple(neg_examples.shape)) == 2

        neg_samples_z = self.encoder(neg_examples)
        neg_samples_z = neg_samples_z.unsqueeze(0)
        neg_samples_z = neg_samples_z.repeat((z_t.shape[0], 1, 1))
        return self.loss_func_neg_sampling(z_t, neg_samples_z).mean()

    def calculate_contrastive_loss(self, neg_examples, x_t=None, z_t=None, pos_examples=None, pos_examples_z=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> [torch.Tensor]
        """
        Negative and Positive examples are converted into a (batch, examples, zdim) structure in this function.
        """
        if z_t is None:
            assert x_t is not None, "Either x_t or z_t should be None."
            z_t = self.encoder(x_t)
        else:
            assert x_t is None, "Either x_t or z_t should be None."
        if len(tuple(z_t.shape)) == 1:  # Add batch dimension to 1D tensor
            z_t = z_t.unsqueeze(0)
        if len(tuple(neg_examples.shape)) == 1:  # Add batch dimension to 1D tensor
            neg_examples = neg_examples.unsqueeze(0)

        neg_samples_z = self.encoder(neg_examples)
        neg_samples_z = neg_samples_z.unsqueeze(0)
        neg_samples_z = neg_samples_z.repeat((z_t.shape[0], 1, 1))
        loss = self.loss_func_neg_sampling(z_t, neg_samples_z)
        # Positive examples below
        if pos_examples is not None:
            assert pos_examples_z is None, "Either x_t or z_t should be None."
            if len(tuple(pos_examples.shape)) == 1:  # Add batch dimension to 1D tensor
                pos_examples = pos_examples.unsqueeze(0)
            pos_examples_z = self.encoder(pos_examples)
        if pos_examples_z is not None:
            if len(tuple(pos_examples_z.shape)) == 1:  # Add batch dimension to 1D tensor
                pos_examples_z = pos_examples_z.unsqueeze(0)
            if len(tuple(pos_examples_z.shape)) == 2:
                if z_t.shape[0] == 1:
                    pos_examples_z = pos_examples_z.unsqueeze(0)
                else:
                    # This accepts 2D pos example tensors where every z_t has the same amount of pos examples
                    assert pos_examples_z.shape[0] % z_t.shape[0] == 0, \
                        "There should be an equal amount of pos examples per z."
                    pos_examples_z = pos_examples_z.view((z_t.shape[0], pos_examples_z.shape[0] // z_t.shape[0], -1))
            z_t_expanded = z_t.unsqueeze(1).repeat((1, pos_examples_z.shape[1], 1))
            pos_loss = (z_t_expanded - pos_examples_z).pow(2).sum(dim=2).clamp(min=self.loss_func_neg_sampling.hinge).mean(dim=1)
            loss += pos_loss
        return loss.mean()

    def forward_fm_only(self, x_t, a_t, x_tp1, eval=False):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool) -> [torch.Tensor, list, dict]
        """"
        Forward pass where gradients are only applied to the forward model.
        """
        if len(tuple(x_t.shape)) == 1:  # Add batch dimension to 1D tensor
            x_t = x_t.unsqueeze(0)
        if len(tuple(x_tp1.shape)) == 1:  # Add batch dimension to 1D tensor
            x_tp1 = x_tp1.unsqueeze(0)
        with torch.no_grad():
            z_t = self.encoder(x_t)
            if self.target_encoder is not None:
                z_tp1 = self.target_encoder(x_tp1)
            else:
                z_tp1 = self.encoder(x_tp1)
        z_diff = self.forward_model(z_t, a_t)
        loss_vector = self.loss_func_distance(z_t + z_diff, z_tp1).sum(dim=1)
        loss = loss_vector.mean()
        loss_dict = None
        if not eval:
            self.train_steps += 1
            self.update_target_encoder()
            loss_dict = {'wm_loss': loss.detach().item(),
                         'wm_trans_loss': loss.detach().mean().item()}
        return loss_vector.detach(), loss, loss_dict

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
            return Encoder_2D(x_dim=self.x_dim,
                              conv_layers=kwargs['conv_layers'],
                              z_dim=self.z_dim,
                              device=self.device)  # type: Encoder_2D

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

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load_encoder(self, path):
        self.encoder.load_state_dict(torch.load(path))


class DeterministicInvDynFeatFM(nn.Module):
    class InverseModel(torch.nn.Module):
        def __init__(self, phi_dim, a_dim, hidden_dim=(64,), batch_norm=False, device='cpu'):
            # type: (tuple, tuple, tuple, bool, str) -> None
            """"
            Network responsible for action prediction given s_t and s_tp1.
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
            self.layers.append(nn.Softmax(dim=1))
            self.model = nn.Sequential(*self.layers).to(self.device)

        def forward(self, phi_t, phi_tp1):
            # type: (torch.tensor, torch.tensor) -> torch.tensor
            # assert len(phi_t.shape) > 1, 'Do not forget batch dim'
            x = torch.cat((phi_t, phi_tp1), dim=1)
            # assert x.shape[0] == phi_t.shape[0]
            # assert len(x.shape) == len(phi_t.shape)
            a_pred = self.model(x)
            # assert a_pred.shape[1:] == self.a_dim
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
        self.inverse_model = self.InverseModel(self.encoder.get_z_dim(),
                                               self.a_dim,
                                               hidden_dim=kwargs['idf_inverse_hdim'],
                                               device=device)
        self.forward_model = ForwardModel(x_dim=self.encoder.get_z_dim(),
                                          a_dim=self.a_dim,
                                          hidden_dim=kwargs['wm_h_dim'],
                                          device=self.device)  # type: ForwardModel
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.loss_func_inverse = nn.BCELoss().to(device)
        self.train_steps = 0

    def forward(self, x_t, a_t, x_tp1, eval=False, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> [torch.Tensor, torch.Tensor, dict]
        # Section necessary for training and eval (Calculate batch-wise translation error in latent space)
        phi_t = self.encoder(x_t)
        phi_tp1 = self.encoder(x_tp1)
        if self.target_encoder is not None:
            with torch.no_grad():
                phi_t = self.target_encoder(x_t)
                phi_tp1 = self.target_encoder(x_tp1)
        phi_diff = self.forward_model(phi_t, a_t)
        loss_trans = self.loss_func_distance(phi_t + phi_diff, phi_tp1).sum(dim=1)
        loss, loss_dict = None, None
        # Section necessary only for training (Calculate inverse dynamics error and overall loss)
        if not eval:
            a_t_pred = self.inverse_model(phi_t, phi_tp1)
            inverse_loss = self.loss_func_inverse(a_t_pred, self.action_index_to_onehot(a_t))
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
            else:  # This should only happen in the case where there is a vector of actions and a_dim == 1
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
            return Encoder_2D(x_dim=self.x_dim,
                              conv_layers=kwargs['conv_layers'],
                              z_dim=self.z_dim,
                              device=self.device)  # type: Encoder_2D

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

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load_encoder(self, path):
        self.encoder.load_state_dict(torch.load(path))


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
        print('WM Architecture:')
        print(self.model)
        if kwargs['wm_opt'] == 'sgd':
            self.optimizer_wm = torch.optim.SGD(self.model.parameters(), lr=kwargs['wm_lr'])
        elif kwargs['wm_opt'] == 'adam':
            self.optimizer_wm = torch.optim.Adam(self.model.parameters(), lr=kwargs['wm_lr'])
        else:
            raise NameError('What optimizer??')
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
        self._its_a_gridworld_bois = kwargs['env_name'][:9] == 'GridWorld'
        self.state_wise_loss = {} if self._its_a_gridworld_bois else None
        self.state_wise_loss_diff = {} if self._its_a_gridworld_bois else None

    def forward(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> torch.Tensor
        assert tuple(x_t.shape[-3:]) == self.x_dim, f'Received: {tuple(x_t.shape[1:])} {self.x_dim}'
        with torch.no_grad():
            intr_reward, *_ = self.model.forward(x_t, a_t, x_tp1, eval=True, **kwargs)
        return intr_reward

    def train(self, x_t, a_t, x_tp1, store_loss=True, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> torch.Tensor
        if len(x_t.shape) == 1:
            x_t = x_t.unsqueeze(0)
        if len(x_tp1.shape) == 1:
            x_tp1 = x_tp1.unsqueeze(0)
        self.model.zero_grad()
        intr_reward, loss, loss_items = self.model.forward(x_t, a_t, x_tp1, **kwargs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        for key in loss_items:
            self.losses[key].append(loss_items[key])
        if self._its_a_gridworld_bois and store_loss and 'distance' in kwargs:
            key = str(x_t.cpu().numpy().tolist()) + str(a_t.cpu().numpy().tolist())
            if key in self.state_wise_loss_diff:
                self.state_wise_loss_diff[key]['list'].append(
                    abs(self.state_wise_loss[key]['list'][-1] - intr_reward.item()))
            new_d, *_ = self.model.forward(x_t, a_t, x_tp1, **kwargs)
            if key in self.state_wise_loss:
                self.state_wise_loss[key]['list'].append(new_d)
            else:
                self.state_wise_loss[key] = {'d': kwargs['distance'], 'list': [new_d]}
                self.state_wise_loss_diff[key] = {'d': kwargs['distance'], 'list': []}
        return intr_reward

    def train_contrastive_encoder(self, x_t, negative_examples, positive_examples=None):
        # type: (torch.Tensor, object, torch.Tensor) -> None
        if isinstance(negative_examples, np.ndarray):
            negative_examples = torch.from_numpy(negative_examples).to(dtype=torch.float32, device=self.device)
        if not isinstance(negative_examples, torch.Tensor):
            negative_examples = torch.from_numpy(negative_examples.sample_states(
                self.model.neg_samples)).to(dtype=torch.float32, device=self.device)
        self.model.zero_grad()
        loss = self.model.calculate_contrastive_loss(negative_examples, x_t=x_t, pos_examples=positive_examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        self.losses['wm_ns_loss'].append(loss.item())

    def train_contrastive_fm(self, x_t, a_t, x_tp1, store_loss=True, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> torch.Tensor
        self.model.zero_grad()
        intr_reward, loss, loss_items = self.model.forward_fm_only(x_t, a_t, x_tp1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        for key in loss_items:
            self.losses[key].append(loss_items[key])
        if self._its_a_gridworld_bois and store_loss:
            key = str(x_t.cpu().numpy().tolist()) + str(a_t.cpu().numpy().tolist())
            if key in self.state_wise_loss_diff:
                self.state_wise_loss_diff[key]['list'].append(
                    abs(self.state_wise_loss[key]['list'][-1] - intr_reward.item()))
            new_d, *_ = self.model.forward_fm_only(x_t, a_t, x_tp1)
            if key in self.state_wise_loss:
                self.state_wise_loss[key]['list'].append(new_d)
            else:
                self.state_wise_loss[key] = {'d': kwargs['distance'], 'list': [new_d]}
                self.state_wise_loss_diff[key] = {'d': kwargs['distance'], 'list': []}
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

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save({'model': self.model,
                    'optimizer_wm': self.optimizer_wm,
                    'losses': self.losses,
                    'decoder': self.decoder,
                    'optimizer_d': self.optimizer_d,
                    'state_wise_loss': self.state_wise_loss,
                    'state_wise_loss_diff': self.state_wise_loss_diff
                    }, folder_path + 'wm_items.pt')

    def save_encoder(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        self.model.save_encoder(folder_path + 'trained_encoder.pt')

    def load_encoder(self, path):
        self.model.load_encoder(path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer_wm = checkpoint['optimizer_wm']
        self.decoder = checkpoint['decoder']
        self.optimizer_d = checkpoint['optimizer_d']
        self.losses = checkpoint['losses']
        self.state_wise_loss = checkpoint['state_wise_loss']
        self.state_wise_loss_diff = checkpoint['state_wise_loss_diff']


class WorldModelContrastive:
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.device = device
        print('Observation space:', self.x_dim)
        assert kwargs['encoder_type'] == 'cont', 'Encoder type should be contrastive, boi..'
        self.model = DeterministicContrastiveEncodedFM(x_dim, a_dim, device=self.device,
                                                       **kwargs)  # type: DeterministicContrastiveEncodedFM
        print('WM Architecture:')
        print(self.model)
        if kwargs['wm_opt'] == 'sgd':
            self.optimizer_wm = torch.optim.SGD(self.model.parameters(), lr=kwargs['wm_lr'])
            self.optimizer_enc = torch.optim.SGD(self.model.encoder.parameters(), lr=kwargs['wm_enc_lr'])
        elif kwargs['wm_opt'] == 'adam':
            self.optimizer_wm = torch.optim.Adam(self.model.parameters(), lr=kwargs['wm_lr'])
            self.optimizer_enc = torch.optim.Adam(self.model.encoder.parameters(), lr=kwargs['wm_enc_lr'])
        else:
            raise NameError('What optimizer??')
        self.losses = {'wm_loss': [], 'wm_trans_loss': [], 'wm_ns_loss': []}
        self._its_a_gridworld_bois = kwargs['env_name'][:9] == 'GridWorld'
        self.state_wise_loss = {} if self._its_a_gridworld_bois else None
        self.state_wise_loss_diff = {} if self._its_a_gridworld_bois else None

    def forward(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> torch.Tensor
        assert tuple(x_t.shape[-3:]) == self.x_dim, f'Received: {tuple(x_t.shape[1:])} {self.x_dim}'
        with torch.no_grad():
            intr_reward, *_ = self.model.forward(x_t, a_t, x_tp1, eval=True, **kwargs)
        return intr_reward

    def train(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> torch.Tensor
        if len(x_t.shape) == 1:
            x_t = x_t.unsqueeze(0)
        if len(x_tp1.shape) == 1:
            x_tp1 = x_tp1.unsqueeze(0)
        # for i in range(10):
        #     self.train_contrastive_encoder(x_t, kwargs['memories'], positive_examples=x_tp1)
        self.train_contrastive_encoder(x_t, kwargs['memories'], positive_examples=x_tp1)
        # self.train_contrastive_encoder(x_t, kwargs['memories'], positive_examples=x_tp1)
        # self.train_contrastive_encoder(x_t, kwargs['memories'], positive_examples=x_tp1)
        # self.train_contrastive_encoder(x_t, kwargs['memories'], positive_examples=x_tp1)
        int_reward = self.train_contrastive_fm(x_t, a_t, x_tp1, **kwargs)
        # int_reward = self.train_contrastive_enc_and_fm(x_t, a_t, x_tp1, **kwargs)
        return int_reward

    def train_contrastive_encoder(self, x_t, negative_examples, positive_examples=None):
        # type: (torch.Tensor, object, torch.Tensor) -> None
        if isinstance(negative_examples, np.ndarray):
            negative_examples = torch.from_numpy(negative_examples).to(dtype=torch.float32, device=self.device)
        if not isinstance(negative_examples, torch.Tensor):
            negative_examples = torch.from_numpy(negative_examples.sample_states(
                self.model.neg_samples)).to(dtype=torch.float32, device=self.device)
        self.optimizer_enc.zero_grad()
        loss = self.model.calculate_contrastive_loss(negative_examples, x_t=x_t, pos_examples=positive_examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_enc.step()
        self.losses['wm_ns_loss'].append(loss.item())

    def train_contrastive_enc_and_fm(self, x_t, a_t, x_tp1, store_loss=True, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> torch.Tensor
        if len(x_t.shape) == 1:
            x_t = x_t.unsqueeze(0)
        if len(x_tp1.shape) == 1:
            x_tp1 = x_tp1.unsqueeze(0)
        self.model.zero_grad()
        intr_reward, loss, loss_items = self.model.forward(x_t, a_t, x_tp1, **kwargs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        for key in loss_items:
            self.losses[key].append(loss_items[key])
        if self._its_a_gridworld_bois and store_loss and 'distance' in kwargs:
            key = str(x_t.cpu().numpy().tolist()) + str(a_t.cpu().numpy().tolist())
            if key in self.state_wise_loss_diff:
                self.state_wise_loss_diff[key]['list'].append(
                    abs(self.state_wise_loss[key]['list'][-1] - intr_reward.item()))
            new_d, *_ = self.model.forward(x_t, a_t, x_tp1, **kwargs)
            if key in self.state_wise_loss:
                self.state_wise_loss[key]['list'].append(new_d)
            else:
                self.state_wise_loss[key] = {'d': kwargs['distance'], 'list': [new_d]}
                self.state_wise_loss_diff[key] = {'d': kwargs['distance'], 'list': []}
        return intr_reward

    def train_contrastive_fm(self, x_t, a_t, x_tp1, store_loss=True, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> torch.Tensor
        self.model.zero_grad()
        intr_reward, loss, loss_items = self.model.forward_fm_only(x_t, a_t, x_tp1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        for key in loss_items:
            self.losses[key].append(loss_items[key])
        if self._its_a_gridworld_bois and store_loss and 'distance' in kwargs:
            key = str(x_t.cpu().numpy().tolist()) + str(a_t.cpu().numpy().tolist())
            if key in self.state_wise_loss_diff:
                self.state_wise_loss_diff[key]['list'].append(
                    abs(self.state_wise_loss[key]['list'][-1] - intr_reward.item()))
            new_d, *_ = self.model.forward_fm_only(x_t, a_t, x_tp1)
            if key in self.state_wise_loss:
                self.state_wise_loss[key]['list'].append(new_d)
            else:
                self.state_wise_loss[key] = {'d': kwargs['distance'], 'list': [new_d]}
                self.state_wise_loss_diff[key] = {'d': kwargs['distance'], 'list': []}
        return intr_reward

    def encode(self, x):
        with torch.no_grad():
            return self.model.encode(x).detach()

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

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save({'model': self.model,
                    'optimizer_wm': self.optimizer_wm,
                    'optimizer_enc': self.optimizer_enc,
                    'losses': self.losses,
                    'state_wise_loss': self.state_wise_loss,
                    'state_wise_loss_diff': self.state_wise_loss_diff
                    }, folder_path + 'wm_items.pt')

    def save_encoder(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        self.model.save_encoder(folder_path + 'trained_encoder.pt')

    def load_encoder(self, path):
        self.model.load_encoder(path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer_wm = checkpoint['optimizer_wm']
        self.optimizer_enc = checkpoint['optimizer_enc']
        self.losses = checkpoint['losses']
        self.state_wise_loss = checkpoint['state_wise_loss']
        self.state_wise_loss_diff = checkpoint['state_wise_loss_diff']


class WorldModelNoEncoder:
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.device = device
        print('Observation space:', self.x_dim)
        self.model = ForwardModel(x_dim, a_dim, hidden_dim=kwargs['wm_h_dim'], device=self.device)  # type: ForwardModel
        print('WM Architecture:')
        print(self.model)
        if kwargs['wm_opt'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=kwargs['wm_lr'])
        elif kwargs['wm_opt'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs['wm_lr'])
        else:
            raise NameError('What optimizer??')
        self.loss_func = torch.nn.MSELoss(reduction='none').to(self.device)
        self.train_steps = 0
        self.losses = {'wm_loss': [], 'wm_pred_error': []}
        self._its_a_gridworld_bois = kwargs['env_name'][:9] == 'GridWorld'
        self.state_wise_loss = {} if self._its_a_gridworld_bois else None
        self.state_wise_loss_diff = {} if self._its_a_gridworld_bois else None

    def forward(self, x_t, a_t, x_tp1):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        with torch.no_grad():
            x_tp1_prime = self.model(x_t, a_t)
            loss_vector = self.loss_func(x_t + x_tp1_prime, x_tp1).sum(dim=1)
        return loss_vector.detach()

    def train(self, x_t, a_t, x_tp1, store_loss=True, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> torch.Tensor
        self.model.zero_grad()
        x_tp1_prime = self.model(x_t, a_t)
        loss_vector = self.loss_func(x_t + x_tp1_prime, x_tp1).sum(dim=1)
        intr_reward = loss_vector.detach()
        loss = loss_vector.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self._its_a_gridworld_bois and store_loss and 'distance' in kwargs:
            key = str(x_t.cpu().numpy().tolist()) + str(a_t.cpu().numpy().tolist())
            if key in self.state_wise_loss_diff:
                self.state_wise_loss_diff[key]['list'].append(
                    abs(self.state_wise_loss[key]['list'][-1] - intr_reward.item()))
            new_d = self.loss_func(self.model.forward(x_t, a_t), x_tp1).sum(dim=1).detach()
            if key in self.state_wise_loss:
                self.state_wise_loss[key]['list'].append(new_d.item())
            else:
                self.state_wise_loss[key] = {'d': kwargs['distance'], 'list': [new_d.item()]}
                self.state_wise_loss_diff[key] = {'d': kwargs['distance'], 'list': []}
        self.losses['wm_loss'].append(loss.item())
        self.losses['wm_pred_error'].append((x_tp1 - x_tp1_prime).abs().sum(dim=1).mean().item())
        self.train_steps += 1
        return intr_reward

    def next(self, x_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        device = x_t.device
        with torch.no_grad():
            x_tp1 = self.model(x_t, a_t)
        return x_tp1.to(device)

    def get_losses(self):
        return self.losses

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save({'model': self.model.cpu(),
                    'optimizer_wm': self.optimizer,
                    'losses': self.losses,
                    'state_wise_loss': self.state_wise_loss,
                    'state_wise_loss_diff': self.state_wise_loss_diff
                    }, folder_path + 'wm_items.pt')

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer = checkpoint['optimizer_wm']
        self.losses = checkpoint['losses']
        self.state_wise_loss = checkpoint['state_wise_loss']
        self.state_wise_loss_diff = checkpoint['state_wise_loss_diff']


class TabularWorldModel:
    def __init__(self, obs_dim, a_dim, lr=0.001, **kwargs):
        self.obs_dim = obs_dim
        self.a_dim = a_dim
        self.predictions = {}
        self.lr = lr
        self.train_steps = 0
        self.losses = {'wm_loss': []}
        self._its_a_gridworld_bois = kwargs['env_name'][:9] == 'GridWorld'
        self.state_wise_loss = {} if self._its_a_gridworld_bois else None
        self.state_wise_loss_diff = {} if self._its_a_gridworld_bois else None

    def train(self, s_t, a_t, s_tp1, store_loss=True, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> torch.Tensor
        assert len(s_t.shape) == 1
        assert len(a_t.shape) == 1
        if len(s_tp1.shape) == 2:
            s_tp1 = s_tp1.squeeze(0)
        assert len(s_tp1.shape) == 1
        s_t_key = str(s_t.cpu().numpy().tolist())
        a_t = a_t[0]
        if s_t_key not in self.predictions:
            self.predictions[s_t_key] = [torch.zeros(self.obs_dim) for _ in range(self.a_dim[0])]
        error = (s_tp1 - (s_t + self.predictions[s_t_key][a_t]))
        r_int = error.pow(2.0).sum()
        self.predictions[s_t_key][a_t] += self.lr * error
        if self._its_a_gridworld_bois and store_loss and 'distance' in kwargs:
            key = str(s_t.cpu().numpy().tolist()) + str(a_t.cpu().numpy().tolist())
            if key in self.state_wise_loss_diff:
                self.state_wise_loss_diff[key]['list'].append(abs(self.state_wise_loss[key]['list'][-1] - r_int.item()))
            new_d = (s_tp1 - (s_t + self.predictions[s_t_key][a_t])).pow(2.0).sum()
            if key in self.state_wise_loss:
                self.state_wise_loss[key]['list'].append(new_d.item())
            else:
                self.state_wise_loss[key] = {'d': kwargs['distance'], 'list': [new_d.item()]}
                self.state_wise_loss_diff[key] = {'d': kwargs['distance'], 'list': []}
        self.losses['wm_loss'].append(r_int)
        self.train_steps += 1
        return r_int

    def forward(self, s_t, a_t, s_tp1):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        assert len(s_t.shape) == 1
        assert len(a_t.shape) == 1
        if len(s_tp1.shape) == 2:
            s_tp1 = s_tp1.squeeze(0)
        assert len(s_tp1.shape) == 1
        s_t_key = str(s_t.tolist())
        a_t = a_t[0]
        if s_t_key not in self.predictions:
            error = s_tp1
        else:
            error = (s_tp1 - self.predictions[s_t_key][a_t])
        return error.pow(2.0).sum()

    def next(self, s_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        assert len(s_t.shape) == 1
        assert len(a_t.shape) == 1
        s_t_key = str(s_t.tolist())
        a_t = a_t[0]
        if s_t_key not in self.predictions:
            return torch.zeros(self.obs_dim)
        else:
            return self.predictions[s_t_key][a_t]

    def get_losses(self):
        return self.losses

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save({'losses': self.losses,
                    'state_wise_loss': self.state_wise_loss,
                    'state_wise_loss_diff': self.state_wise_loss_diff
                    }, folder_path + 'wm_items.pt')

    def load(self, path):
        checkpoint = torch.load(path)
        self.losses = checkpoint['losses']
        self.state_wise_loss = checkpoint['state_wise_loss']
        self.state_wise_loss_diff = checkpoint['state_wise_loss_diff']
