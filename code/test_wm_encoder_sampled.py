# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import torch.nn as nn
import gym
from modules.replay_buffers.replay_buffer import ReplayBuffer
from modules.world_models.forward_model import ForwardModel_SigmaOutputOnly
from utils.utils import resize_to_standard_dim_numpy, channel_first_numpy, INPUT_DIM
from modules.encoders.learned_encoders import Encoder_2D_Sigma
from modules.decoders.decoder import Decoder_2D

import copy
import pickle

torch.set_printoptions(edgeitems=10)


class Model(nn.Module):
    def __init__(self, x_dim, a_dim, z_dim=(40,), target_steps=500, tau=0.001, device='cpu'):
        # type: (tuple, tuple, tuple, int, float, str) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.tau = tau
        self.device = device
        self.encoder = Encoder_2D_Sigma(x_dim=self.x_dim,
                                        z_dim=self.z_dim,
                                        device=self.device)  # type: Encoder_2D_Sigma
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_steps = target_steps
        self.world_model = ForwardModel_SigmaOutputOnly(x_dim=self.encoder.get_z_dim(),
                                                        a_dim=self.a_dim,
                                                        vector_actions=False,
                                                        device=self.device)  # type: ForwardModel_SigmaOutputOnly

        # self.loss_func = torch.nn.SmoothL1Loss().to(self.device)
        self.loss_func = torch.distributions.kl.kl_divergence
        self.steps = 0

    def forward(self, x_t, x_tp1, a_t):
        mu_t, log_sigma_t = self.encoder(x_t)
        noise = torch.normal(torch.zeros_like(mu_t), torch.ones_like(mu_t))  ##
        z_t = mu_t + log_sigma_t.exp() * noise  ##
        mu_tp1_prime, log_sigma_tp1_prime = self.world_model(z_t, a_t)
        mu_tp1, log_sigma_tp1 = self.target_encode(x_tp1)
        mu_tp1, log_sigma_tp1 = mu_tp1.detach(), log_sigma_tp1.detach()
        distribution_tp1 = torch.distributions.MultivariateNormal(mu_tp1_prime,
                                                                  torch.diag_embed((log_sigma_tp1_prime.exp() + 1e-8)))
        distribution_target = torch.distributions.MultivariateNormal(mu_tp1,
                                                                     torch.diag_embed((log_sigma_tp1.exp() + 1e-8)))
        loss_reg = 0.5 * torch.sum(log_sigma_t.exp() + mu_t ** 2 - log_sigma_t - 1, dim=1)
        loss_recon = self.loss_func(distribution_target, distribution_tp1)
        loss = torch.mean(loss_recon + 0.001 * loss_reg)
        # loss = self.loss_func(distribution_target, distribution_tp1).mean()
        if torch.isnan(loss).sum() != 0:
            print(log_sigma_tp1.min(), log_sigma_tp1.max(), log_sigma_tp1_prime.min(), log_sigma_tp1_prime.max())
            raise ValueError('Nan found in WM forward.')
        # loss = self.loss_func(mu_tp1_prime, mu_tp1.detach()) + \
        #     self.loss_func(log_sigma_tp1_prime, log_sigma_tp1.detach())

        self.steps += 1
        # if self.steps % self.target_steps == 0:
        #     self.update_target()
        self.soft_update_target(self.tau)
        return loss

    def update_target(self):
        self.target_encoder = copy.deepcopy(self.encoder)

    def soft_update_target(self, tau):
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

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


class Architecture:
    def __init__(self, x_dim, a_dim):
        # type: (tuple, tuple) -> None
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        print('Observation space:', self.x_dim)
        self.model = Model(x_dim, a_dim, device=self.device)  # type: Model
        self.optimizer_wm = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.decoder = Decoder_2D(z_dim=self.model.z_dim,
                                  x_dim=x_dim,
                                  device=self.device)
        self.loss_func_d = nn.BCELoss().to(self.device)  # type: Decoder_2D
        self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=0.0001)

        self.losses = {'world_model': [], 'decoder': []}

    def train_wm(self, x_t, x_tp1, a_t):
        x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[1:]) == self.x_dim

        self.model.zero_grad()
        loss = self.model(x_t, x_tp1, a_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        self.losses['world_model'].append(loss.item())

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

    def save_wm(self, path='saved_objects/wm_encoder.pt'):
        torch.save({'model': self.model,
                    'optimizer': self.optimizer_wm,
                    'losses': self.losses['world_model']
                    }, path)

    def load_wm(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        self.optimizer_wm = checkpoint['optimizer']
        self.losses['world_model'] = checkpoint['losses']


def main(env):
    max_steps = 80000
    buffer = ReplayBuffer(20000)
    obs_dim = INPUT_DIM
    a_dim = (env.action_space.n,)
    model = Architecture(obs_dim, a_dim)
    for ep in range(100):
        s_t = env.reset()
        s_t = channel_first_numpy(resize_to_standard_dim_numpy(s_t)) / 256  # Reshape and normalise input
        done = False
        while not done:
            a_t = torch.randint(a_dim[0], (1,))
            for i in range(3):
                _ = env.step(a_t)
            s_tp1, r_t, done, _ = env.step(a_t)
            s_tp1 = channel_first_numpy(resize_to_standard_dim_numpy(s_tp1)) / 256  # Reshape and normalise input
            buffer.add(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1
            if done:
                break
    with open(r"saved_objects/buffer_Freeway.pkl", "wb") as file:
        pickle.dump(buffer, file)
    # with open(r"saved_objects/buffer.pkl", "rb") as file:
    #     buffer = pickle.load(file)
    print('Buffer size:', len(buffer))
    # model.load_wm(path='../saved_objects/wm_encoder_sampled.pt')
    for i in range(max_steps - model.model.steps):
        batch = buffer.sample(64)
        model.train_wm(torch.from_numpy(batch[0]).to(dtype=torch.float32),
                       torch.from_numpy(batch[3]).to(dtype=torch.float32),
                       torch.from_numpy(batch[1]).to(dtype=torch.float32))
        if i % 100 == 0:
            print('Step:', i, '/', max_steps, ' WM loss:', model.get_losses()['world_model'][-1])
            model.save_wm(path='saved_objects/wm_encoder_sampled.pt')
    for i in range(max_steps):
        batch = buffer.sample(64)
        model.train_d(torch.from_numpy(batch[0]).to(dtype=torch.float32))
        if i % 100 == 0:
            print('Step:', i, '/', max_steps, 'D loss:', model.get_losses()['decoder'][-1])
    env.close()

    import matplotlib.pyplot as plt
    (s_ts, a_ts, _, s_tp1s, _) = buffer.sample(10)
    s_ts = torch.from_numpy(s_ts).to(dtype=torch.float32)
    a_ts = torch.from_numpy(a_ts).to(dtype=torch.float32)
    s_tp1s = torch.from_numpy(s_tp1s).to(dtype=torch.float32)
    for i, (s_t, a_t, s_tp1) in enumerate(zip(s_ts, a_ts, s_tp1s)):
        fig, axs = plt.subplots(2, 3, sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        (ax1, ax2, ax3), (ax4, ax5, ax6) = axs

        ax1.imshow(s_t.permute(1, 2, 0).numpy())
        mu_t, _ = model.encode(s_t)
        s_t_prime = model.decode(mu_t)
        s_t_prime = s_t_prime.cpu().squeeze(0).permute(1, 2, 0)
        ax2.imshow(s_t_prime.numpy())
        ax3.imshow(s_tp1.permute(1, 2, 0).numpy())

        mu_t, log_sigma_t = model.encode(s_t)
        mu_tp1, log_sigma_tp1 = model.next_z(mu_t, a_t)
        s_tp1_prime = model.decode(mu_tp1)
        s_tp1_prime = s_tp1_prime.cpu().squeeze(0).permute(1, 2, 0)
        ax4.imshow(s_tp1_prime.numpy())

        mu_t2, log_sigma_t2 = model.next_z(mu_tp1, a_t)
        s_tp2_prime = model.decode(mu_t2)
        s_tp2_prime = s_tp2_prime.cpu().squeeze(0).permute(1, 2, 0)
        ax5.imshow(s_tp2_prime.numpy())

        mu_t3, log_sigma_t3 = model.next_z(mu_t2, a_t)
        s_tp3_prime = model.decode(mu_t3)
        s_tp3_prime = s_tp3_prime.cpu().squeeze(0).permute(1, 2, 0)
        ax6.imshow(s_tp3_prime.numpy())
        fig.suptitle('Train steps:' + str(max_steps) + ', sampled')


if __name__ == "__main__":
    environment = gym.make('Freeway-v0')
    try:
        main(environment)
    finally:
        environment.close()
