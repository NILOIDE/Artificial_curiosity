# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import torch.nn as nn
import gym
from modules.replay_buffers.replay_buffer import ReplayBuffer
from modules.world_models.world_model import WorldModel_Sigma
from utils import resize_to_standard_dim_numpy, channel_first_numpy, INPUT_DIM
from modules.encoders.learned_encoders import Encoder_2D
from modules.decoders.decoder import Decoder_1D, Decoder_2D

import copy

torch.set_printoptions(edgeitems=10)


class Model(nn.Module):
    def __init__(self, x_dim, a_dim, z_dim=(40,), target_steps=250, device='cpu'):
        # type: (tuple, tuple, tuple, int, str) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder_2D(x_dim=self.x_dim,
                                  z_dim=self.z_dim,
                                  device=self.device)  # type: Encoder_2D
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_steps = target_steps
        self.world_model = WorldModel_Sigma(x_dim=self.encoder.get_z_dim(),
                                            a_dim=self.a_dim,
                                            vector_actions=False,
                                            hidden_dim=(256, 256, 256),
                                            device=self.device)  # type: WorldModel_Sigma

        # self.loss_func = torch.nn.SmoothL1Loss().to(self.device)
        self.loss_func = torch.distributions.kl.kl_divergence
        self.steps = 0

    def forward(self, x_t, x_tp1, a_t):
        mu_t, log_sigma_t = self.encoder(x_t)
        mu_tp1_prime, log_sigma_tp1_prime = self.world_model(mu_t, log_sigma_t, a_t)
        mu_tp1, log_sigma_tp1 = self.target_encode(x_tp1)
        mu_tp1, log_sigma_tp1 = mu_tp1.detach(), log_sigma_tp1.detach()
        distribution_tp1 = torch.distributions.MultivariateNormal(mu_tp1_prime,
                                                                  torch.diag_embed((log_sigma_tp1_prime.exp() + 1e-8)))
        distribution_target = torch.distributions.MultivariateNormal(mu_tp1,
                                                                     torch.diag_embed((log_sigma_tp1.exp() + 1e-8)))
        loss = self.loss_func(distribution_target, distribution_tp1).mean()
        if torch.isnan(loss).sum() != 0:
            print(log_sigma_tp1.min(), log_sigma_tp1.max(), log_sigma_tp1_prime.min(), log_sigma_tp1_prime.max())
        # loss = self.loss_func(mu_tp1_prime, mu_tp1.detach()) + \
        #     self.loss_func(log_sigma_tp1_prime, log_sigma_tp1.detach())

        self.steps += 1
        if self.steps % self.target_steps == 0:
            self.update_target_encoder()
        return loss

    def update_target_encoder(self):
        self.target_encoder = copy.deepcopy(self.encoder)

    def encode(self, x):
        return self.encoder(x)

    def target_encode(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def next_z(self, mu_t, log_sigma_t, a_t):
        with torch.no_grad():
            mu_tp1, log_sigma_tp1 = self.world_model(mu_t, log_sigma_t, a_t)
        return mu_tp1, log_sigma_tp1

    def next_z_from_x(self, x_t, a_t):
        with torch.no_grad():
            mu_t, log_sigma_t = self.encode(x_t)
            mu_tp1, log_sigma_tp1 = self.world_model(mu_t, log_sigma_t, a_t)
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
        self.optimizer_wm = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.decoder = Decoder_1D(z_dim=self.model.z_dim,
                                  x_dim=x_dim,
                                  device=self.device)
        self.loss_func_d = nn.BCELoss().to(self.device)  # type: Decoder_1D
        self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=0.001)

        self.losses = {'world_model': [], 'decoder': []}

    def train_wm(self, x_t, x_tp1, a_t):
        x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[1:]) == self.x_dim

        self.model.zero_grad()
        loss = self.model(x_t, x_tp1, a_t)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_wm.step()
        self.losses['world_model'].append(loss.item())

    def train_d(self, x_t):
        x_t = x_t.to(self.device)
        self.decoder.zero_grad()
        mu_t, _ = self.model.encode(x_t)
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

    def next_z(self, mu_t, log_sigma_t, a_t):
        mu_tp1, log_sigma_tp1 = self.model.next_z(mu_t, log_sigma_t, a_t)
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


def main(env):
    # env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
    # for i in gym.envs.registry.all():
    #     print(i)
    buffer = ReplayBuffer(20000)
    obs_dim = INPUT_DIM
    a_dim = (env.action_space.n,)
    model = Architecture(obs_dim, a_dim)
    for ep in range(40):
        s_t = env.reset()
        s_t = channel_first_numpy(resize_to_standard_dim_numpy(s_t)) / 256  # Reshape and normalise input
        # env.render('human')
        done = False
        while not done:
            a_t = torch.randint(a_dim[0], (1,))
            s_tp1, r_t, done, _ = env.step(a_t)
            s_tp1 = channel_first_numpy(resize_to_standard_dim_numpy(s_tp1)) / 256  # Reshape and normalise input
            # env.render('human')
            buffer.add(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1
            if done:
                break
    print('Training:', len(buffer))
    for i in range(40000):
        batch = buffer.sample(64)
        model.train_wm(torch.from_numpy(batch[0]).to(dtype=torch.float32),
                       torch.from_numpy(batch[3]).to(dtype=torch.float32),
                       torch.from_numpy(batch[1]).to(dtype=torch.float32))
        if i % 100 == 0:
            print('Step:', i, 'WM loss:', model.get_losses()['world_model'][-1])
    for i in range(5000):
        batch = buffer.sample(64)
        model.train_d(torch.from_numpy(batch[0]).to(dtype=torch.float32))
        if i % 100 == 0:
            print('Step:', i, 'D loss:', model.get_losses()['decoder'][-1])
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

        # s_tp1_prime = model.predict_next_obs(s_t, a_t)
        mu_t, log_sigma_t = model.encode(s_t)
        mu_tp1, log_sigma_tp1 = model.next_z(mu_t, log_sigma_t, a_t)
        s_tp1 = model.decode(mu_tp1)
        s_tp1 = s_tp1.cpu().squeeze(0).permute(1, 2, 0)
        ax4.imshow(s_tp1.numpy())

        mu_t2, log_sigma_t2 = model.next_z(mu_tp1, log_sigma_tp1, a_t)
        s_tp2 = model.decode(mu_t2)
        s_tp2 = s_tp2.cpu().squeeze(0).permute(1, 2, 0)
        ax5.imshow(s_tp2.numpy())

        mu_t3, log_sigma_t3 = model.next_z(mu_t2, log_sigma_t2, a_t)
        s_tp3 = model.decode(mu_t3)
        s_tp3 = s_tp3.cpu().squeeze(0).permute(1, 2, 0)
        ax6.imshow(s_tp3.numpy())


if __name__ == "__main__":
    environment = gym.make('Riverraid-v0')
    try:
        main(environment)
    finally:
        environment.close()
