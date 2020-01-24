# Atari-py working by following: https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
import torch
import torch.nn as nn
import gym
import numpy as np
from modules.replay_buffers.replay_buffer import ReplayBuffer
from modules.encoders.random_encoder import RandomEncoder_2D
from modules.world_models.world_model import WorldModel, WorldModel_Recurrent
from modules.decoders.decoder import Decoder
torch.set_printoptions(edgeitems=10)

class Architecture:
    def __init__(self, x_dim, a_dim):
        # type: (tuple, tuple) -> None
        self.x_dim = x_dim
        print('Observation space:', self.x_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = RandomEncoder_2D(x_dim=x_dim,
                                        device=self.device,
                                        z_dim=(40,))
        self.decoder = Decoder(z_dim=self.encoder.get_z_dim(),
                               x_dim=x_dim,
                               device=self.device)
        self.loss_func_d = nn.BCELoss(reduction='none').to(self.device)
        self.optimizer_d = torch.optim.Adam(self.decoder.parameters())

        self.world_model = WorldModel(x_dim=self.encoder.get_z_dim(),
                                      a_dim=a_dim,
                                      vector_actions=False,
                                      device=self.device)
        self.loss_func_wm = nn.MSELoss(reduction='none').to(self.device)
        self.optimizer_wm = torch.optim.Adam(self.world_model.parameters())

        self.steps = 0
        self.losses = {'world_model': [], 'decoder': []}

    def train(self, x_t, x_tp1, a_t):
        x_t, x_tp1, a_t = x_t.to(self.device), x_tp1.to(self.device), a_t.to(self.device)
        assert x_t.shape == x_tp1.shape
        assert tuple(x_t.shape[1:]) == self.x_dim

        z_t = self.encoder(x_t)
        z_tp1 = self.encoder(x_tp1)

        self.world_model.zero_grad()
        z_tp1_prime = self.world_model(z_t, a_t)
        loss_wm = torch.sum(self.loss_func_wm(z_tp1_prime, z_tp1))
        self.losses['world_model'].append(loss_wm.item())
        loss_wm.backward()
        self.optimizer_wm.step()

        self.decoder.zero_grad()
        x_t_prime = self.decoder(z_t)
        loss_d_t = torch.sum(self.loss_func_d(x_t_prime, x_t))
        self.losses['decoder'].append(loss_d_t.item())
        loss_d_t.backward()
        self.optimizer_d.step()
        self.steps += 1
        # x_tp1_prime = self.decoder(z_tp1)
        # loss_d_tp1 = torch.sum(self.loss_func(x_tp1_prime, x_tp1))
        # loss_d_tp1.backward()
        # self.optimizer_d.step()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward_model(self, z):
        return self.world_model(z)

    def predict_next_obs(self, x_t, a_t):
        z_t = self.encoder(x_t)
        z_tp1_prime = self.world_model(z_t, a_t)
        return self.decoder(z_tp1_prime).detach()

    def get_losses(self):
        return self.losses


def main(env):
    # env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
    # for i in gym.envs.registry.all():
    #     print(i)
    buffer = ReplayBuffer(5000)
    obs_dim = env.observation_space.shape
    a_dim = (env.action_space.n,)
    x_dim = (obs_dim[2], obs_dim[0], obs_dim[1])
    model = Architecture(x_dim, a_dim)
    for ep in range(20):
        s_t = env.reset()
        s_t = np.transpose(s_t, (2, 0, 1)).astype(np.float) / 256            # Channel first and normalize input
        # env.render('human')
        for t in range(100):
            a_t = torch.randint(a_dim[0], (1,))
            s_tp1, r_t, done, _ = env.step(a_t)
            s_tp1 = np.transpose(s_tp1, (2, 0, 1)).astype(np.float) / 256    # Channel first and normalize input
            # env.render('human')
            buffer.add(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1
            batch = buffer.sample(32)
            model.train(torch.from_numpy(batch[0]).to(dtype=torch.float32),
                        torch.from_numpy(batch[3]).to(dtype=torch.float32),
                        torch.from_numpy(batch[1]).to(dtype=torch.float32))

        if ep % 5 == 0:
            print('Ep:', ep, 'WM loss:', model.get_losses()['world_model'][-1], '    Decoder loss:', model.get_losses()['decoder'][-1])
    env.close()

    import matplotlib.pyplot as plt
    (s_ts, a_ts, _, s_tp1s, _) = buffer.sample(10)
    s_ts = torch.from_numpy(s_ts).to(dtype=torch.float32)
    a_ts = torch.from_numpy(a_ts).to(dtype=torch.float32)
    s_tp1s = torch.from_numpy(s_tp1s).to(dtype=torch.float32)
    for i, (s_t, a_t, s_tp1) in enumerate(zip(s_ts, a_ts, s_tp1s)):
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        (ax1, ax2), (ax3, ax4) = axs

        ax1.imshow(s_t.permute(1, 2, 0).numpy())
        s_t_prime = model.decode(model.encode(s_t))
        s_t_prime = s_t_prime.detach().cpu().squeeze(0).permute(1, 2, 0)
        ax2.imshow(s_t_prime.numpy())

        ax3.imshow(s_tp1.permute(1, 2, 0).numpy())
        s_tp1_prime = model.predict_next_obs(s_t, a_t)
        s_tp1_prime = s_tp1_prime.detach().cpu().squeeze(0).permute(1, 2, 0)
        ax4.imshow(s_tp1_prime.numpy())


if __name__ == "__main__":
    environment = gym.make('Riverraid-v0')
    try:
        main(environment)
    finally:
        environment.close()
