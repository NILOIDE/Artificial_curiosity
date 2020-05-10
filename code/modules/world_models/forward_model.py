import torch
import torch.nn as nn


class BaseForwardModel(nn.Module):

    def __init__(self, x_dim, a_dim, device='cpu'):
        # type: (tuple, tuple, str) -> None
        """"
        This is the base class for the world models below.
        Contains all the required shared variables and functions.
        """
        super().__init__()
        if len(x_dim) != 1:
            raise ValueError("World model's state input should be 1-dimensional. Received:", x_dim)
        self.x_dim = x_dim
        if len(a_dim) != 1:
            raise ValueError("World model's action input should be 1-dimensional. Received:", a_dim)
        self.a_dim = a_dim
        print('World model has input dimensions:', x_dim, '+', a_dim)
        if device in {'cuda', 'cpu'}:
            self.device = device
            self.cuda = True if device == 'cuda' else False

    def apply_tensor_constraints(self, x, required_dim, name=''):
        # type: (torch.Tensor, tuple, str) -> torch.Tensor
        # assert type(x) == torch.Tensor
        # if len(tuple(x.shape)) != 1 and len(tuple(x.shape)) != 2:
        #     if name != '':
        #         name += ' '
        #     raise ValueError("World model " + name + "input tensor should be 1D (single example) or 2D (batch).")
        if len(tuple(x.shape)) == 1:  # Add batch dimension to 1D tensor
            if x.shape[0] == required_dim:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)
        # if self.cuda:
        x = x.to(self.device)
        return x

    def apply_state_constraints(self, x, x_name=''):
        # type: (torch.Tensor, str) -> torch.Tensor
        """"
        Apply constraints to state input tensors.
        """
        x = self.apply_tensor_constraints(x, self.x_dim, x_name)
        assert x.shape[-1] == self.x_dim[0], f'Expected {self.x_dim[0]}, received {x.shape[-1]}'
        return x

    def apply_action_constraints(self, a, name='action'):
        # type: (torch.Tensor, str) -> torch.Tensor
        """"
        Apply constraints to action input tensors.
        """
        a = self.apply_tensor_constraints(a, self.a_dim, name)
        # assert a.shape[1] == self.a_dim[0] or a.shape[1] == 1
        if a.shape[1] == 1:
            a = self.action_index_to_onehot(a)
        return a

    def action_index_to_onehot(self, a_i):
        # type: (torch.Tensor) -> torch.Tensor
        """"
        Given an action's index, return a one-hot vector where the action in question has the value 1.
        """
        assert type(a_i) == torch.Tensor
        batch_size = a_i.shape[0]
        one_hot = torch.zeros((batch_size, self.a_dim[0]), device=self.device, dtype=torch.float32)
        batch_dim = torch.arange(batch_size, device=self.device)
        one_hot[batch_dim, a_i[:, 0].to(dtype=torch.long)] += 1.0  # Place ones into taken action indices
        return one_hot

    def get_x_dim(self) -> tuple:
        return self.x_dim

    def get_a_dim(self) -> tuple:
        return self.a_dim


class ForwardModel(BaseForwardModel):

    def __init__(self, x_dim, a_dim, hidden_dim=(512,), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        """"
        A world model predicts the state at t+1, given state at t using a feed-forward model.
        This class predicts a distribution over latent space variables. Distributions are assumed Gaussian.
        """
        super().__init__(x_dim, a_dim, device)
        self.layers = []
        h_dim_prev = self.x_dim[0] + self.a_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, self.x_dim[0]))
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, x_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Perform forward pass of world model.
        Make sure that any constraints are enforced.
        """
        x_t = self.apply_state_constraints(x_t)        # Make sure dimensions are correct
        a_t = self.apply_action_constraints(a_t)       # Make sure dimensions are correct
        assert x_t.shape[0] == a_t.shape[0]            # Same batch size
        xa_t = torch.cat((x_t, a_t), dim=1)
        x_tp1 = self.model(xa_t)
        # x_tp1 = self.apply_state_constraints(x_tp1)    # Make sure dimensions are correct
        return x_tp1


class ForwardModel_SigmaOutputOnly(BaseForwardModel):

    def __init__(self, x_dim, a_dim, hidden_dim=(256, 256), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        """"
        A world model predicts the state at t+1, given state at t using a feed-forward model.
        This class predicts a distribution over latent space variables. Distributions are assumed Gaussian.
        """
        super().__init__(x_dim, a_dim, device)
        self.layers = []
        self.input_dim = (x_dim[0] + a_dim[0],)
        h_dim_prev = self.input_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.shared_layers = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Sequential(nn.Linear(h_dim_prev, x_dim[0])).to(self.device)
        # St.dev. can't be negative, thus ReLU
        self.sigma_head = nn.Sequential(nn.Linear(h_dim_prev, x_dim[0]), nn.ReLU()).to(self.device)

    def forward(self, z_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of world model. Means and variances are returned
        Make sure that any constraints are enforced.
        """
        # z_t = self.apply_state_constraints(z_t)                     # Make sure dimensions are correct
        a_t = self.apply_action_constraints(a_t)                     # Make sure dimensions are correct
        assert z_t.shape[0] == a_t.shape[0]                         # Same batch size

        xa_t = torch.cat((z_t, a_t), dim=1)
        assert xa_t.shape[0] == z_t.shape[0]                        # Same batch size
        assert xa_t.shape[1] == self.input_dim[0]                    # Dimensions of model input
        out = self.shared_layers(xa_t)
        mu_tp1 = self.mu_head(out)
        log_sigma_tp1 = self.sigma_head(out)

        mu_tp1 = self.apply_state_constraints(mu_tp1)                # Make sure dimensions are correct
        log_sigma_tp1 = self.apply_state_constraints(log_sigma_tp1)  # Make sure dimensions are correct
        return mu_tp1, log_sigma_tp1


class ForwardModel_SigmaInputOutput(BaseForwardModel):

    def __init__(self, x_dim, a_dim, hidden_dim=(256, 256), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        """"
        A world model predicts the state at t+1, given state at t using a feed-forward model.
        This class predicts a distribution over latent space variables. Distributions are assumed Gaussian.
        """
        super().__init__(x_dim, a_dim, device)
        self.layers = []
        self.input_dim = (x_dim[0] * 2 + a_dim[0],)
        h_dim_prev = self.input_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.shared_layers = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Sequential(nn.Linear(h_dim_prev, x_dim[0])).to(self.device)
        # St.dev. can't be negative, thus ReLU
        self.sigma_head = nn.Sequential(nn.Linear(h_dim_prev, x_dim[0]), nn.ReLU()).to(self.device)

    def forward(self, mu_t, log_sigma_t, a_t):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of world model. Means and variances are returned
        Make sure that any constraints are enforced.
        """
        mu_t = self.apply_state_constraints(mu_t)                    # Make sure dimensions are correct
        log_sigma_t = self.apply_state_constraints(log_sigma_t)              # Make sure dimensions are correct
        a_t = self.apply_action_constraints(a_t)                     # Make sure dimensions are correct
        assert log_sigma_t.shape == mu_t.shape                           # Each latent has a mu-sigma pair
        assert mu_t.shape[0] == a_t.shape[0]                         # Same batch size

        xa_t = torch.cat((mu_t, log_sigma_t, a_t), dim=1)
        assert xa_t.shape[0] == mu_t.shape[0]                        # Same batch size
        assert xa_t.shape[1] == self.input_dim[0]                    # Dimensions of model input
        out = self.shared_layers(xa_t)
        mu_tp1 = self.mu_head(out)
        log_sigma_tp1 = self.sigma_head(out)

        mu_tp1 = self.apply_state_constraints(mu_tp1)                # Make sure dimensions are correct
        log_sigma_tp1 = self.apply_state_constraints(log_sigma_tp1)  # Make sure dimensions are correct
        return mu_tp1, log_sigma_tp1


class ForwardModel_Recurrent(BaseForwardModel):

    def __init__(self, x_dim, a_dim, cell_dim=256, linear_dim=(256,), device='cpu'):
        # type: (tuple, tuple, int, tuple, str) -> None
        """"
        A world model predicts the state at t+1, given state at t using a feed-forward model.
        This class predicts a distribution over latent space variables. Distributions are assumed Gaussian.
        """
        super().__init__(x_dim, a_dim, device)
        self.layers = []
        self.lstm = nn.LSTM(input_size=self.x_dim[0] + self.a_dim[0],
                            hidden_size=cell_dim,
                            num_layers=1,
                            bias=True,
                            dropout=0.0,
                            batch_first=True).to(self.device)
        h_dim_prev = cell_dim
        for h_dim in linear_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, self.x_dim[0]))
        self.nonlinear_layers = nn.Sequential(*self.layers).to(self.device)

    def forward(self, x_t, a_t, last_state=None):
        # type: (torch.Tensor, torch.Tensor, tuple) -> [torch.Tensor, tuple]
        """
        Perform forward pass of world model.
        Make sure that any constraints are enforced.
        """
        x_t = self.apply_state_constraints(x_t)                 # Make sure dimensions are correct
        a_t = self.apply_action_constraints(a_t)                # Make sure dimensions are correct
        assert x_t.shape[0] == a_t.shape[0]                     # Same batch size
        xa_t = torch.cat((x_t, a_t), dim=1)
        assert xa_t.shape[0] == x_t.shape[0]                    # Same batch size

        out, last_state = self.lstm(xa_t, last_state)
        x_tp1 = self.nonlinear_layers(out)
        self.apply_state_constraints(x_tp1)                     # Make sure dimensions are correct
        return x_tp1, last_state


class ForwardModel_Recurrent_Sigma(BaseForwardModel):

    def __init__(self, x_dim, a_dim, cell_dim=256, linear_dim=(256,), device='cpu'):
        # type: (tuple, tuple, int, tuple, str) -> None
        """"
        A world model predicts the state at t+1, given state at t using a feed-forward model.
        This class predicts a distribution over latent space variables. Distributions are assumed Gaussian.
        """
        super().__init__(x_dim, a_dim, device)
        self.input_dim = (x_dim[0] * 2 + a_dim[0],)
        self.layers = []
        self.lstm = nn.LSTM(input_size=self.input_dim[0],
                            hidden_size=cell_dim,
                            num_layers=1,
                            bias=True,
                            dropout=0.0,
                            batch_first=True).to(self.device)
        h_dim_prev = cell_dim
        for h_dim in linear_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.nonlinear_layers = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Sequential(nn.Linear(h_dim_prev, self.x_dim[0])).to(self.device)
        # St.dev. can't be negative, thus ReLU
        self.sigma_head = nn.Sequential(nn.Linear(h_dim_prev, x_dim[0]), nn.ReLU()).to(self.device)

    def forward(self, mu_t, sigma_t, a_t, last_state=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, tuple) -> [torch.Tensor, torch.Tensor, tuple]
        """
        Perform forward pass of world model. Means and variances are returned
        Make sure that any constraints are enforced.
        """
        mu_t = self.apply_state_constraints(mu_t)                    # Make sure dimensions are correct
        sigma_t = self.apply_state_constraints(sigma_t)              # Make sure dimensions are correct
        a_t = self.apply_action_constraints(a_t)                     # Make sure dimensions are correct
        assert sigma_t.shape == mu_t.shape                           # Each latent has a mu-sigma pair
        assert mu_t.shape[0] == a_t.shape[0]                         # Same batch size

        xa_t = torch.cat((mu_t, sigma_t, a_t), dim=1)
        assert xa_t.shape[0] == mu_t.shape[0]                        # Same batch size
        assert xa_t.shape[1] == self.input_dim[0]                    # Dimensions of model input

        out, last_state = self.lstm(xa_t, last_state)
        out = self.nonlinear_layers(out)
        mu_tp1 = self.mu_head(out)
        log_sigma_tp1 = self.sigma_head(out)

        mu_tp1 = self.apply_state_constraints(mu_tp1)                # Make sure dimensions are correct
        log_sigma_tp1 = self.apply_state_constraints(log_sigma_tp1)  # Make sure dimensions are correct
        return mu_tp1, log_sigma_tp1, last_state
