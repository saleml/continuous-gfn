import torch
import torch.nn as nn
from torch.distributions import Categorical, Beta, MixtureSameFamily


class NeuralNet(nn.Module):
    def __init__(self, dim=2, hidden_dim=64, n_hidden=2, torso=None, output_dim=3):
        super().__init__()
        self.dim = dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim
        if torso is not None:
            self.torso = torso
        else:
            self.torso = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ELU(),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU(),
                    )
                    for _ in range(n_hidden)
                ],
            )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.output_layer(self.torso(x))
        return out


class CirclePF(NeuralNet):
    def __init__(
        self,
        hidden_dim=64,
        n_hidden=2,
        n_components_s0=1,
        n_components=1,
        beta_min=0.1,
        beta_max=2.0,
    ):
        output_dim = 1 + 3 * n_components
        super().__init__(
            dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden, output_dim=output_dim
        )

        # The following parameters are for PF(. | s0)
        self.PFs0 = nn.ParameterDict(
            {
                "log_alpha_r": nn.Parameter(torch.zeros(n_components_s0)),
                "log_alpha_theta": nn.Parameter(torch.zeros(n_components_s0)),
                "log_beta_r": nn.Parameter(torch.zeros(n_components_s0)),
                "log_beta_theta": nn.Parameter(torch.zeros(n_components_s0)),
                "logits": nn.Parameter(torch.zeros(n_components_s0)),
            }
        )

        self.n_components = n_components
        self.n_components_s0 = n_components_s0
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x):
        out = super().forward(x)
        pre_sigmoid_exit = out[..., 0]
        mixture_logits = out[..., 1 : 1 + self.n_components]
        log_alpha = out[..., 1 + self.n_components : 1 + 2 * self.n_components]
        log_beta = out[..., 1 + 2 * self.n_components : 1 + 3 * self.n_components]

        exit_proba = torch.sigmoid(pre_sigmoid_exit)
        return (
            exit_proba,
            mixture_logits,
            self.beta_max * torch.sigmoid(log_alpha) + self.beta_min,
            self.beta_max * torch.sigmoid(log_beta) + self.beta_min,
        )

    def to_dist(self, x):
        if torch.all(x[0] == 0.0):
            assert torch.all(
                x == 0.0
            )  # If one of the states is s0, all of them must be
            alpha_r = self.PFs0["log_alpha_r"]
            alpha_r = self.beta_max * torch.sigmoid(alpha_r) + self.beta_min
            alpha_theta = self.PFs0["log_alpha_theta"]
            alpha_theta = self.beta_max * torch.sigmoid(alpha_theta) + self.beta_min
            beta_r = self.PFs0["log_beta_r"]
            beta_r = self.beta_max * torch.sigmoid(beta_r) + self.beta_min
            beta_theta = self.PFs0["log_beta_theta"]
            beta_theta = self.beta_max * torch.sigmoid(beta_theta) + self.beta_min

            logits = self.PFs0["logits"]
            dist_r = MixtureSameFamily(
                Categorical(logits=logits),
                Beta(alpha_r, beta_r),
            )
            dist_theta = MixtureSameFamily(
                Categorical(logits=logits),
                Beta(alpha_theta, beta_theta),
            )
            return dist_r, dist_theta

        # Otherwise, we use the neural network
        exit_proba, mixture_logits, alpha, beta = self.forward(x)
        if self.one_component:
            dist = Beta(alpha.squeeze(), beta.squeeze())
            return exit_proba, dist
        dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha, beta),
        )

        return exit_proba, dist


class CirclePB(NeuralNet):
    def __init__(
        self,
        hidden_dim=64,
        n_hidden=2,
        torso=None,
        uniform=False,
        n_components=1,
        beta_min=0.1,
        beta_max=2.0,
    ):
        output_dim = 3 * n_components
        super().__init__(
            dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden, output_dim=output_dim
        )
        if torso is not None:
            self.torso = torso
        self.uniform = uniform
        self.n_components = n_components
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x):
        # x is a batch of states, a tensor of shape (batch_size, dim) with dim == 2
        out = super().forward(x)
        mixture_logits = out[:, 0 : self.n_components]
        log_alpha = out[:, self.n_components : 2 * self.n_components]
        log_beta = out[:, 2 * self.n_components : 3 * self.n_components]
        return (
            mixture_logits,
            self.beta_max * torch.sigmoid(log_alpha) + self.beta_min,
            self.beta_max * torch.sigmoid(log_beta) + self.beta_min,
        )

    def to_dist(self, x):
        if self.uniform:
            return Beta(torch.ones(x.shape[0]), torch.ones(x.shape[0]))
        mixture_logits, alpha, beta = self.forward(x)
        dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha, beta),
        )
        return dist
