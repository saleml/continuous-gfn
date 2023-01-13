import torch
import torch.nn as nn
from torchtyping import TensorType
from torch.distributions import Categorical, Beta, MixtureSameFamily, Independent


class NeuralNet(nn.Module):
    def __init__(self, dim=2, hidden_dim=64, n_hidden=2, output_dim=3):
        super().__init__()
        self.dim = dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim
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
    def __init__(self, hidden_dim=64, n_hidden=2, n_components=1):
        output_dim = 3 if n_components == 1 else 1 + 3 * n_components
        super().__init__(
            dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden, output_dim=output_dim
        )

        # The following parameters are for PF(. | s0)
        self.PFs0 = nn.ParameterDict(
            {
                "log_alpha_r": nn.Parameter(torch.zeros(n_components)),
                "log_alpha_theta": nn.Parameter(torch.zeros(n_components)),
                "log_beta_r": nn.Parameter(torch.zeros(n_components)),
                "log_beta_theta": nn.Parameter(torch.zeros(n_components)),
            }
        )
        if n_components > 1:
            self.PFs0["logits"] = nn.Parameter(torch.zeros(n_components))

        self.n_components = n_components

    def forward(self, x):
        # x is a batch of states, a tensor of shape (batch_size, dim) with dim == 2
        if torch.all(x[0] == 0.0):
            assert torch.all(
                x == 0.0
            )  # If one of the states is s0, all of them must be
            return tuple([param.exp() for param in self.PFs0.values()])

        # Otherwise, we use the neural network
        out = super().forward(x)
        pre_sigmoid_exit = out[:, 0]
        log_alpha = out[:, 1]
        log_beta = out[:, 2]

        exit_proba = torch.sigmoid(pre_sigmoid_exit)
        return (
            exit_proba,
            torch.sigmoid(2 * log_alpha) + 0.1,
            torch.sigmoid(2 * log_beta) + 0.1,
        )


class CirclePB(NeuralNet):
    def __init__(self, hidden_dim=64, n_hidden=2, torso=None, uniform=False):
        super().__init__(dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden, output_dim=2)
        if torso is not None:
            self.torso = torso
        self.uniform = uniform

    def forward(self, x):
        # x is a batch of states, a tensor of shape (batch_size, dim) with dim == 2
        if self.uniform:
            return torch.ones(x.shape[0]), torch.ones(x.shape[0])
        out = super().forward(x)
        log_alpha = out[:, 0]
        log_beta = out[:, 1]
        return torch.sigmoid(2 * log_alpha) + 0.1, torch.sigmoid(2 * log_beta) + 0.1
