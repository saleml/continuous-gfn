import torch
import torch.nn as nn
from torchtyping import TensorType


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
        out[..., 0] = out[..., 0]
        return out
