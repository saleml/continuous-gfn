import torch
import torch.nn as nn
from torchtyping import TensorType
from sampler import ActionsSampler
from env import Box
from TruncatedNormal import TruncatedNormal


class BoxModel(nn.Module):
    """Neural network with parametrizable number of hidden layers and units.
    It has an additional parameter dim, that controls the output dimensionality.
    An additional boolean parameter logvar, controls whether we output means and logvars
    or just means. Both means and logvars are of dimension dim.
    An additional output of dimension 1 is added.
    """

    def __init__(self, dim=2, hidden_dim=64, n_hidden=2, logvar=False):
        super().__init__()
        self.dim = dim
        self.n_hidden = n_hidden
        self.logvar = logvar
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ELU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU(),
                )
                for _ in range(n_hidden)
            ],
            nn.Linear(hidden_dim, 1 + (2 * dim if logvar else dim)),
        )

    def forward(self, x):
        out = self.model(x)
        terminate_prob = torch.sigmoid(out[..., 0])  # P_B(s0|s) for bw models
        mean = torch.sigmoid(out[..., 1 : 1 + self.dim])
        logvar = out[..., -self.dim :] if self.logvar else None
        return terminate_prob, mean, logvar


# The following class is obsolete - It was supposed to be TruncatedNormal - could be fixed
class BoxModelActionsSampler(ActionsSampler):
    def __init__(self, env: Box, model: BoxModel, std: float | None):
        super().__init__(env)
        self.sink_state = env.sink_state
        self.model = model
        self.std = std
        assert std is not None or model.logvar

    def sample_pre_actions(self, states):
        actions = torch.full_like(states, -float("inf"))
        logprobs = torch.zeros_like(states[:, 0])
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        non_sink_states = states[non_sink_mask]
        non_sink_actions = actions[non_sink_mask]
        non_sink_logprobs = logprobs[non_sink_mask]
        terminate_prob, mean, logvar = self.model(non_sink_states)
        if self.std is not None:
            std = torch.full_like(mean, self.std)
        else:
            std = torch.exp(logvar / 2)
        terminating_states_mask = torch.rand_like(terminate_prob) < terminate_prob
        non_sink_logprobs[terminating_states_mask] = torch.log(
            terminate_prob[terminating_states_mask]
        )
        mean = mean[~terminating_states_mask]
        std = std[~terminating_states_mask]

        actions_dist = TruncatedNormal(loc=mean, scale=std, a=0, b=1)
        sampled_actions = actions_dist.sample()
        non_sink_logprobs[~terminating_states_mask] = actions_dist.log_prob(
            sampled_actions
        ).sum(dim=-1) + torch.log(1 - terminate_prob[~terminating_states_mask])
        non_sink_actions[~terminating_states_mask] = sampled_actions
        actions[non_sink_mask] = non_sink_actions
        logprobs[non_sink_mask] = non_sink_logprobs

        return actions, logprobs


def evaluate_backward_logprobs_boxmodel(
    env: Box,
    model: BoxModel,
    trajectories: TensorType["n_traj", "n", "dim"],
    actionss: TensorType["n_traj", "n-1", "dim"],
    bw_std: float | None = None,
) -> TensorType["n_traj"]:
    """Compute logprobs of trajectories in a backward model.
    Args:
        model: A backward model
        trajectories: A tensor of trajectories
    Returns:
        logprobs: A tensor of logprobs of trajectories
    """
    backward_logprobs = torch.zeros_like(trajectories[:, 1:-1, 0])
    non_sink_mask = ~torch.all(trajectories[:, 1:-1, :] == env.sink_state, dim=-1)
    terminate_prob, mean, logvar = model(non_sink_states)
    # terminate_prob[
    #     torch.any(non_sink_states >= env.delta, dim=-1)
    # ] = 0.0
    true_terminate_prob = torch.zeros_like(terminate_prob)
    terminate_prob_mask = torch.all(non_sink_states < env.delta, dim=-1)
    true_terminate_prob[terminate_prob_mask] = terminate_prob[terminate_prob_mask]

    s1_indices = (
        torch.cat(
            (
                torch.zeros(1, device=env.device),
                non_sink_mask.sum(dim=-1).cumsum(dim=-1).unique()[:-1],
            )
        )
        .unique()
        .long()
    )

    backward_logprobs[:, 0][non_sink_mask[:, 0]] = torch.log(
        true_terminate_prob[s1_indices]
    )

    s1_mask = torch.zeros_like(mean[:, 0]).bool()
    s1_mask[s1_indices] = True

    if bw_std is not None:
        std = torch.full_like(mean[~s1_mask], bw_std)
    else:
        assert model.logvar
        std = torch.exp(logvar[~s1_mask] / 2)

    actions_dist = TruncatedNormal(loc=mean[~s1_mask], scale=std, a=0, b=1)
    backward_logprobs[:, 1:][non_sink_mask[:, 1:]] = actions_dist.log_prob(
        actionss[:, 1:][~torch.all(actionss[:, 1:] == env.sink_state, dim=-1)]
    ).sum(dim=-1) + torch.log(1 - true_terminate_prob[~s1_mask])

    return backward_logprobs.sum(dim=-1)
