import torch
from torch.distributions import Beta, Categorical, Independent, MixtureSameFamily
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import wandb
import argparse
from sklearn.neighbors import KernelDensity

from env import Box, get_last_states
from model import CirclePF, CirclePB

from utils import (
    fit_kde,
    plot_reward,
    sample_from_reward,
    plot_samples,
    estimate_jsd,
    plot_trajectories,
    plot_termination_probabilities,
)


USE_WANDB = True

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--delta", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=1e-4)
parser.add_argument(
    "--n_components",
    type=int,
    default=1,
    help="Number of components in Mixture Of Betas",
)
parser.add_argument(
    "--PB",
    type=str,
    choices=["learnable", "tied", "uniform"],
    default="learnable",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_Z", type=float, default=1e-1)
parser.add_argument("--BS", type=int, default=128)
parser.add_argument("--n_iterations", type=int, default=100000)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--n_hidden", type=int, default=3)
args = parser.parse_args()

if USE_WANDB:
    wandb.init(project="cont_hypergrid", save_code=True)
    wandb.config.update(args)

dim = args.dim
delta = args.delta
epsilon = args.epsilon
seed = args.seed
lr = args.lr
lr_Z = args.lr_Z
n_iterations = args.n_iterations
BS = args.BS
n_components = args.n_components

if seed == 0:
    seed = np.random.randint(int(1e6))

torch.manual_seed(seed)
np.random.seed(seed)

env = Box(
    dim=dim,
    delta=delta,
    epsilon=epsilon,
    device_str="cpu",
    reward_cos=False,
    verify_actions=False,
)

# Get the true KDE
samples = sample_from_reward(env, n_samples=10000)
true_kde, fig1 = fit_kde(samples)

if USE_WANDB:
    # log the reward figure
    fig2 = plot_reward(env)

    wandb.log(
        {
            "reward": wandb.Image(fig2),
            "reward_kde": wandb.Image(fig1),
        }
    )


model = CirclePF(hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
bw_model = CirclePB(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    torso=model.torso if args.PB == "tied" else None,
    uniform=args.PB == "uniform",
)


def sample_actions(model, states, on_policy=False):
    # states is a tensor of shape (n, dim)
    batch_size = states.shape[0]
    out = model(states)
    if len(out) == 4:  # s0 input
        alpha_r, alpha_theta, beta_r, beta_theta = out
        alpha = torch.stack([alpha_r, alpha_theta], dim=1)
        beta = torch.stack([beta_r, beta_theta], dim=1)
        dist = Beta(alpha, beta)

        samples = dist.sample(torch.Size((batch_size,))).squeeze(1)
        r, theta = samples[:, 0], samples[:, 1]
        actions = (
            torch.stack(
                [
                    r * torch.cos(torch.pi / 2.0 * theta),
                    r * torch.sin(torch.pi / 2.0 * theta),
                ],
                dim=1,
            )
            * env.delta
        )
        logprobs = dist.log_prob(samples).sum(dim=1) - torch.log(r * env.delta)

    else:
        exit_proba, alpha, beta = out
        if not on_policy:
            exit_proba = exit_proba.clamp_min(0.1)
        exit = torch.bernoulli(exit_proba).bool()
        exit[torch.norm(1 - states, dim=1) <= env.delta] = True
        exit[torch.any(states >= 1 - env.epsilon, dim=-1)] = True
        A = torch.where(
            states[:, 0] <= 1 - env.delta,
            0.0,
            2.0 / torch.pi * torch.arccos((1 - states[:, 0]) / env.delta),
        )
        B = torch.where(
            states[:, 1] <= 1 - env.delta,
            1.0,
            2.0 / torch.pi * torch.arcsin((1 - states[:, 1]) / env.delta),
        )
        assert torch.all(
            B[~torch.any(states >= 1 - env.delta, dim=-1)]
            >= A[~torch.any(states >= 1 - env.delta, dim=-1)]
        )
        dist = Beta(alpha, beta)
        samples = dist.sample()
        actions = samples * (B - A) + A
        actions *= torch.pi / 2.0
        actions = (
            torch.stack([torch.cos(actions), torch.sin(actions)], dim=1) * env.delta
        )

        logprobs = (
            dist.log_prob(samples)
            + torch.log(1 - exit_proba)
            + torch.log(2.0 / (torch.pi * env.delta * (B - A)))
            # - 0.5 * torch.log(1 - torch.cos(actions[:, 0]) ** 2)
        )

        actions[exit] = -float("inf")
        logprobs[exit] = torch.log(exit_proba[exit])
        logprobs[torch.norm(1 - states, dim=1) <= env.delta] = 0.0
        logprobs[torch.any(states >= 1 - env.epsilon, dim=-1)] = 0.0

    return actions, logprobs


def sample_trajectories(model, n_trajectories, on_policy=False):
    states = torch.zeros((n_trajectories, env.dim), device=env.device)
    actionss = []
    trajectories = [states]
    trajectories_logprobs = torch.zeros((n_trajectories,), device=env.device)
    while not torch.all(states == env.sink_state):
        non_terminal_mask = torch.all(states != env.sink_state, dim=-1)
        actions = torch.full(
            (n_trajectories, env.dim), -float("inf"), device=env.device
        )
        non_terminal_actions, logprobs = sample_actions(
            model, states[non_terminal_mask], on_policy=on_policy
        )
        actions[non_terminal_mask] = non_terminal_actions
        actionss.append(actions)
        states = env.step(states, actions)
        trajectories.append(states)
        trajectories_logprobs[non_terminal_mask] += logprobs
    trajectories = torch.stack(trajectories, dim=1)
    actionss = torch.stack(actionss, dim=1)
    return trajectories, actionss, trajectories_logprobs


def evaluate_backward_logprobs(model, trajectories):
    logprobs = torch.zeros((trajectories.shape[0],), device=env.device)
    for i in range(trajectories.shape[1] - 2, 1, -1):
        non_sink_mask = torch.all(trajectories[:, i] != env.sink_state, dim=-1)
        current_states = trajectories[:, i][non_sink_mask]
        previous_states = trajectories[:, i - 1][non_sink_mask]
        difference_1 = current_states[:, 0] - previous_states[:, 0]
        difference_1.clamp_(
            min=0.0, max=env.delta
        )  # Should be the case already - just to avoid numerical issues
        A = torch.where(
            current_states[:, 0] >= env.delta,
            0.0,
            2.0 / torch.pi * torch.arccos((current_states[:, 0]) / env.delta),
        )
        B = torch.where(
            current_states[:, 1] >= env.delta,
            1.0,
            2.0 / torch.pi * torch.arcsin((current_states[:, 1]) / env.delta),
        )

        alpha, beta = model(current_states)
        dist = Beta(alpha, beta)

        step_logprobs = (
            dist.log_prob(
                (
                    1.0
                    / (B - A)
                    * (2.0 / torch.pi * torch.acos(difference_1 / env.delta) - A)
                ).clamp(1e-4, 1 - 1e-4)
            ).clamp_max(100)
            + torch.log(2.0 / (torch.pi * env.delta * (B - A)))
            # - 0.5 * torch.log(1 - (difference_1 / env.delta) ** 2)
        )

        if torch.any(torch.isnan(step_logprobs)):
            raise ValueError("NaN in backward logprobs")

        if torch.any(torch.isinf(step_logprobs)):
            raise ValueError("Inf in backward logprobs")

        logprobs[non_sink_mask] += step_logprobs

    return logprobs


logZ = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if args.PB != "uniform":
    optimizer.add_param_group(
        {
            "params": bw_model.output_layer.parameters()
            if args.PB == "tied"
            else bw_model.parameters(),
            "lr": lr,
        }
    )
optimizer.add_param_group({"params": [logZ], "lr": lr_Z})

jsd = float("inf")
for i in trange(n_iterations):
    optimizer.zero_grad()
    trajectories, actionss, logprobs = sample_trajectories(model, BS, on_policy=True)
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs = evaluate_backward_logprobs(bw_model, trajectories)

    loss = torch.mean((logZ + logprobs - bw_logprobs - logrewards) ** 2)
    if torch.isinf(loss):
        raise ValueError("Infinite loss")
    loss.backward()
    # clip the gradients for bw_model
    for p in bw_model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-10, 10).nan_to_num_(0.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-10, 10).nan_to_num_(0.0)
    optimizer.step()

    if any(
        [
            torch.isnan(list(model.parameters())[i]).any()
            for i in range(len(list(model.parameters())))
        ]
    ):
        raise ValueError("NaN in model parameters")

    if i % 100 == 0:
        if USE_WANDB:
            wandb.log(
                {
                    "loss": loss.item(),
                    "logZdiff": logZ.item() - np.log(env.Z),
                    "states_visited": (i + 1) * BS,
                },
                step=i,
            )
        tqdm.write(
            f"{i}: {loss.item()}, {logZ.item()}, {np.log(env.Z)}, {jsd}, {tuple(trajectories.shape)}"
        )
    if i % 1000 == 0:
        trajectories, actionss, logprobs = sample_trajectories(
            model, 1000, on_policy=True
        )
        last_states = get_last_states(env, trajectories)
        kde, fig4 = fit_kde(last_states)
        jsd = estimate_jsd(kde, true_kde)

        if USE_WANDB:
            colors = plt.cm.rainbow(np.linspace(0, 1, 10))

            fig1 = plot_samples(last_states.detach().cpu().numpy())
            fig2 = plot_trajectories(trajectories.detach().cpu().numpy()[:20])
            fig3 = plot_termination_probabilities(model)

            wandb.log(
                {
                    "last_states": wandb.Image(fig1),
                    "trajectories": wandb.Image(fig2),
                    "termination_probs": wandb.Image(fig3),
                    "kde": wandb.Image(fig4),
                    "JSD": jsd,
                },
                step=i,
            )


if USE_WANDB:
    wandb.finish()
