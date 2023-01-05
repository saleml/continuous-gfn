import torch
from env import Box, get_last_states
from sampler2 import (
    BetaActionsSampler,
    MixtureOfBetasActionsSampler,
    TruncatedNormalActionsSampler,
    TrajectoriesSampler,
)
from model import NeuralNet

import matplotlib.pyplot as plt
import numpy as np
import wandb
from argparse import ArgumentParser

import io
from PIL import Image
from tqdm import tqdm, trange


parser = ArgumentParser()

parser.add_argument("--wandb", action="store_true", help="Use wandb")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--n_components",
    type=int,
    default=1,
    help="Number of components of the mixture of betas",
)


parser.add_argument("--dim", type=int, default=2, help="Dimension of the Box")
parser.add_argument("--delta", type=float, default=0.1, help="Delta for the Box")
parser.add_argument("--cos", action="store_true", help="Use cosinus reward for the Box")

parser.add_argument(
    "--sf_bias", type=float, default=0.0, help="To subtract from exit action logits"
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.0,
    help="with proba epsilon, take random actions for non terminating states",
)


parser.add_argument(
    "--PB",
    type=str,
    choices=["learnable", "tied"],
    default="learnable",
    help="Learnable or tied parameters for the BoxModel",
)

parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--lr_Z", type=float, default=1e-1, help="Learning rate")


parser.add_argument(
    "--n_validation_trajectories",
    type=int,
    default=10000,
    help="Number of trajectories for validation",
)

parser.add_argument(
    "--n_iterations",
    type=int,
    default=100000,
    help="Number of iterations for training",
)

args = parser.parse_args()

print(args)

torch.manual_seed(args.seed)
dim = args.dim
delta = args.delta

use_wandb = args.wandb
if use_wandb:
    wandb.init(project="cont_hypergrid")
    wandb.config.update(args)

env = Box(dim=dim, delta=delta, device_str="cpu", reward_cos=args.cos)

if args.n_components == 1:
    output_dim = 1 + 2 * dim
else:
    output_dim = (
        1 + (2 * dim + 1) * args.n_components
    )  # The extra 1 is because logits (of mixture) are always overparametrized by one
model = NeuralNet(dim=dim, hidden_dim=128, n_hidden=2, output_dim=output_dim)
bw_model = NeuralNet(dim=dim, hidden_dim=128, n_hidden=2, output_dim=output_dim)
if args.PB == "tied":
    bw_model.torso = model.torso

if args.n_components == 1:
    actions_sampler = BetaActionsSampler(
        env, model, sf_bias=args.sf_bias, epsilon_uniform=args.epsilon
    )
else:
    actions_sampler = MixtureOfBetasActionsSampler(
        env,
        model,
        n_components=args.n_components,
        sf_bias=args.sf_bias,
        epsilon_uniform=args.epsilon,
    )
trajectories_sampler = TrajectoriesSampler(env, actions_sampler)


n_validation_trajectories = args.n_validation_trajectories

lr = args.lr
lr_Z = args.lr_Z
BS = args.batch_size

logZ = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if args.PB == "learnable":
    optimizer.add_param_group({"params": bw_model.parameters(), "lr": lr})
elif args.PB == "tied":
    optimizer.add_param_group({"params": bw_model.output_layer.parameters(), "lr": lr})
else:
    raise ValueError("Unknown PB")
optimizer.add_param_group({"params": [logZ], "lr": lr_Z})

for i in trange(args.n_iterations):
    optimizer.zero_grad()
    trajectories, actionss, logprobs = trajectories_sampler.sample(BS)
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs = actions_sampler.evaluate_backward_logprobs(
        bw_model, trajectories, actionss
    )
    loss = torch.mean((logZ + logprobs - bw_logprobs - logrewards) ** 2)
    if torch.isinf(loss):
        raise ValueError("Infinite loss")
    loss.backward()
    optimizer.step()

    if any(
        [
            torch.isnan(list(model.parameters())[i]).any()
            for i in range(len(list(model.parameters())))
        ]
    ):
        raise ValueError("NaN in model parameters")

    if i % 100 == 0:
        print(torch.sigmoid(model(torch.zeros(dim, device=env.device))[0]).item())
        if use_wandb:
            wandb.log(
                {
                    "loss": loss.item(),
                    "logZdiff": logZ.item() - np.log(env.Z),
                    "states_visited": (i + 1) * BS,
                    "termination_prob_at_00": torch.sigmoid(
                        model(torch.zeros(dim, device=env.device))[0]
                    ).item(),
                },
                step=i,
            )
        tqdm.write(
            f"{i}: {loss.item()}, {logZ.item()}, {np.log(env.Z + env.R0)}, {tuple(actionss.shape)}"
        )
    if i % 1000 == 0:
        trajectories, actionss, logprobs = trajectories_sampler.sample(
            n_validation_trajectories
        )
        last_states = get_last_states(env, trajectories)

        if use_wandb:
            colors = plt.cm.rainbow(np.linspace(0, 1, 10))

            plt.close("all")
            fig1 = plt.figure(figsize=(6 * 16 / 9, 6))
            _ = plt.scatter(last_states[:, 0], last_states[:, 1])

            plt.close("all")
            fig2 = plt.figure(figsize=(6 * 16 / 9, 6))
            for j in range(10):
                plt.plot(trajectories[j, :, 0], trajectories[j, :, 1], color=colors[j])

            plt.close("all")
            fig3 = plt.figure(figsize=(6 * 16 / 9, 6))
            n = 100
            x = torch.linspace(0, 1, n)
            y = torch.linspace(0, 1, n)
            xx, yy = torch.meshgrid(x, y)
            states = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

            out = model(states)
            termination_probs = torch.sigmoid(out[:, 0]).reshape(n, n)

            plt.imshow(termination_probs.detach().numpy(), cmap="hot")
            plt.xlim(0, n)
            plt.ylim(0, n)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            wandb.log(
                {
                    "last_states": wandb.Image(fig1),
                    "trajectories": wandb.Image(fig2),
                    "termination_probs": wandb.Image(fig3),
                },
                step=i,
            )

            plt.close("all")

            # import scipy.stats as st

            # kernel = st.gaussian_kde(last_states.numpy())
