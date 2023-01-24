import json
import os
import torch
from torch.distributions import Distribution, Beta
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
NO_PLOT = False

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--delta", type=float, default=0.25)
parser.add_argument("--env-epsilon", type=float, default=1e-10)
parser.add_argument(
    "--n_components",
    type=int,
    default=2,
    help="Number of components in Mixture Of Betas",
)
parser.add_argument("--reward_debug", action="store_true", default=False)
parser.add_argument("--reward_gaussian", action="store_true", default=False)

parser.add_argument(
    "--n_components_s0",
    type=int,
    default=4,
    help="Number of components in Mixture Of Betas",
)
parser.add_argument(
    "--beta-min",
    type=float,
    default=0.1,
    help="Minimum value for the concentration parameters of the Beta distribution",
)
parser.add_argument(
    "--beta-max",
    type=float,
    default=5.0,
    help="Maximum value for the concentration parameters of the Beta distribution",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.0,
    help="OFF-POLICY: with proba epsilon, sample from the uniform distribution in the quarter circle",
)
parser.add_argument(
    "--min-terminate-proba",
    type=float,
    default=0.0,
    help="OFF-POLICY: all terminating probabilities below this value are set to this value",
)
parser.add_argument(
    "--max-terminate-proba",
    type=float,
    default=1.0,
    help="OFF-POLICY: all terminating probabilities above this value are set to this value",
)
parser.add_argument(
    "--max-terminate-proba-shift",
    type=float,
    default=0.0,
    help="OFF-POLICY: after each action, add this shift to max-terminate-proba (so that it gets to 1 eventually)",
)
parser.add_argument(
    "--directed-exploration",
    action="store_true",
    default=False,
    help="If true, use a Beta(0.5, 0.5) instead of uniform to temper",
)
parser.add_argument("--loss", type=str, choices=["tb", "hvi", "soft"], default="tb")
parser.add_argument(
    "--PB",
    type=str,
    choices=["learnable", "tied", "uniform"],
    default="learnable",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_Z", type=float, default=5e-2)
parser.add_argument("--BS", type=int, default=128)
parser.add_argument("--n_iterations", type=int, default=20000)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--n_evaluation_trajectories", type=int, default=10000)
parser.add_argument("--no_plot", action="store_true", default=False)
args = parser.parse_args()

if args.no_plot:
    NO_PLOT = True

if USE_WANDB:
    wandb.init(project="continuous_gflownets", save_code=True)
    wandb.config.update(args)

dim = args.dim
delta = args.delta
seed = args.seed
lr = args.lr
lr_Z = args.lr_Z
n_iterations = args.n_iterations
BS = args.BS
n_components = args.n_components
n_components_s0 = args.n_components_s0
epsilon = args.epsilon
min_terminate_proba = args.min_terminate_proba
max_terminate_proba = args.max_terminate_proba
max_terminate_proba_shift = args.max_terminate_proba_shift
loss_type = args.loss

if seed == 0:
    seed = np.random.randint(int(1e6))

run_name = f"d{delta}_{loss_type}_PB{args.PB}_lr{lr}_lrZ{lr_Z}_sd{seed}"
run_name += f"_n{n_components}_n0{n_components_s0}_eps{epsilon}{'_dir' if args.directed_exploration else ''}"
run_name += f"_min{min_terminate_proba}_max{max_terminate_proba}_shift{max_terminate_proba_shift}"
print(run_name)
if USE_WANDB:
    wandb.run.name = run_name  # type: ignore

torch.manual_seed(seed)
np.random.seed(seed)

env = Box(
    dim=dim,
    delta=delta,
    epsilon=args.env_epsilon,
    device_str="cpu",
    reward_debug=args.reward_debug,
    reward_gaussian=args.reward_gaussian,
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


model = CirclePF(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    n_components=n_components,
    n_components_s0=n_components_s0,
    one_component=n_components == 1 and n_components_s0 == 1,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
)
bw_model = CirclePB(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    torso=model.torso if args.PB == "tied" else None,
    uniform=args.PB == "uniform",
    n_components=n_components,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
)


def sample_actions(
    model, states, min_terminate_proba=0.0, max_terminate_proba=1.0, epsilon=0.0
):
    # with probability epsilon, sample uniformly in the quarter circle
    # states is a tensor of shape (n, dim)
    batch_size = states.shape[0]
    out = model.to_dist(states)
    if isinstance(out[0], Distribution):  # s0 input
        dist_r, dist_theta = out
        samples_r = dist_r.sample(torch.Size((batch_size,)))
        samples_theta = dist_theta.sample(torch.Size((batch_size,)))
        if epsilon > 0:
            uniform_mask = torch.rand(batch_size) < epsilon
            samples_r[uniform_mask] = torch.rand_like(samples_r[uniform_mask])
            if args.directed_exploration:
                exploration_distribution = torch.distributions.Beta(
                    torch.tensor([0.5]), torch.tensor([0.5])
                )
                samples_theta[uniform_mask] = exploration_distribution.sample(
                    torch.Size((samples_r[uniform_mask].shape[0],))
                ).squeeze()
            else:
                samples_theta[uniform_mask] = torch.rand_like(
                    samples_theta[uniform_mask]
                )

        actions = (
            torch.stack(
                [
                    samples_r * torch.cos(torch.pi / 2.0 * samples_theta),
                    samples_r * torch.sin(torch.pi / 2.0 * samples_theta),
                ],
                dim=1,
            )
            * env.delta
        )
        logprobs = (
            dist_r.log_prob(samples_r)
            + dist_theta.log_prob(samples_theta)
            # + np.log(4 / (env.delta**2 * np.pi))  # debugging
            - torch.log(samples_r * env.delta)
            - np.log(np.pi / 2)
            - np.log(env.delta)  # why ?
        )
        # print("logprobs", logprobs.exp().mean(), logprobs.exp().std())
        # print(
        #     (2 * dist_r.log_prob(samples_r).exp() / samples_r).mean(),  # pdf of radius for uniform in disk
        #     (2 * dist_r.log_prob(samples_r).exp() / samples_r).std(),
        # )
        # print(
        #     dist_theta.log_prob(samples_theta).exp().mean(),
        #     dist_theta.log_prob(samples_theta).exp().std(),
        # )
        if n_components == 1 and n_components_s0 == 1:
            entropies = dist_r.entropy() + dist_theta.entropy()
        else:
            entropies = torch.zeros(
                batch_size
            )  # the previous line is not implemented in general
    else:
        exit_proba, dist = out
        exit_proba = exit_proba.clamp(min_terminate_proba, max_terminate_proba)

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
        samples = dist.sample()
        if epsilon > 0:
            uniform_mask = torch.rand(batch_size) < epsilon
            if args.directed_exploration:
                exploration_distribution = torch.distributions.Beta(
                    torch.tensor([0.5]), torch.tensor([0.5])
                )
                samples[uniform_mask] = exploration_distribution.sample(
                    torch.Size((samples[uniform_mask].shape[0],))
                ).squeeze()
            else:
                samples[uniform_mask] = torch.rand_like(samples[uniform_mask])
        actions = samples * (B - A) + A
        actions *= torch.pi / 2.0
        actions = (
            torch.stack([torch.cos(actions), torch.sin(actions)], dim=1) * env.delta
        )

        logprobs = (
            dist.log_prob(samples)
            + torch.log(1 - exit_proba)
            - np.log(env.delta)
            - np.log(np.pi / 2)
            - torch.log(B - A)
            # + torch.log(2.0 / (torch.pi * env.delta * (B - A)))
            # - 0.5 * torch.log(1 - torch.cos(actions[:, 0]) ** 2)
        )

        actions[exit] = -float("inf")
        logprobs[exit] = torch.log(exit_proba[exit])
        logprobs[torch.norm(1 - states, dim=1) <= env.delta] = 0.0
        logprobs[torch.any(states >= 1 - env.epsilon, dim=-1)] = 0.0

        if n_components == 1 and n_components_s0 == 1:
            entropies = (
                -exit_proba * torch.log(exit_proba)
                - (1 - exit_proba) * torch.log(1 - exit_proba)
                + dist.entropy() * (1 - exit_proba)
            )
        else:
            entropies = torch.zeros(
                batch_size
            )  # the previous line is not implemented in general

    return actions, logprobs, entropies


def sample_trajectories(
    model, n_trajectories, min_terminate_proba=0.0, max_terminate_proba=1.0, epsilon=0.0
):
    step = 0
    states = torch.zeros((n_trajectories, env.dim), device=env.device)
    actionss = []
    trajectories = [states]
    trajectories_logprobs = torch.zeros((n_trajectories,), device=env.device)
    trajectories_entropies = torch.zeros((n_trajectories,), device=env.device)
    while not torch.all(states == env.sink_state):
        non_terminal_mask = torch.all(states != env.sink_state, dim=-1)
        actions = torch.full(
            (n_trajectories, env.dim), -float("inf"), device=env.device
        )
        non_terminal_actions, logprobs, entropies = sample_actions(
            model,
            states[non_terminal_mask],
            min_terminate_proba=min_terminate_proba,
            max_terminate_proba=max_terminate_proba + step * max_terminate_proba_shift,
            epsilon=epsilon,
        )
        actions[non_terminal_mask] = non_terminal_actions.reshape(-1, env.dim)
        actionss.append(actions)
        states = env.step(states, actions)
        trajectories.append(states)
        trajectories_logprobs[non_terminal_mask] += logprobs
        trajectories_entropies[non_terminal_mask] += entropies
        step += 1
    trajectories = torch.stack(trajectories, dim=1)
    actionss = torch.stack(actionss, dim=1)
    return trajectories, actionss, trajectories_logprobs, trajectories_entropies


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

        dist = model.to_dist(current_states)

        step_logprobs = (
            dist.log_prob(
                (
                    1.0
                    / (B - A)
                    * (2.0 / torch.pi * torch.acos(difference_1 / env.delta) - A)
                ).clamp(1e-4, 1 - 1e-4)
            ).clamp_max(100)
            - np.log(env.delta)
            - np.log(np.pi / 2)
            - torch.log(B - A)
            # + torch.log(2.0 / (torch.pi * env.delta * (B - A)))
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
    current_epsilon = epsilon
    current_min_terminate_proba = min_terminate_proba
    current_max_terminate_proba = max_terminate_proba
    optimizer.zero_grad()
    trajectories, actionss, logprobs, entropies = sample_trajectories(
        model,
        BS,
        epsilon=current_epsilon,
        min_terminate_proba=current_min_terminate_proba,
        max_terminate_proba=current_max_terminate_proba,
    )
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs = evaluate_backward_logprobs(bw_model, trajectories)

    if loss_type == "tb":
        loss = torch.mean((logZ + logprobs - bw_logprobs - logrewards) ** 2)
    elif loss_type == "hvi":
        scores = logprobs - bw_logprobs - logrewards
        baseline = scores.mean().detach()
        loss = logprobs * (scores.detach() - baseline) - bw_logprobs
        loss = torch.mean(loss)
    elif loss_type == "soft":
        entropic_rewards = logrewards + entropies.detach() * 0.01
        loss = -logprobs * entropic_rewards
        loss = torch.mean(loss)
    else:
        raise ValueError("Unknown loss type")
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
                    "logZdiff": np.log(env.Z) - logZ.item(),
                    "states_visited": (i + 1) * BS,
                },
                step=i,
            )
        tqdm.write(
            f"{i}: {loss.item()}, {logZ.item()}, {np.log(env.Z)}, {jsd}, {tuple(trajectories.shape)}"
        )
    if i % 1000 == 0:
        # print(list(model.PFs0.values()))
        trajectories, actionss, logprobs, entropies = sample_trajectories(
            model, args.n_evaluation_trajectories
        )
        last_states = get_last_states(env, trajectories)
        kde, fig4 = fit_kde(last_states)
        jsd = estimate_jsd(kde, true_kde)

        if USE_WANDB:
            if NO_PLOT:
                wandb.log(
                    {
                        "jsd": jsd,
                    },
                    step=i,
                )
            else:
                colors = plt.cm.rainbow(np.linspace(0, 1, 10))

                fig1 = plot_samples(last_states[:2000].detach().cpu().numpy())
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

# Save model and arguments as JSON
save_path = os.path.join("saved_models", run_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    torch.save(bw_model.state_dict(), os.path.join(save_path, "bw_model.pt"))
    torch.save(logZ, os.path.join(save_path, "logZ.pt"))
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f)
