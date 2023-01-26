import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import argparse

from env import Box, get_last_states
from model import CirclePF, CirclePB, NeuralNet
from sampling import (
    sample_trajectories,
    evaluate_backward_logprobs,
    evaluate_state_flows,
)

from utils import (
    fit_kde,
    plot_reward,
    sample_from_reward,
    plot_samples,
    estimate_jsd,
    plot_trajectories,
    plot_termination_probabilities,
)

try:
    import wandb
except ModuleNotFoundError:
    pass


USE_WANDB = False
NO_PLOT = False

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--delta", type=float, default=0.25)
parser.add_argument("--env_epsilon", type=float, default=1e-10)
parser.add_argument(
    "--n_components",
    type=int,
    default=2,
    help="Number of components in Mixture Of Betas",
)

parser.add_argument("--reward_debug", action="store_true", default=False)

parser.add_argument(
    "--n_components_s0",
    type=int,
    default=4,
    help="Number of components in Mixture Of Betas",
)
parser.add_argument(
    "--beta_min",
    type=float,
    default=0.1,
    help="Minimum value for the concentration parameters of the Beta distribution",
)
parser.add_argument(
    "--beta_max",
    type=float,
    default=5.0,
    help="Maximum value for the concentration parameters of the Beta distribution",
)
parser.add_argument(
    "--loss", type=str, choices=["tb", "db", "modifieddb"], default="tb"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=1.0,
    help="Weight of the reward term in DB",
)
parser.add_argument(
    "--alpha_schedule",
    type=float,
    default=1.0,
    help="every 1000 iterations, divide alpha by this value - the maximum value of alpha is 1.0",
)
parser.add_argument(
    "--PB",
    type=str,
    choices=["learnable", "tied", "uniform"],
    default="learnable",
)
parser.add_argument("--gamma_scheduler", type=float, default=1.0)
parser.add_argument("--scheduler_milestone", type=int, default=5000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_Z", type=float, default=1e-3)
parser.add_argument("--BS", type=int, default=128)
parser.add_argument("--n_iterations", type=int, default=20000)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--n_evaluation_trajectories", type=int, default=10000)
parser.add_argument("--no_plot", action="store_true", default=False)
parser.add_argument("--no_wandb", action="store_true", default=False)
args = parser.parse_args()

if args.no_plot:
    NO_PLOT = True

if args.no_wandb:
    USE_WANDB = False

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
loss_type = args.loss

if seed == 0:
    seed = np.random.randint(int(1e6))

run_name = f"d{delta}_{loss_type}_PB{args.PB}_lr{lr}_lrZ{lr_Z}_sd{seed}"
run_name += f"_n{n_components}_n0{n_components_s0}"
run_name += f"_gamma{args.gamma_scheduler}_mile{args.scheduler_milestone}"
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
)

# Get the true KDE
samples = sample_from_reward(env, n_samples=10000)
true_kde, fig1 = fit_kde(samples, plot=True)

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
if args.loss == "db":
    flow_model = NeuralNet(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        torso=model.torso,
        output_dim=1,
    )


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

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[i * args.scheduler_milestone for i in range(1, 10)],
    gamma=args.gamma_scheduler,
)

jsd = float("inf")

current_alpha = args.alpha * args.alpha_schedule

for i in trange(n_iterations):
    if i % 1000 == 0:
        current_alpha = max(current_alpha / args.alpha_schedule, 1.0)
        print(f"current optimizer LR: {optimizer.param_groups[0]['lr']}")

    optimizer.zero_grad()
    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(
        env,
        model,
        BS,
    )
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(
        env, bw_model, trajectories
    )

    if loss_type == "tb":
        loss = torch.mean((logZ + logprobs - bw_logprobs - logrewards) ** 2)
    elif loss_type == "modifieddb":
        exits = torch.full(
            (trajectories.shape[0], trajectories.shape[1] - 1), -float("inf")
        )
        msk = torch.all(trajectories[:, 1:] != -float("inf"), dim=-1)
        middle_states = trajectories[:, 1:][msk]
        exit_proba, _ = model.to_dist(middle_states)
        true_exit_log_probs = torch.zeros_like(exit_proba)  # type: ignore
        edgy_middle_states_mask = torch.norm(1 - middle_states, dim=-1) <= env.delta
        other_edgy_middle_states_mask = torch.any(
            middle_states >= 1 - env.epsilon, dim=-1
        )
        true_exit_log_probs[edgy_middle_states_mask] = 0
        true_exit_log_probs[other_edgy_middle_states_mask] = 0
        true_exit_log_probs[
            ~edgy_middle_states_mask & ~other_edgy_middle_states_mask
        ] = torch.log(
            exit_proba[~edgy_middle_states_mask & ~other_edgy_middle_states_mask]  # type: ignore
        )

        exits[msk] = true_exit_log_probs
        exits = torch.cat([torch.zeros((trajectories.shape[0], 1)), exits], dim=1)
        non_infinity_mask = all_logprobs != -float("inf")
        _, indices = torch.max(non_infinity_mask.flip(1), dim=1)
        indices = all_logprobs.shape[1] - indices - 1
        new_all_logprobs = all_logprobs.scatter(1, indices.unsqueeze(1), -float("inf"))

        all_log_rewards = torch.full(
            (trajectories.shape[0], trajectories.shape[1] - 1), -float("inf")
        )
        log_rewards = env.reward(trajectories[:, 1:][msk]).log()
        all_log_rewards[msk] = log_rewards

        all_log_rewards = torch.cat(
            [logZ * torch.ones((trajectories.shape[0], 1)), all_log_rewards], dim=1
        )
        preds = new_all_logprobs[:, :-1] + exits[:, 1:-1] + all_log_rewards[:, :-2]
        targets = all_bw_logprobs + exits[:, :-2] + all_log_rewards[:, 1:-1]
        flat_preds = preds[preds != -float("inf")]
        flat_targets = targets[targets != -float("inf")]
        loss = torch.mean((flat_preds - flat_targets) ** 2)
    elif loss_type == "db":
        log_state_flows = evaluate_state_flows(env, flow_model, trajectories, logZ)  # type: ignore
        db_preds = all_logprobs + log_state_flows
        db_targets = all_bw_logprobs + log_state_flows[:, 1:]
        if args.alpha == 1.0:
            db_targets = torch.cat(
                [
                    db_targets,
                    torch.full(
                        (db_targets.shape[0], 1),
                        -float("inf"),
                        device=db_targets.device,
                    ),
                ],
                dim=1,
            )
            infinity_mask = db_targets == -float("inf")
            _, indices_of_first_inf = torch.max(infinity_mask, dim=1)
            db_targets = db_targets.scatter(
                1, indices_of_first_inf.unsqueeze(1), logrewards.unsqueeze(1)
            )
            flat_db_preds = db_preds[db_preds != -float("inf")]
            flat_db_targets = db_targets[db_targets != -float("inf")]
            loss = torch.mean((flat_db_preds - flat_db_targets) ** 2)
        else:
            non_infinity_mask = db_preds.flip(1) != -float("inf")
            _, reverse_indices_of_last_non_inf = torch.max(non_infinity_mask, dim=1)
            indices_of_last_non_inf = (
                db_preds.shape[1] - 1 - reverse_indices_of_last_non_inf
            )
            db_preds_rewards = db_preds.gather(1, indices_of_last_non_inf.unsqueeze(1))
            db_preds2 = db_preds.scatter(
                1, indices_of_last_non_inf.unsqueeze(1), -float("inf")
            )
            flat_db_preds = db_preds2[db_preds2 != -float("inf")]
            flat_db_targets = db_targets[db_targets != -float("inf")]
            loss = torch.mean(
                (flat_db_preds - flat_db_targets) ** 2
            ) + current_alpha * torch.mean((db_preds_rewards - logrewards) ** 2)

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
    scheduler.step()

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
            # Loss with 3 digits of precision, logZ with 2 digits of precision, true logZ with 2 digits of precision
            # Last computed JSD with 4 digits of precision
            f"Loss: {loss.item():.3f}, logZ: {logZ.item():.2f}, true logZ: {np.log(env.Z):.2f}, JSD: {jsd:.4f}"
        )
    if i % 500 == 0:
        trajectories, _, _, _ = sample_trajectories(
            env, model, args.n_evaluation_trajectories
        )
        last_states = get_last_states(env, trajectories)
        kde, fig4 = fit_kde(last_states, plot=True)
        jsd = estimate_jsd(kde, true_kde)

        if USE_WANDB:
            if NO_PLOT:
                wandb.log(
                    {
                        "JSD": jsd,
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
