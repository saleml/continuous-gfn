import torch
from torch.distributions import Distribution, Beta
import numpy as np


def sample_actions(env, model, states):
    # states is a tensor of shape (n, dim)
    batch_size = states.shape[0]
    out = model.to_dist(states)
    if isinstance(out[0], Distribution):  # s0 input
        dist_r, dist_theta = out
        samples_r = dist_r.sample(torch.Size((batch_size,)))
        samples_theta = dist_theta.sample(torch.Size((batch_size,)))

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
            - torch.log(samples_r * env.delta)
            - np.log(np.pi / 2)
            - np.log(env.delta)  # why ?
        )
    else:
        exit_proba, dist = out

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
        )

        actions[exit] = -float("inf")
        logprobs[exit] = torch.log(exit_proba[exit])
        logprobs[torch.norm(1 - states, dim=1) <= env.delta] = 0.0
        logprobs[torch.any(states >= 1 - env.epsilon, dim=-1)] = 0.0

    return actions, logprobs


def sample_trajectories(env, model, n_trajectories):
    step = 0
    states = torch.zeros((n_trajectories, env.dim), device=env.device)
    actionss = []
    trajectories = [states]
    trajectories_logprobs = torch.zeros((n_trajectories,), device=env.device)
    all_logprobs = []
    while not torch.all(states == env.sink_state):
        step_logprobs = torch.full((n_trajectories,), -float("inf"), device=env.device)
        non_terminal_mask = torch.all(states != env.sink_state, dim=-1)
        actions = torch.full(
            (n_trajectories, env.dim), -float("inf"), device=env.device
        )
        non_terminal_actions, logprobs = sample_actions(
            env,
            model,
            states[non_terminal_mask],
        )
        actions[non_terminal_mask] = non_terminal_actions.reshape(-1, env.dim)
        actionss.append(actions)
        states = env.step(states, actions)
        trajectories.append(states)
        trajectories_logprobs[non_terminal_mask] += logprobs
        step_logprobs[non_terminal_mask] = logprobs
        all_logprobs.append(step_logprobs)
        step += 1
    trajectories = torch.stack(trajectories, dim=1)
    actionss = torch.stack(actionss, dim=1)
    all_logprobs = torch.stack(all_logprobs, dim=1)
    return trajectories, actionss, trajectories_logprobs, all_logprobs


def evaluate_backward_logprobs(env, model, trajectories):
    logprobs = torch.zeros((trajectories.shape[0],), device=env.device)
    all_logprobs = []
    for i in range(trajectories.shape[1] - 2, 1, -1):
        all_step_logprobs = torch.full(
            (trajectories.shape[0],), -float("inf"), device=env.device
        )
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
        )

        if torch.any(torch.isnan(step_logprobs)):
            raise ValueError("NaN in backward logprobs")

        if torch.any(torch.isinf(step_logprobs)):
            raise ValueError("Inf in backward logprobs")

        logprobs[non_sink_mask] += step_logprobs
        all_step_logprobs[non_sink_mask] = step_logprobs

        all_logprobs.append(all_step_logprobs)

    all_logprobs.append(torch.zeros((trajectories.shape[0],), device=env.device))
    all_logprobs = torch.stack(all_logprobs, dim=1)

    return logprobs, all_logprobs.flip(1)


def evaluate_state_flows(env, model, trajectories, logZ):
    state_flows = torch.full(
        (trajectories.shape[0], trajectories.shape[1]),
        -float("inf"),
        device=trajectories.device,
    )
    non_sink_mask = torch.all(trajectories != env.sink_state, dim=-1)
    state_flows[non_sink_mask] = model(trajectories[non_sink_mask]).squeeze(-1)
    state_flows[:, 0] = logZ

    return state_flows[:, :-1]


if __name__ == "__main__":
    from model import CirclePF, CirclePB, NeuralNet
    from env import Box, get_last_states

    env = Box(dim=2, delta=0.25)

    model = CirclePF()
    bw_model = CirclePB()

    flow = NeuralNet(output_dim=1)

    logZ = torch.zeros(1, requires_grad=True)

    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(env, model, 5)

    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(
        env, bw_model, trajectories
    )

    exits = torch.full(
        (trajectories.shape[0], trajectories.shape[1] - 1), -float("inf")
    )
    msk = torch.all(trajectories[:, 1:] != -float("inf"), dim=-1)
    middle_states = trajectories[:, 1:][msk]
    exit_proba, _ = model.to_dist(middle_states)
    true_exit_log_probs = torch.zeros_like(exit_proba)  # type: ignore
    edgy_middle_states_mask = torch.norm(1 - middle_states, dim=-1) <= env.delta
    other_edgy_middle_states_mask = torch.any(middle_states >= 1 - env.epsilon, dim=-1)
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
