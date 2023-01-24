import torch
from torch.distributions import Distribution, Beta
import numpy as np


def sample_actions(
    env, model, states, min_terminate_proba=0.0, max_terminate_proba=1.0, epsilon=0.0
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
            samples_theta[uniform_mask] = torch.rand_like(samples_theta[uniform_mask])

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

    return actions, logprobs


def sample_trajectories(
    env,
    model,
    n_trajectories,
    min_terminate_proba=0.0,
    max_terminate_proba=1.0,
    max_terminate_proba_shift=0.0,
    epsilon=0.0,
):
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
            min_terminate_proba=min_terminate_proba,
            max_terminate_proba=max_terminate_proba + step * max_terminate_proba_shift,
            epsilon=epsilon,
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
            # + torch.log(2.0 / (torch.pi * env.delta * (B - A)))
            # - 0.5 * torch.log(1 - (difference_1 / env.delta) ** 2)
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

    print(trajectories)

    print(logprobs, all_logprobs)

    print(bw_logprobs, all_bw_logprobs)

    log_state_flows = evaluate_state_flows(env, flow, trajectories, logZ)

    print(log_state_flows)

    print("FPF:", all_logprobs + log_state_flows)

    print("FPB", all_bw_logprobs + log_state_flows[:, 1:])

    last_states = get_last_states(env, trajectories)
    print(last_states)
    logrewards = env.reward(last_states).log()
    print("R", logrewards)

    # assert False

    db_preds = all_logprobs + log_state_flows

    db_targets = all_bw_logprobs + log_state_flows[:, 1:]
    db_targets = torch.cat(
        [
            db_targets,
            torch.full(
                (db_targets.shape[0], 1), -float("inf"), device=db_targets.device
            ),
        ],
        dim=1,
    )
    infinity_mask = db_targets == -float("inf")

    _, indices_of_first_inf = torch.max(infinity_mask, dim=1)

    db_targets2 = db_targets.scatter(
        1, indices_of_first_inf.unsqueeze(1), logrewards.unsqueeze(1)
    )

    print(db_preds)
    print(db_targets2)

    flat_db_preds = db_preds[db_preds != -float("inf")]
    flat_db_targets = db_targets2[db_targets2 != -float("inf")]

    print(flat_db_preds)
    print(flat_db_targets)

    assert False
