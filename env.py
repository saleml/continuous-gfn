import torch
from torch.distributions import MultivariateNormal


class Box:
    """D-dimensional box with lower bound 0 and upper bound 1. A maximum step size 0<delta<1 defines
    the maximum unidimensional step size in each dimension.
    """

    def __init__(
        self,
        dim=2,
        delta=0.1,
        epsilon=1e-4,
        R0=0.1,
        R1=0.5,
        R2=2.0,
        reward_debug=False,
        device_str="cpu",
        verify_actions=False,
    ):
        # Set verify_actions to False to disable action verification for faster step execution.
        self.dim = dim
        self.delta = delta
        self.epsilon = epsilon
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.terminal_action = torch.full((dim,), -float("inf"), device=self.device)
        self.sink_state = torch.full((dim,), -float("inf"), device=self.device)
        self.verify_actions = verify_actions

        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reward_debug = reward_debug

    def is_actions_valid(self, states, actions
    ):
        """Check if actions are valid: First, verify that no state component is within epsilon distance from the bounds,
        then for each state [x_1, ..., x_d], the action [a_1, ..., a_d] needs to satisfy
        0 <= a_i < min(self.delta_max, 1 - x_i) for all i. Assume all actions are non terminal. Basically, this means
        that if one coordinate is >= 1 - self.epsilon, then the corresponding action should be "exit"."""
        first_condition = torch.all(
            torch.logical_and(
                states >= 0,
                states <= 1 - self.epsilon,
            )
        )

        second_condition = torch.all(
            torch.logical_and(
                actions >= 0,
                actions
                <= torch.min(
                    torch.full((self.dim,), self.delta, device=self.device),
                    1 - states,
                ),
            )
        )
        out = first_condition and second_condition
        return out

    def is_terminal_action_mask(self, actions):
        """Return a mask of terminal actions."""
        return torch.all(actions == self.terminal_action, dim=-1)

    def step(self, states, actions) :
        """Take a step in the environment. The states can include the sink state [-inf, ..., -inf].
        In which case, the corresponding actions are ignored."""
        # First, select the states that are not the sink state.
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        non_sink_states = states[non_sink_mask]
        non_sink_actions = actions[non_sink_mask]
        # Then, select states and actions not corresponding to terminal actions, for the non sink states and actions.
        non_terminal_mask = ~self.is_terminal_action_mask(non_sink_actions)
        non_terminal_states = non_sink_states[non_terminal_mask]
        non_terminal_actions = non_sink_actions[non_terminal_mask]
        # Then, if verify_actions is True, check if actions are valid.
        if self.verify_actions:
            assert self.is_actions_valid(non_terminal_states, non_terminal_actions)
        # Then, take a step and store that in a new tensor.
        new_states = torch.full_like(states, -float("inf"))
        non_sink_new_states = new_states[non_sink_mask]
        non_sink_new_states[non_terminal_mask] = (
            non_terminal_states + non_terminal_actions
        )
        new_states[non_sink_mask] = non_sink_new_states
        # Finally, return the new states.
        return new_states

    def reward(self, final_states):
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states - 0.5)
        if not self.reward_debug:
            reward = (
                R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
            )
        elif self.reward_debug:
            reward = torch.ones(final_states.shape[0], device=self.device)
            reward[final_states.norm(dim=-1) > self.delta] = 1e-8
        else:
            raise NotImplementedError

        return reward

    @property
    def Z(self):
        if not self.reward_debug:
            return (
                self.R0
                + (2 * 0.25) ** self.dim * self.R1
                + (2 * 0.1) ** self.dim * self.R2
            )
        else:
            if self.dim != 2:
                raise NotImplementedError("Only implemented for dim=2")
            return torch.pi * self.delta ** 2 / 4.


def get_last_states(env: Box, trajectories):
    """Get last states from trajectories.
    Args:
        trajectories: A tensor of trajectories
    Returns:
        last_states: A tensor of last states
    """
    non_sink = ~torch.all(trajectories == env.sink_state, dim=-1)

    mask = torch.zeros_like(non_sink).bool()
    mask.scatter_(1, non_sink.cumsum(dim=-1).argmax(dim=-1, keepdim=True), True)

    return trajectories[mask]
        