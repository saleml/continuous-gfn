import torch
from torchtyping import TensorType


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
        reward_cos=False,
        device_str="cpu",
        verify_actions=True,
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
        self.reward_cos = reward_cos

    def is_actions_valid(
        self, states: TensorType["n", "dim"], actions: TensorType["n", "dim"]
    ) -> TensorType["n", torch.bool]:
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

    def is_terminal_action_mask(
        self, actions: TensorType["n", "dim"]
    ) -> TensorType["n", torch.bool]:
        """Return a mask of terminal actions."""
        return torch.all(actions == self.terminal_action, dim=-1)

    def step(
        self, states: TensorType["n", "dim"], actions: TensorType["n", "dim"]
    ) -> TensorType["n", "dim"]:
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

    def reward(self, final_states: TensorType["n", "dim"]) -> TensorType["n"]:
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states - 0.5)
        if not self.reward_cos:
            reward = (
                R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
            )
        else:
            pdf_input = ax * 5
            pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
            reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        return reward

    @property
    def Z(self):
        if not self.reward_cos:
            return (
                self.R0
                + (2 * 0.25) ** self.dim * self.R1
                + (2 * 0.1) ** self.dim * self.R2
            )
        else:
            return self.R0 + self.R1 * 0.1973 ** 2  # 0.1973 is the integral of the pdf, evaluated with Wolfram Alpha


def get_last_states(
    env: Box, trajectories: TensorType["n_traj", "n", "dim"]
) -> TensorType["n_traj", "dim"]:
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
        