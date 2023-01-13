from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from torchtyping import TensorType
from env import Box
from model import NeuralNet
from TruncatedNormal import TruncatedNormal


class ActionsSampler(ABC):
    def __init__(self, env: Box, epsilon_uniform=0.0):
        self.delta = env.delta
        self.env = env
        self.device = env.device
        self.epsilon_uniform = epsilon_uniform

    @abstractmethod
    def sample_pre_actions(
        self, states: TensorType["n", "dim"]
    ) -> Tuple[TensorType["n", "dim"], TensorType["n"]]:
        pass

    def sample(
        self, states: TensorType["n", "dim"]
    ) -> Tuple[TensorType["n", "dim"], TensorType["n"]]:
        actions, logprobs = self.sample_pre_actions(states)
        actions *= torch.min(
            torch.full_like(actions, self.delta),
            1 - states,
        )
        actions[torch.any(states > 1 - self.env.epsilon, dim=-1)] = torch.full_like(
            actions[0], -float("inf")
        )
        return actions, logprobs


class RandomActionsSampler(ActionsSampler):
    def __init__(self, env: Box, terminal_action_prob=0.2):
        super().__init__(env)
        self.dim = env.dim
        self.sink_state = env.sink_state
        self.terminal_action_prob = terminal_action_prob

    def sample_pre_actions(self, states):
        actions = torch.full_like(states, -float("inf"))
        logprobs = torch.zeros_like(states[:, 0])  # Not really needed
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        terminating_states_mask = (
            torch.rand_like(states[:, 0]) < self.terminal_action_prob
        )
        non_sink_and_non_terminating_mask = non_sink_mask & ~terminating_states_mask
        valid_states = states[non_sink_and_non_terminating_mask]
        actions[non_sink_and_non_terminating_mask] = torch.rand_like(valid_states)

        return actions, logprobs


class TrajectoriesSampler:
    def __init__(self, env: Box, sampler: ActionsSampler) -> None:
        self.env = env
        self.sampler = sampler

    def sample(
        self, n_trajectories: int
    ) -> Tuple[
        TensorType["n_trajectories", "n", "dim"],
        TensorType["n_trajectories", "n-1", "dim"],
        TensorType["n_trajectories"],
    ]:
        states = torch.zeros((n_trajectories, self.env.dim), device=self.env.device)
        actionss = []
        trajectories = [states]
        trajectories_logprobs = torch.zeros((n_trajectories,), device=self.env.device)
        while not torch.all(states == self.env.sink_state):
            actions, logprobs = self.sampler.sample(states)
            actionss.append(actions)
            states = self.env.step(states, actions)
            trajectories.append(states)
            trajectories_logprobs = trajectories_logprobs + logprobs
        trajectories = torch.stack(trajectories, dim=1)
        actionss = torch.stack(actionss, dim=1)
        return trajectories, actionss, trajectories_logprobs


class ParametrizedDistributionActionsSampler(ActionsSampler, ABC):
    def __init__(
        self, env: Box, model: NeuralNet, epsilon=1e-4, sf_bias=0.0, epsilon_uniform=0.0
    ):
        super().__init__(env, epsilon_uniform=epsilon_uniform)
        self.dim = env.dim
        self.sink_state = env.sink_state
        self.model = model
        self.epsilon = epsilon  # to ensure terminating in finite time a.s. (min value for P_F(sf|s))
        self.sf_bias = (
            sf_bias  # to subtract to presigmoid of P_F(sf|s) to allow for longer paths
        )
        assert model.output_dim == self.expected_model_output_dim

    @property
    @abstractmethod
    def expected_model_output_dim(self) -> int:
        pass

    @abstractmethod
    def output_to_distribution_parameters(self, out) -> List:
        pass

    @abstractmethod
    def distribution_from_parameters(
        self, parameters
    ) -> torch.distributions.Distribution:
        # Should be a batched distribution
        pass

    def sample_pre_actions(self, states):
        actions = torch.full_like(states, -float("inf"))
        logprobs = torch.zeros_like(states[:, 0])
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        non_sink_states = states[non_sink_mask]
        non_sink_actions = actions[non_sink_mask]
        non_sink_logprobs = logprobs[non_sink_mask]
        out = self.model(non_sink_states)
        terminate_pre_sigmoid_prob = out[..., 0]
        terminate_pre_sigmoid_prob[
            torch.any(non_sink_states > 1 - self.env.epsilon, dim=-1)
        ] = float("inf")
        terminate_prob = torch.sigmoid(terminate_pre_sigmoid_prob - 2)
        terminate_prob = terminate_prob.clamp_min(self.epsilon)
        # The following line is kind of off policy sampling
        terminating_states_mask = torch.rand_like(terminate_prob) < torch.sigmoid(
            terminate_pre_sigmoid_prob - 2 - self.sf_bias
        ).clamp_min(self.epsilon)
        non_sink_logprobs[terminating_states_mask] = torch.log(
            terminate_prob[terminating_states_mask]
        )
        if not torch.all(terminating_states_mask):
            parameters = self.output_to_distribution_parameters(
                out[~terminating_states_mask][..., 1:]
            )

            distribution = self.distribution_from_parameters(parameters)

            uniform_actions_mask = (
                torch.rand_like(parameters[0][:, 0]) < self.epsilon_uniform
            )
            uniform_actions = torch.rand_like(parameters[0])
            sampled_actions = distribution.sample()
            sampled_actions[uniform_actions_mask] = uniform_actions[
                uniform_actions_mask
            ]

            non_sink_actions[~terminating_states_mask] = sampled_actions
            non_sink_logprobs[~terminating_states_mask] = distribution.log_prob(
                sampled_actions
            ) + torch.log(1 - terminate_prob[~terminating_states_mask])

        actions[non_sink_mask] = non_sink_actions
        logprobs[non_sink_mask] = non_sink_logprobs

        return actions, logprobs

    def evaluate_backward_logprobs(
        self,
        bw_model: NeuralNet,
        trajectories: TensorType["n_traj", "n", "dim"],
        actionss: TensorType["n_traj", "n-1", "dim"],
    ):
        assert bw_model.output_dim == self.expected_model_output_dim

        if trajectories.shape[1] == 2:  # only start and end states
            return torch.zeros(trajectories.shape[0], device=self.device)

        backward_logprobs = torch.zeros_like(trajectories[:, 1:-1, 0])
        non_sink_mask = ~torch.all(trajectories[:, 1:-1, :] == self.sink_state, dim=-1)
        non_sink_states = trajectories[:, 1:-1, :][non_sink_mask]
        out = bw_model(non_sink_states)
        terminate_pre_sigmoid_prob = out[..., 0]

        terminate_pre_sigmoid_prob[
            torch.any(non_sink_states > self.delta, dim=-1)
        ] = -float("inf")

        terminate_prob = torch.sigmoid(terminate_pre_sigmoid_prob)

        s1_indices = (
            torch.cat(
                (
                    torch.zeros(1, device=self.device),
                    non_sink_mask.sum(dim=-1).cumsum(dim=-1).unique()[:-1],
                )
            )
            .unique()
            .long()
        )

        backward_logprobs[:, 0][non_sink_mask[:, 0]] = torch.log(
            terminate_prob[s1_indices]
        )

        s1_mask = torch.zeros_like(out[:, 0]).bool()
        s1_mask[s1_indices] = True

        if not torch.all(s1_mask):
            parameters = self.output_to_distribution_parameters(out[~s1_mask][..., 1:])

            distribution = self.distribution_from_parameters(parameters)

            actions = actionss[:, 1:][
                ~torch.all(actionss[:, 1:] == self.sink_state, dim=-1)
            ]

            pre_actionss = actions / torch.min(
                torch.full_like(actions, self.delta),
                trajectories[:, 1:-1, :][non_sink_mask][~s1_mask],
            )

            backward_logprobs[:, 1:][non_sink_mask[:, 1:]] = distribution.log_prob(
                pre_actionss
            ) + torch.log(1 - terminate_prob[~s1_mask])

        return backward_logprobs.sum(dim=-1)


class BetaActionsSampler(ActionsSampler):
    def __init__(
        self, env: Box, model: NeuralNet, epsilon=1e-4, sf_bias=0.0, epsilon_uniform=0.0
    ):
        super().__init__(env, epsilon_uniform=epsilon_uniform)
        self.dim = env.dim
        self.sink_state = env.sink_state
        self.model = model
        self.epsilon = epsilon  # to ensure terminating in finite time a.s. (min value for P_F(sf|s))
        self.sf_bias = (
            sf_bias  # to subtract to presigmoid of P_F(sf|s) to allow for longer paths
        )
        assert model.output_dim == 1 + 2 * self.dim

    def sample_pre_actions(self, states):
        actions = torch.full_like(states, -float("inf"))
        logprobs = torch.zeros_like(states[:, 0])
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        non_sink_states = states[non_sink_mask]
        non_sink_actions = actions[non_sink_mask]
        non_sink_logprobs = logprobs[non_sink_mask]
        out = self.model(non_sink_states)
        terminate_pre_sigmoid_prob = out[..., 0]
        terminate_pre_sigmoid_prob[
            torch.any(non_sink_states > 1 - self.env.epsilon, dim=-1)
        ] = float("inf")
        terminate_prob = torch.sigmoid(terminate_pre_sigmoid_prob - 2)
        terminate_prob = terminate_prob.clamp_min(self.epsilon)
        # The following line is kind of off policy sampling
        terminating_states_mask = torch.rand_like(terminate_prob) < torch.sigmoid(
            terminate_pre_sigmoid_prob - 2 - self.sf_bias
        ).clamp_min(self.epsilon)
        non_sink_logprobs[terminating_states_mask] = torch.log(
            terminate_prob[terminating_states_mask]
        )
        if not torch.all(terminating_states_mask):
            log_alphas = out[..., 1 : self.dim + 1]
            log_betas = out[..., self.dim + 1 :]
            log_alphas = log_alphas[~terminating_states_mask]
            log_betas = log_betas[~terminating_states_mask]

            alphas = torch.exp(log_alphas).clamp(0.1, 5.0)
            betas = torch.exp(log_betas).clamp(0.1, 5.0)

            distribution = torch.distributions.Beta(alphas, betas)
            distribution = torch.distributions.Independent(distribution, 1)

            uniform_actions_mask = torch.rand_like(alphas[:, 0]) < self.epsilon_uniform
            uniform_actions = torch.rand_like(alphas)
            sampled_actions = distribution.sample()
            sampled_actions[uniform_actions_mask] = uniform_actions[
                uniform_actions_mask
            ]

            non_sink_actions[~terminating_states_mask] = sampled_actions
            non_sink_logprobs[~terminating_states_mask] = distribution.log_prob(
                sampled_actions
            ) + torch.log(1 - terminate_prob[~terminating_states_mask])

        actions[non_sink_mask] = non_sink_actions
        logprobs[non_sink_mask] = non_sink_logprobs

        return actions, logprobs

    def evaluate_backward_logprobs(
        self,
        bw_model: NeuralNet,
        trajectories: TensorType["n_traj", "n", "dim"],
        actionss: TensorType["n_traj", "n-1", "dim"],
    ):
        assert bw_model.output_dim == 1 + 2 * self.dim

        if trajectories.shape[1] == 2:  # only start and end states
            return torch.zeros(trajectories.shape[0], device=self.device)

        backward_logprobs = torch.zeros_like(trajectories[:, 1:-1, 0])
        non_sink_mask = ~torch.all(trajectories[:, 1:-1, :] == self.sink_state, dim=-1)
        non_sink_states = trajectories[:, 1:-1, :][non_sink_mask]
        out = bw_model(non_sink_states)
        terminate_pre_sigmoid_prob = out[..., 0]

        terminate_pre_sigmoid_prob[
            torch.any(non_sink_states > self.delta, dim=-1)
        ] = -float("inf")

        terminate_prob = torch.sigmoid(terminate_pre_sigmoid_prob)

        s1_indices = (
            torch.cat(
                (
                    torch.zeros(1, device=self.device),
                    non_sink_mask.sum(dim=-1).cumsum(dim=-1).unique()[:-1],
                )
            )
            .unique()
            .long()
        )

        backward_logprobs[:, 0][non_sink_mask[:, 0]] = torch.log(
            terminate_prob[s1_indices]
        )

        s1_mask = torch.zeros_like(out[:, 0]).bool()
        s1_mask[s1_indices] = True

        if not torch.all(s1_mask):
            log_alphas = out[..., 1 : self.dim + 1]
            log_betas = out[..., self.dim + 1 :]
            log_alphas = log_alphas[~s1_mask]
            log_betas = log_betas[~s1_mask]

            alphas = torch.exp(log_alphas)
            betas = torch.exp(log_betas)

            distribution = torch.distributions.Beta(alphas, betas)
            distribution = torch.distributions.Independent(distribution, 1)

            actions = actionss[:, 1:][
                ~torch.all(actionss[:, 1:] == self.sink_state, dim=-1)
            ]

            pre_actionss = actions / torch.min(
                torch.full_like(actions, self.delta),
                trajectories[:, 1:-1, :][non_sink_mask][~s1_mask],
            )

            backward_logprobs[:, 1:][non_sink_mask[:, 1:]] = distribution.log_prob(
                pre_actionss
            ) + torch.log(1 - terminate_prob[~s1_mask])

        return backward_logprobs.sum(dim=-1)


# The following resembles
class MixtureOfBetasActionsSampler(ActionsSampler):
    def __init__(
        self,
        env: Box,
        model: NeuralNet,
        n_components=10,
        epsilon=1e-4,
        sf_bias=0.0,
        epsilon_uniform=0.0,
    ):
        super().__init__(env, epsilon_uniform=epsilon_uniform)
        self.dim = env.dim
        self.sink_state = env.sink_state
        self.model = model
        self.n_components = n_components
        self.epsilon = epsilon
        self.sf_bias = sf_bias
        assert model.output_dim == 1 + n_components * (2 * self.dim + 1)

    def sample_pre_actions(self, states):
        actions = torch.full_like(states, -float("inf"))
        logprobs = torch.zeros_like(states[:, 0])
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        non_sink_states = states[non_sink_mask]
        non_sink_actions = actions[non_sink_mask]
        non_sink_logprobs = logprobs[non_sink_mask]
        out = self.model(non_sink_states)
        terminate_pre_sigmoid_prob = out[..., 0]
        terminate_pre_sigmoid_prob[
            torch.any(non_sink_states > 1 - self.env.epsilon, dim=-1)
        ] = float("inf")
        terminate_prob = torch.sigmoid(terminate_pre_sigmoid_prob - 2)
        # The following line is kind of off policy sampling
        terminating_states_mask = torch.rand_like(terminate_prob) < torch.sigmoid(
            terminate_pre_sigmoid_prob - 2 - self.sf_bias
        ).clamp_min(self.epsilon)
        non_sink_logprobs[terminating_states_mask] = torch.log(
            terminate_prob[terminating_states_mask]
        )
        if not torch.all(terminating_states_mask):
            mixture_logits = out[..., 1 : self.n_components + 1]
            mixture_logits = mixture_logits[~terminating_states_mask]

            mixture_distributions = torch.distributions.Categorical(
                logits=mixture_logits
            )

            components_log_alphas, components_log_betas = (
                out[
                    ..., self.n_components + 1 : self.n_components * (self.dim + 1) + 1
                ],
                out[..., self.n_components * (self.dim + 1) + 1 :],
            )
            components_log_alphas = components_log_alphas[~terminating_states_mask]
            components_log_alphas = components_log_alphas.reshape(
                -1, self.n_components, self.dim
            )
            components_log_betas = components_log_betas[~terminating_states_mask]
            components_log_betas = components_log_betas.reshape(
                -1, self.n_components, self.dim
            )

            components_alphas = torch.exp(components_log_alphas).clamp(0.1, 5.0)
            components_betas = torch.exp(components_log_betas).clamp(0.1, 5.0)

            components_distributions = torch.distributions.Beta(
                components_alphas, components_betas
            )
            components_distributions = torch.distributions.Independent(
                components_distributions, 1
            )

            final_distribution = torch.distributions.MixtureSameFamily(
                mixture_distributions, components_distributions
            )

            uniform_actions_mask = (
                torch.rand_like(components_alphas[:, 0, 0]) < self.epsilon_uniform
            )
            uniform_actions = torch.rand_like(components_alphas[:, 0, :])
            sampled_actions = final_distribution.sample()
            sampled_actions[uniform_actions_mask] = uniform_actions[
                uniform_actions_mask
            ]

            non_sink_actions[~terminating_states_mask] = sampled_actions

            non_sink_logprobs[~terminating_states_mask] = final_distribution.log_prob(
                sampled_actions
            ) + torch.log(1 - terminate_prob[~terminating_states_mask])

        actions[non_sink_mask] = non_sink_actions
        logprobs[non_sink_mask] = non_sink_logprobs

        return actions, logprobs

    def evaluate_backward_logprobs(
        self,
        bw_model: NeuralNet,
        trajectories: TensorType["n_traj", "n", "dim"],
        actionss: TensorType["n_traj", "n-1", "dim"],
    ):
        assert bw_model.output_dim == 1 + self.n_components * (2 * self.dim + 1)

        if trajectories.shape[1] == 2:  # only start and end states
            return torch.zeros(trajectories.shape[0], device=self.device)

        backward_logprobs = torch.zeros_like(trajectories[:, 1:-1, 0])
        non_sink_mask = ~torch.all(trajectories[:, 1:-1, :] == self.sink_state, dim=-1)
        non_sink_states = trajectories[:, 1:-1, :][non_sink_mask]
        out = bw_model(non_sink_states)
        terminate_pre_sigmoid_prob = out[..., 0]

        terminate_pre_sigmoid_prob[
            torch.any(non_sink_states > self.delta, dim=-1)
        ] = -float("inf")

        terminate_prob = torch.sigmoid(terminate_pre_sigmoid_prob)

        s1_indices = (
            torch.cat(
                (
                    torch.zeros(1, device=self.device),
                    non_sink_mask.sum(dim=-1).cumsum(dim=-1).unique()[:-1],
                )
            )
            .unique()
            .long()
        )

        backward_logprobs[:, 0][non_sink_mask[:, 0]] = torch.log(
            terminate_prob[s1_indices]
        )

        s1_mask = torch.zeros_like(out[:, 0]).bool()
        s1_mask[s1_indices] = True

        if not torch.all(s1_mask):

            mixture_logits = out[..., 1 : self.n_components + 1]
            mixture_logits = mixture_logits[~s1_mask]

            mixture_distributions = torch.distributions.Categorical(
                logits=mixture_logits
            )

            components_log_alphas, components_log_betas = (
                out[
                    ..., self.n_components + 1 : self.n_components * (self.dim + 1) + 1
                ],
                out[..., self.n_components * (self.dim + 1) + 1 :],
            )
            components_log_alphas = components_log_alphas[~s1_mask]
            components_log_alphas = components_log_alphas.reshape(
                -1, self.n_components, self.dim
            )
            components_log_betas = components_log_betas[~s1_mask]
            components_log_betas = components_log_betas.reshape(
                -1, self.n_components, self.dim
            )

            components_alphas = torch.exp(components_log_alphas)
            components_betas = torch.exp(components_log_betas)

            components_distributions = torch.distributions.Beta(
                components_alphas, components_betas
            )
            components_distributions = torch.distributions.Independent(
                components_distributions, 1
            )

            final_distribution = torch.distributions.MixtureSameFamily(
                mixture_distributions, components_distributions
            )

            actions = actionss[:, 1:][
                ~torch.all(actionss[:, 1:] == self.sink_state, dim=-1)
            ]

            pre_actionss = actions / torch.min(
                torch.full_like(actions, self.delta),
                trajectories[:, 1:-1, :][non_sink_mask][~s1_mask],
            )

            backward_logprobs[:, 1:][
                non_sink_mask[:, 1:]
            ] = final_distribution.log_prob(pre_actionss) + torch.log(
                1 - terminate_prob[~s1_mask]
            )

        return backward_logprobs.sum(dim=-1)
