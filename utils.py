import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity
from scipy.special import logsumexp


def get_test_states(n=100, maxi=1.0):
    x = np.linspace(0.001, maxi, n)
    y = np.linspace(0.001, maxi, n)
    xx, yy = np.meshgrid(x, y)
    test_states = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    return test_states, n


def plot_reward(env):
    test_states, n = get_test_states()
    test_states = torch.FloatTensor(test_states)
    reward = env.reward(test_states)
    reward = reward.reshape(n, n)
    fig = plt.imshow(reward, origin="lower", extent=[0, 1, 0, 1])
    plt.xticks([])
    plt.yticks([])
    plt.close("all")
    return fig


def plot_transitions(env, states, new_states):
    terminating_mask = torch.all(new_states == env.sink_state, dim=-1)
    terminating_states = states[terminating_mask]
    non_terminating_states = states[~terminating_mask]
    plt.scatter(terminating_states[:, 0], terminating_states[:, 1], color="green")
    plt.scatter(non_terminating_states[:, 0], non_terminating_states[:, 1], color="red")
    non_terminating_new_states = new_states[~terminating_mask]
    plt.scatter(
        non_terminating_new_states[:, 0], non_terminating_new_states[:, 1], color="blue"
    )

    plt.show()


def fit_kde(
    last_states, kernel="exponential", bandwidth=0.1, plot=False, show_plot=False
):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(last_states.numpy())

    fig = None
    if plot:
        test_states, n = get_test_states()

        log_dens = kde.score_samples(test_states)
        fig = plt.imshow(
            np.exp(log_dens).reshape(n, n), origin="lower", extent=[0, 1, 0, 1]
        )
        plt.colorbar()
    if show_plot:
        plt.show()
    plt.close("all")
    return kde, fig


def estimate_jsd(kde1, kde2):
    # Estimate Jensen-Shannon divergence between two KDEs
    test_states, n = get_test_states()
    log_dens1 = kde1.score_samples(test_states)
    log_dens1 = log_dens1 - logsumexp(log_dens1)
    log_dens2 = kde2.score_samples(test_states)
    log_dens2 = log_dens2 - logsumexp(log_dens2)
    log_dens = np.log(0.5 * np.exp(log_dens1) + 0.5 * np.exp(log_dens2))
    jsd = np.sum(np.exp(log_dens1) * (log_dens1 - log_dens))
    jsd += np.sum(np.exp(log_dens2) * (log_dens2 - log_dens))
    return jsd / 2.0


def plot_trajectories(trajectories, plot=False):
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    fig = plt.figure()
    for j in range(trajectories.shape[0]):
        plt.plot(trajectories[j, :, 0], trajectories[j, :, 1], color=colors[j % 10])
    if plot:
        plt.show()
    plt.close("all")
    return fig


def plot_termination_probabilities(model, plot=False):
    fig = plt.figure(figsize=(6 * 16 / 9, 6))
    test_states, n = get_test_states()

    out = model(torch.FloatTensor(test_states))
    termination_probs = torch.sigmoid(out[0]).reshape(n, n)

    plt.imshow(termination_probs.detach().numpy(), cmap="hot")
    plt.xlim(0, n)
    plt.ylim(0, n)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    if plot:
        plt.show()

    plt.close("all")
    return fig


def plot_samples(samples, plot=False):
    fig = plt.scatter(samples[:, 0], samples[:, 1])
    if plot:
        plt.show()
    plt.close("all")
    return fig


def sample_from_reward(env, n_samples):
    # Implement rejection sampling, with proposal being uniform distribution in [0, 1]^2
    samples = []
    while len(samples) < n_samples:
        sample = np.random.rand(n_samples, 2)
        rewards = env.reward(torch.FloatTensor(sample)).numpy()
        mask = np.random.rand(n_samples) * (env.R0 + max(env.R1, env.R2)) < rewards
        true_samples = sample[mask]
        samples.extend(true_samples[-(n_samples - len(samples)) :])
    return torch.FloatTensor(np.array(samples))
