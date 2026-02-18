"""
COS435: Reinforcement Learning - Homework 1
Shared utilities: helpers, plotting, and visualization.

Do NOT modify this file.
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Helper Functions
# ============================================================

def get_transition_model(env):
    """Extract the transition model from a gymnasium environment.

    Returns:
        nS: Number of states
        nA: Number of actions
        P:  Transition dict where P[s][a] = [(prob, next_state, reward, terminated), ...]
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    P = env.unwrapped.P
    return nS, nA, P


def compute_q_value(s, a, V, P, gamma):
    """Compute Q(s, a) = sum_{s'} p(s'|s,a) * [r + gamma * V(s') * (1-term)].

    When a transition is terminal (terminated=True), the future value is 0.

    Args:
        s: state index
        a: action index
        V: value function array, shape [nS]
        P: transition dict from get_transition_model
        gamma: discount factor

    Returns:
        float: the Q-value for (s, a)
    """
    q = 0.0
    for prob, next_state, reward, terminated in P[s][a]:
        if terminated:
            q += prob * reward
        else:
            q += prob * (reward + gamma * V[next_state])
    return q


# ============================================================
# Plotting and Visualization
# ============================================================

def plot_vi_convergence(histories, labels, title, filename):
    """Plot VI convergence curves (semi-log scale).

    Args:
        histories: List of residual histories (one list per algorithm)
        labels: Algorithm names (same length as histories)
        title: Plot title
        filename: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(range(1, len(history) + 1), history, label=label, linewidth=2)
    plt.xlabel("Iteration (sweep)", fontsize=12)
    plt.ylabel("Max Bellman Residual", fontsize=12)
    plt.yscale("log")
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def print_policy_arrows(Q, env_name):
    """Print the greedy policy as arrows on the environment grid.

    Args:
        Q: Q-table, shape [nS, nA]
        env_name: "FrozenLake-v1" or "CliffWalking-v0"
    """
    policy = np.argmax(Q, axis=1)

    if env_name == "FrozenLake-v1":
        nrows, ncols = 4, 4
        arrows = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}

        env = gym.make("FrozenLake-v1", is_slippery=True)
        desc = env.unwrapped.desc
        env.close()

        print(f"\nOptimal Policy ({env_name}):")
        print("+" + "---+" * ncols)
        for r in range(nrows):
            row_str = "|"
            for c in range(ncols):
                s = r * ncols + c
                cell = desc[r][c]
                if isinstance(cell, bytes):
                    cell = cell.decode()
                if cell == "H":
                    row_str += " H |"
                elif cell == "G":
                    row_str += " G |"
                else:
                    row_str += f" {arrows[policy[s]]} |"
            print(row_str)
            print("+" + "---+" * ncols)
        print("Legend: H=Hole, G=Goal, \u2190\u2193\u2192\u2191=Actions")

    elif env_name == "CliffWalking-v0":
        nrows, ncols = 4, 12
        arrows = {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"}

        print(f"\nOptimal Policy ({env_name}):")
        print("+" + "---+" * ncols)
        for r in range(nrows):
            row_str = "|"
            for c in range(ncols):
                s = r * ncols + c
                if r == 3 and c == 0:
                    row_str += " S |"
                elif r == 3 and c == 11:
                    row_str += " G |"
                elif r == 3 and 1 <= c <= 10:
                    row_str += " C |"
                else:
                    row_str += f" {arrows[policy[s]]} |"
            print(row_str)
            print("+" + "---+" * ncols)
        print("Legend: S=Start, G=Goal, C=Cliff, \u2191\u2192\u2193\u2190=Actions")

    else:
        raise ValueError(f"Unknown environment: {env_name}")


def plot_policy_arrows(Q, env_name, filename):
    """Plot the greedy policy as arrows on a colored grid using matplotlib.

    Args:
        Q: Q-table, shape [nS, nA]
        env_name: "FrozenLake-v1" or "CliffWalking-v0"
        filename: Path to save the figure
    """
    from matplotlib.colors import ListedColormap

    policy = np.argmax(Q, axis=1)

    if env_name == "FrozenLake-v1":
        nrows, ncols = 4, 4
        dx_map = {0: -1, 1: 0, 2: 1, 3: 0}
        dy_map = {0: 0, 1: 1, 2: 0, 3: -1}

        env = gym.make("FrozenLake-v1", is_slippery=True)
        desc = env.unwrapped.desc
        env.close()

        grid = np.zeros((nrows, ncols))
        for r in range(nrows):
            for c in range(ncols):
                cell = desc[r][c]
                if isinstance(cell, bytes):
                    cell = cell.decode()
                if cell == "H":
                    grid[r, c] = 1
                elif cell == "G":
                    grid[r, c] = 2
                elif cell == "S":
                    grid[r, c] = 3

        cmap = ListedColormap(["#E8F4FD", "#2C2C2C", "#4CAF50", "#FFC107"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(grid, cmap=cmap, origin="upper")

        for r in range(nrows):
            for c in range(ncols):
                s = r * ncols + c
                cell = desc[r][c]
                if isinstance(cell, bytes):
                    cell = cell.decode()
                if cell == "H":
                    ax.text(c, r, "H", ha="center", va="center",
                            fontsize=16, fontweight="bold", color="white")
                elif cell == "G":
                    ax.text(c, r, "G", ha="center", va="center",
                            fontsize=16, fontweight="bold", color="white")
                else:
                    dx = dx_map[policy[s]] * 0.3
                    dy = dy_map[policy[s]] * 0.3
                    ax.arrow(c - dx * 0.5, r - dy * 0.5, dx, dy,
                             head_width=0.15, head_length=0.08,
                             fc="black", ec="black")

        ax.set_xticks(np.arange(ncols))
        ax.set_yticks(np.arange(nrows))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(-0.5, ncols), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows), minor=True)
        ax.grid(which="minor", color="gray", linewidth=1.5)
        ax.tick_params(which="minor", size=0)
        ax.set_title(f"Optimal Policy \u2014 {env_name}", fontsize=13)

    elif env_name == "CliffWalking-v0":
        nrows, ncols = 4, 12
        dx_map = {0: 0, 1: 1, 2: 0, 3: -1}
        dy_map = {0: -1, 1: 0, 2: 1, 3: 0}

        grid = np.zeros((nrows, ncols))
        grid[3, 0] = 3
        grid[3, 11] = 2
        grid[3, 1:11] = 1

        cmap = ListedColormap(["#E8F4FD", "#D32F2F", "#4CAF50", "#FFC107"])
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.imshow(grid, cmap=cmap, origin="upper")

        for r in range(nrows):
            for c in range(ncols):
                s = r * ncols + c
                if r == 3 and c == 0:
                    ax.text(c, r, "S", ha="center", va="center",
                            fontsize=14, fontweight="bold")
                elif r == 3 and c == 11:
                    ax.text(c, r, "G", ha="center", va="center",
                            fontsize=14, fontweight="bold", color="white")
                elif r == 3 and 1 <= c <= 10:
                    ax.text(c, r, "C", ha="center", va="center",
                            fontsize=12, fontweight="bold", color="white")
                else:
                    dx = dx_map[policy[s]] * 0.3
                    dy = dy_map[policy[s]] * 0.3
                    ax.arrow(c - dx * 0.5, r - dy * 0.5, dx, dy,
                             head_width=0.15, head_length=0.08,
                             fc="black", ec="black")

        ax.set_xticks(np.arange(ncols))
        ax.set_yticks(np.arange(nrows))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(-0.5, ncols), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows), minor=True)
        ax.grid(which="minor", color="gray", linewidth=1.5)
        ax.tick_params(which="minor", size=0)
        ax.set_title(f"Optimal Policy \u2014 {env_name}", fontsize=13)

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_alpha_sweep(env, filename, alphas=None, n_short=100, n_long=10000,
                     epsilon=0.1, gamma=0.99, n_runs=1, module_name="hw1"):
    """Reproduce Fig. 2 from van Seijen et al. (2009).

    For each learning rate alpha, runs SARSA, Q-Learning, and Expected SARSA,
    then plots the average return over the first n_short and n_long episodes.

    Args:
        env: gymnasium environment (CliffWalking-v0)
        filename: Path to save the figure
        alphas: Array of learning rates to sweep
        n_short: Number of episodes for the "short" average (default 100)
        n_long: Number of episodes for the "long" average (default 10000)
        epsilon: Exploration rate
        gamma: Discount factor
        n_runs: Number of independent runs to average over
        module_name: Python module to import TD algorithms from (default "hw1")
    """
    from tqdm import tqdm
    import importlib

    if alphas is None:
        alphas = np.linspace(0.1, 1.0, 10)

    # Import algorithm functions from the specified module
    mod = importlib.import_module(module_name)
    alg_fns = {
        "Sarsa": mod.sarsa,
        "Q-learning": mod.q_learning,
        "Expected Sarsa": mod.expected_sarsa,
    }
    alg_names = ["Sarsa", "Q-learning", "Expected Sarsa"]

    styles = {
        "Sarsa":          {"color": "blue",  "marker": "v"},
        "Q-learning":     {"color": "black", "marker": "s"},
        "Expected Sarsa": {"color": "red",   "marker": "x"},
    }

    results = {name: {"short": [], "long": []} for name in alg_names}

    total = len(alphas) * len(alg_names)
    pbar = tqdm(total=total, desc="Alpha sweep", unit="config")

    for alpha in alphas:
        for alg_name in alg_names:
            alg_fn = alg_fns[alg_name]
            short_accum = []
            long_accum = []
            for run in range(n_runs):
                np.random.seed(run)
                result = alg_fn(env, num_episodes=n_long, alpha=alpha,
                                gamma=gamma, epsilon=epsilon)
                rewards = result[1]
                short_accum.append(np.mean(rewards[:n_short]))
                long_accum.append(np.mean(rewards))
            results[alg_name]["short"].append(np.mean(short_accum))
            results[alg_name]["long"].append(np.mean(long_accum))
            pbar.set_postfix_str(f"\u03b1={alpha:.1f} {alg_name}")
            pbar.update(1)
    pbar.close()

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))

    for alg_name in alg_names:
        s = styles[alg_name]
        short_vals = results[alg_name]["short"]
        long_vals = results[alg_name]["long"]

        ax.plot(alphas, short_vals, linestyle="--", marker=s["marker"],
                color=s["color"], markerfacecolor="none",
                markeredgecolor=s["color"], markersize=7,
                label=f"n = {n_short}, {alg_name}")

        ax.plot(alphas, long_vals, linestyle="-", marker=s["marker"],
                color=s["color"], markerfacecolor="none",
                markeredgecolor=s["color"], markersize=7,
                label=f"n = {n_long:.0e}, {alg_name}")

        best_idx = int(np.argmax(long_vals))
        ax.plot(alphas[best_idx], long_vals[best_idx], "o",
                color=s["color"], markersize=10)

    ax.set_xlabel(r"$\alpha$", fontsize=13)
    ax.set_ylabel("average return", fontsize=13)
    ax.set_title("Average Return vs Learning Rate \u2014 CliffWalking-v0", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(alphas[0] - 0.02, alphas[-1] + 0.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_td_convergence(reward_histories, labels, title, filename, window=100,
                        linestyles=None):
    """Plot smoothed episode-reward curves for TD algorithms.

    Args:
        reward_histories: List of reward lists (one per algorithm)
        labels: Algorithm names
        title: Plot title
        filename: Path to save the figure
        window: Rolling-average window size
        linestyles: Optional list of matplotlib linestyle strings (e.g. "-", ":")
                    one per curve.  Defaults to solid lines for all.
    """
    if linestyles is None:
        linestyles = ["-"] * len(reward_histories)
    plt.figure(figsize=(10, 6))
    for rewards, label, ls in zip(reward_histories, labels, linestyles):
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        else:
            smoothed = np.array(rewards)
        plt.plot(smoothed, label=label, linewidth=1.5, alpha=0.85, linestyle=ls)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(f"Average Reward (window={window})", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
