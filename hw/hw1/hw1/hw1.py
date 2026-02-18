"""
COS435: Reinforcement Learning - Homework 1
Value Iteration and Temporal Difference Learning

In this assignment you will implement:
  Part 1: Three Value Iteration variants on FrozenLake-v1 and convergence plotting
  Part 2: Three TD learning algorithms on CliffWalking-v0, policy visualization,
          and reward plotting

Fill in every section marked with TODO.  Do NOT change function signatures.
Run tests:   pytest test_hw1.py
Run script:  python hw1.py
"""

import gymnasium as gym
import numpy as np
import heapq

from utils import (
    get_transition_model,
    compute_q_value,
    plot_vi_convergence,
    print_policy_arrows,
    plot_policy_arrows,
    plot_alpha_sweep,
    plot_td_convergence,
)


# ============================================================
# Part 1: Value Iteration Algorithms
# ============================================================

def value_iteration(env, gamma=0.99, theta=1e-8):
    """Standard (synchronous) Value Iteration.

    At each sweep, compute V_{k+1}(s) = max_a Q(s, a; V_k) for ALL states
    using the value function from the *previous* sweep (do not update in-place).

    Convergence criterion: stop when max_s |V_{k+1}(s) - V_k(s)| < theta.

    Args:
        env: gymnasium environment
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: Optimal value function, numpy array of shape [nS]
        policy: Greedy policy w.r.t. V, numpy array of shape [nS]
        history: List of max Bellman residuals (one float per sweep)
    """
    nS, nA, P = get_transition_model(env)
    V = np.zeros(nS)
    history = []

    # TODO: Implement synchronous value iteration.

    policy = extract_policy(env, V, gamma)
    return V, policy, history


def gauss_seidel_vi(env, gamma=0.99, theta=1e-8):
    """Gauss-Seidel (asynchronous / in-place) Value Iteration.

    Same as standard VI except values are updated IN-PLACE: when computing
    Q(s, a), the latest V is used (which already includes updates for s' < s).

    Args / Returns: same as value_iteration.
    """
    nS, nA, P = get_transition_model(env)
    V = np.zeros(nS)
    history = []

    # TODO: Implement in-place (Gauss-Seidel) value iteration.

    raise NotImplementedError

    policy = extract_policy(env, V, gamma)
    return V, policy, history



def prioritized_sweeping_vi(env, gamma=0.99, theta=1e-8, max_updates=1000):
    """Prioritized Sweeping Value Iteration.

    Instead of sweeping over states in order, sweep them in order of Bellman
    error for a margin of error above theta.

    Args:
        env, gamma, theta: same as value_iteration
        max_updates: maximum number of individual state updates

    Returns:
        V, policy: same as value_iteration
        history: list of max Bellman residuals recorded every nS updates
                 (so each entry is comparable to one full sweep)
    """
    nS, nA, P = get_transition_model(env)
    V = np.zeros(nS)
    history = []

    # TODO: Implement prioritized sweeping.

    raise NotImplementedError

    policy = extract_policy(env, V, gamma)
    return V, policy, history


def extract_policy(env, V, gamma=0.99):
    """Extract the greedy policy from a value function.

    pi(s) = argmax_a Q(s, a; V)

    Args:
        env: gymnasium environment
        V: Value function, shape [nS]
        gamma: Discount factor

    Returns:
        policy: numpy int array of shape [nS]
    """
    nS, nA, P = get_transition_model(env)
    policy = np.zeros(nS, dtype=int)

    # TODO: For each state, find the action that maximizes Q(s, a; V).
    # Use compute_q_value and np.argmax.
    raise NotImplementedError

    return policy


# ============================================================
# Part 2: TD Learning Algorithms
# ============================================================

def epsilon_greedy(Q, state, epsilon, n_actions):
    """Epsilon-greedy action selection.

    With probability epsilon:   return a uniformly random action.
    With probability 1-epsilon: return argmax_a Q[state, a].

    Args:
        Q: Q-table, shape [nS, nA]
        state: Current state index
        epsilon: Exploration probability
        n_actions: Size of the action space

    Returns:
        int: Selected action
    """
    # TODO: Implement epsilon-greedy.
    raise NotImplementedError


def sarsa(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """SARSA: On-policy TD control.

    Update rule (after each transition s,a -> r, s',a'):
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

    where a' is selected by the SAME epsilon-greedy policy.

    Args:
        env: gymnasium environment
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Q: Learned Q-table, shape [nS, nA]
        episode_rewards: List of total rewards per episode
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    episode_rewards = []

    # TODO: Implement SARSA.
    raise NotImplementedError

    return Q, episode_rewards


def expected_sarsa(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Expected SARSA: Uses the expected Q-value under the current policy.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * E_pi[Q(s',a')] - Q(s,a)]

    where E_pi[Q(s',a')] = sum_a' pi(a'|s') * Q(s',a')
    and pi is the current epsilon-greedy policy.

    Args / Returns: same as sarsa.
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    episode_rewards = []

    # TODO: Implement Expected SARSA.
    raise NotImplementedError

    return Q, episode_rewards


def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Q-Learning: Off-policy TD control.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    The max makes Q-learning off-policy: it learns the optimal Q*
    regardless of the exploration policy used.

    Args:
        env: gymnasium environment
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Q: Learned Q-table, shape [nS, nA]
        episode_rewards: List of total rewards per episode (behavioral policy)
        greedy_rewards: List of total rewards per episode (greedy/target policy)
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    episode_rewards = []
    greedy_rewards = []

    # TODO: Implement Q-learning.
    # After each training episode, also run a greedy evaluation episode
    # (with a step limit of 200 to avoid infinite loops early in training):
    #   greedy_reward = 0; reset env; step with argmax(Q[state]) until done
    #   greedy_rewards.append(greedy_reward)
    raise NotImplementedError

    return Q, episode_rewards, greedy_rewards


# ============================================================
# Main -- runs everything and generates plots
# ============================================================

if __name__ == "__main__":
    import os

    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Part 1: Value Iteration on FrozenLake-v1
    # ----------------------------------------------------------
    env_name = "FrozenLake-v1"
    print(f"\n{'='*60}")
    print(f"  Part 1: Value Iteration on {env_name}")
    print(f"{'='*60}")

    env = gym.make(env_name, is_slippery=True)

    V_std, pol_std, hist_std = value_iteration(env, gamma=0.99, theta=1e-8)
    V_gs, pol_gs, hist_gs = gauss_seidel_vi(env, gamma=0.99, theta=1e-8)
    V_ps, pol_ps, hist_ps = prioritized_sweeping_vi(env, gamma=0.99, theta=1e-8)

    print(f"\nStandard VI:         {len(hist_std)} sweeps")
    print(f"Gauss-Seidel VI:     {len(hist_gs)} sweeps")
    print(f"Prioritized Sweep:   {len(hist_ps)} equivalent sweeps")

    # Build Q-table from V for arrow display
    nS, nA, P = get_transition_model(env)
    Q_vi = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            Q_vi[s, a] = compute_q_value(s, a, V_std, P, 0.99)
    print_policy_arrows(Q_vi, env_name)
    plot_policy_arrows(Q_vi, env_name,
                       os.path.join(plot_dir, "policy_frozenlake_v1.png"))
    print("Saved: plots/policy_frozenlake_v1.png")

    plot_vi_convergence(
        [hist_std, hist_gs, hist_ps],
        ["Standard VI", "Gauss-Seidel VI", "Prioritized Sweeping"],
        f"VI Convergence \u2014 {env_name}",
        os.path.join(plot_dir, "vi_convergence_frozenlake_v1.png"),
    )
    print(f"Saved: plots/vi_convergence_frozenlake_v1.png")
    env.close()

    # ----------------------------------------------------------
    # Part 2: TD Learning on CliffWalking-v0
    # ----------------------------------------------------------
    env_name = "CliffWalking-v0"
    print(f"\n{'='*60}")
    print(f"  Part 2: TD Learning on {env_name}")
    print(f"{'='*60}")

    env = gym.make(env_name)
    num_ep = 5000

    print(f"\nRunning SARSA ({num_ep} episodes)...")
    Q_s, rew_s = sarsa(env, num_episodes=num_ep, alpha=0.1, gamma=0.99, epsilon=0.1)

    print(f"Running Expected SARSA ({num_ep} episodes)...")
    Q_es, rew_es = expected_sarsa(env, num_episodes=num_ep, alpha=0.1, gamma=0.99, epsilon=0.1)

    print(f"Running Q-Learning ({num_ep} episodes)...")
    Q_ql, rew_ql, rew_ql_greedy = q_learning(env, num_episodes=num_ep, alpha=0.1, gamma=0.99, epsilon=0.1)

    print("\n--- SARSA Policy ---")
    print_policy_arrows(Q_s, env_name)
    plot_policy_arrows(Q_s, env_name,
                       os.path.join(plot_dir, "policy_sarsa_cliffwalking.png"))
    print("Saved: plots/policy_sarsa_cliffwalking.png")

    print("\n--- Expected SARSA Policy ---")
    print_policy_arrows(Q_es, env_name)
    plot_policy_arrows(Q_es, env_name,
                       os.path.join(plot_dir, "policy_expected_sarsa_cliffwalking.png"))
    print("Saved: plots/policy_expected_sarsa_cliffwalking.png")

    print("\n--- Q-Learning Policy ---")
    print_policy_arrows(Q_ql, env_name)
    plot_policy_arrows(Q_ql, env_name,
                       os.path.join(plot_dir, "policy_qlearning_cliffwalking.png"))
    print("Saved: plots/policy_qlearning_cliffwalking.png")

    plot_td_convergence(
        [rew_s, rew_es, rew_ql, rew_ql_greedy],
        ["SARSA", "Expected SARSA",
         "Q-Learning (behavioral)", "Q-Learning (target)"],
        f"TD Learning \u2014 {env_name}",
        os.path.join(plot_dir, "td_convergence_cliffwalking_v0.png"),
        linestyles=["-", "-", "-", ":"],
    )
    print(f"\nSaved: plots/td_convergence_cliffwalking_v0.png")

    # ----------------------------------------------------------
    # Part 2b: Learning Rate Sweep
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Learning Rate Sweep on {env_name}")
    print(f"{'='*60}\n")

    plot_alpha_sweep(
        env,
        os.path.join(plot_dir, "alpha_sweep_cliffwalking.png"),
    )
    print(f"\nSaved: plots/alpha_sweep_cliffwalking.png")
    env.close()

    print(f"\n{'='*60}")
    print("  Done! Check the plots/ directory for convergence figures.")
    print(f"{'='*60}")
