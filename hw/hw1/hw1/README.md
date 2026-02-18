# Homework 1: Value Iteration & Temporal Difference Learning

## Overview

In this assignment you will implement classic tabular reinforcement learning algorithms and compare their convergence behaviour on two standard environments.

**Part 1 — Value Iteration on FrozenLake-v1 (model-based)**
- Standard (synchronous) Value Iteration
- Gauss-Seidel (in-place / asynchronous) Value Iteration
- Prioritized Sweeping Value Iteration
- Convergence plotting

**Part 2 — TD Learning on CliffWalking-v0 (model-free)**
- SARSA (on-policy)
- Expected SARSA
- Q-Learning (off-policy)
- Policy visualisation as arrows on the grid

## Environments

Both environments come from the [Gymnasium](https://gymnasium.farama.org/) library.

| Environment | Part | States | Actions | Dynamics |
|---|---|---|---|---|
| `FrozenLake-v1` (is_slippery=True) | Part 1 (VI) | 16 (4x4 grid) | 4 (LEFT, DOWN, RIGHT, UP) | Stochastic (1/3 chance per perpendicular) |
| `CliffWalking-v0` | Part 2 (TD) | 48 (4x12 grid) | 4 (UP, RIGHT, DOWN, LEFT) | Deterministic |

## Setup

```bash
pip install gymnasium==0.29.0 numpy matplotlib pytest tqdm
```

## Files

| File | Description |
|---|---|
| `hw1.py` | **Your code goes here.** Fill in every `TODO`. |
| `utils.py` | Shared utilities (transition model helpers, plotting functions). Do not modify. |
| `plots/` | Directory where convergence figures and policy plots are saved. |

## Instructions

1. Open `hw1.py` and implement every function marked `TODO`.
   - **Provided in `utils.py` (do not modify):** `get_transition_model`, `compute_q_value`, `plot_vi_convergence`, `print_policy_arrows`, `plot_policy_arrows`, `plot_alpha_sweep`, `plot_td_convergence`
   - **You implement in `hw1.py`:** `value_iteration`, `gauss_seidel_vi`, `prioritized_sweeping_vi`, `extract_policy`, `epsilon_greedy`, `sarsa`, `expected_sarsa`, `q_learning`

2. Once implemented and working, run the full script to generate plots:
   ```bash
   python hw1.py
   ```
3. Inspect the plots in `plots/` and answer the discussion questions in the .tex / .pdf

## What to Submit

- `hw1.py` with all TODOs implemented.
- The generated plots (`plots/*.png`).
- The pdf of hw1, compiled from latex (plots should also be embedded in the relevant responses here.

