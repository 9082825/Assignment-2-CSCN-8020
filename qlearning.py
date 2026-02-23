"""
CSCN 8020 – Assignment 2
Q-Learning on Taxi-v3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import json
from assignment2_utils import load_environment, print_env_info


def train_qlearning(
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.9,
    n_episodes: int = 10_000,
    max_steps: int = 200,
    seed: int = 42,
):
    """
    Train a Q-Learning agent on Taxi-v3.

    Returns
    -------
    Q : np.ndarray  – final Q-table (500 x 6)
    metrics : dict  – episode returns, steps, and running averages
    """
    env = load_environment()
    n_states = env.observation_space.n   # 500
    n_actions = env.action_space.n       # 6

    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))

    episode_returns = []
    episode_steps   = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(max_steps):
            # ε-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[obs]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning update
            best_next = np.max(Q[next_obs])
            Q[obs, action] += alpha * (reward + gamma * best_next - Q[obs, action])

            obs = next_obs
            total_reward += reward

            if done:
                break

        episode_returns.append(total_reward)
        episode_steps.append(step + 1)

    env.close()

    # Compute 100-episode rolling average
    avg_returns = np.convolve(episode_returns, np.ones(100) / 100, mode="valid")
    avg_steps   = np.convolve(episode_steps,   np.ones(100) / 100, mode="valid")

    metrics = {
        "episode_returns": episode_returns,
        "episode_steps":   episode_steps,
        "avg_returns":     avg_returns.tolist(),
        "avg_steps":       avg_steps.tolist(),
        "total_episodes":  n_episodes,
    }
    return Q, metrics


def plot_metrics(metrics_dict: dict, title_suffix: str, save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, m in metrics_dict.items():
        x = range(1, len(m["avg_returns"]) + 1)
        axes[0].plot(x, m["avg_returns"], label=label)
        axes[1].plot(x, m["avg_steps"],   label=label)

    axes[0].set_title("Average Return per Episode (100-ep window)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Return")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Average Steps per Episode (100-ep window)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average Steps")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"qlearning_{title_suffix}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved plot → {fname}")


def summarise(label: str, metrics: dict):
    print(f"\n[{label}]")
    print(f"  Total episodes       : {metrics['total_episodes']}")
    print(f"  Mean steps/episode   : {np.mean(metrics['episode_steps']):.2f}")
    print(f"  Mean return/episode  : {np.mean(metrics['episode_returns']):.2f}")
    print(f"  Final 100-ep return  : {metrics['avg_returns'][-1]:.2f}")


def run_experiments():
    N_EPISODES = 10_000
    SAVE_DIR   = "plots"
    results    = {}

    # Baseline
    print("\n=== Baseline: α=0.1, ε=0.1, γ=0.9 ===")
    Q_base, m_base = train_qlearning(alpha=0.1, epsilon=0.1, gamma=0.9,
                                     n_episodes=N_EPISODES)
    summarise("Baseline α=0.1 ε=0.1 γ=0.9", m_base)
    results["Baseline"] = m_base
    np.save("qtable_baseline.npy", Q_base)

    # Learning Rate sweep
    print("\n=== Learning Rate Sweep ===")
    lr_metrics = {}
    for alpha in [0.01, 0.001, 0.2]:
        label = f"α={alpha}"
        print(f"  Training {label} …")
        _, m = train_qlearning(alpha=alpha, epsilon=0.1, gamma=0.9,
                               n_episodes=N_EPISODES)
        summarise(label, m)
        lr_metrics[label] = m
        results[f"alpha_{alpha}"] = m

    lr_metrics["α=0.1 (baseline)"] = m_base
    plot_metrics(lr_metrics, "learning_rate_sweep", SAVE_DIR)

    # Exploration Factor sweep
    # (Your prompt labels this as γ; here it’s correctly treated as ε.)
    print("\n=== Exploration Factor (ε) Sweep ===")
    eps_metrics = {}
    for epsilon in [0.2, 0.3]:
        label = f"ε={epsilon}"
        print(f"  Training {label} …")
        _, m = train_qlearning(alpha=0.1, epsilon=epsilon, gamma=0.9,
                               n_episodes=N_EPISODES)
        summarise(label, m)
        eps_metrics[label] = m
        results[f"epsilon_{epsilon}"] = m

    eps_metrics["ε=0.1 (baseline)"] = m_base
    plot_metrics(eps_metrics, "exploration_factor_sweep", SAVE_DIR)

    # Best combination (kept simple; update after you inspect metrics)
    print("\n=== Best Combination: α=0.2, ε=0.1, γ=0.9 ===")
    Q_best, m_best = train_qlearning(alpha=0.2, epsilon=0.1, gamma=0.9,
                                     n_episodes=N_EPISODES)
    summarise("Best α=0.2 ε=0.1", m_best)
    results["best"] = m_best
    np.save("qtable_best.npy", Q_best)

    best_metrics = {
        "Baseline α=0.1 ε=0.1": m_base,
        "Best α=0.2 ε=0.1":     m_best,
    }
    plot_metrics(best_metrics, "best_combination", SAVE_DIR)

    with open("all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll metrics saved → all_metrics.json")
    print("\n=== All experiments complete. Plots saved in ./plots/ ===")


if __name__ == "__main__":
    env = load_environment()
    print_env_info(env)
    env.close()
    run_experiments()
