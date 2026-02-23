"""
CSCN 8020 – Assignment 2
Deep Q-Network (DQN) on Taxi-v3
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from assignment2_utils import load_environment, print_env_info


class DQN(nn.Module):
    """Fully-connected Q-Network for discrete state/action spaces."""
    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def one_hot(state: int, n_states: int) -> np.ndarray:
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


def train_dqn(
    alpha: float  = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end:   float = 0.05,
    epsilon_decay: int   = 5_000,
    gamma: float   = 0.9,
    n_episodes: int = 2_000,
    max_steps:  int = 200,
    batch_size: int = 64,
    target_update: int = 100,
    buffer_capacity: int = 10_000,
    seed: int = 42,
):
    """Train a DQN agent on Taxi-v3 and return training metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = load_environment()
    n_states  = env.observation_space.n   # 500
    n_actions = env.action_space.n        # 6

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    buffer    = ReplayBuffer(capacity=buffer_capacity)

    episode_returns = []
    episode_steps   = []
    global_step     = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(max_steps):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -global_step / epsilon_decay
            )

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_t = torch.tensor(one_hot(obs, n_states),
                                       dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = int(policy_net(state_t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(one_hot(obs, n_states), action, reward,
                        one_hot(next_obs, n_states), float(done))

            obs = next_obs
            total_reward += reward
            global_step  += 1

            if len(buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)

                states_t      = torch.tensor(states_b,      dtype=torch.float32).to(device)
                actions_t     = torch.tensor(actions_b,     dtype=torch.long).to(device)
                rewards_t     = torch.tensor(rewards_b,     dtype=torch.float32).to(device)
                next_states_t = torch.tensor(next_states_b, dtype=torch.float32).to(device)
                dones_t       = torch.tensor(dones_b,       dtype=torch.float32).to(device)

                current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(dim=1).values
                    target_q   = rewards_t + gamma * max_next_q * (1 - dones_t)

                loss = nn.functional.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_returns.append(total_reward)
        episode_steps.append(step + 1)

    env.close()

    avg_returns = np.convolve(episode_returns, np.ones(100) / 100, mode="valid")
    avg_steps   = np.convolve(episode_steps,   np.ones(100) / 100, mode="valid")

    metrics = {
        "episode_returns": episode_returns,
        "episode_steps":   episode_steps,
        "avg_returns":     avg_returns.tolist(),
        "avg_steps":       avg_steps.tolist(),
        "total_episodes":  n_episodes,
    }

    torch.save(policy_net.state_dict(), "dqn_model.pth")
    return policy_net, metrics


def plot_dqn_metrics(metrics: dict, save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = range(1, len(metrics["avg_returns"]) + 1)
    axes[0].plot(x, metrics["avg_returns"], color="steelblue")
    axes[0].set_title("DQN – Average Return (100-ep window)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Return")
    axes[0].grid(True)

    axes[1].plot(x, metrics["avg_steps"], color="darkorange")
    axes[1].set_title("DQN – Average Steps (100-ep window)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average Steps")
    axes[1].grid(True)

    plt.tight_layout()
    fname = os.path.join(save_dir, "dqn_training.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved plot → {fname}")


def summarise_dqn(metrics: dict):
    print("\n[DQN Results]")
    print(f"  Total episodes       : {metrics['total_episodes']}")
    print(f"  Mean steps/episode   : {np.mean(metrics['episode_steps']):.2f}")
    print(f"  Mean return/episode  : {np.mean(metrics['episode_returns']):.2f}")
    print(f"  Final 100-ep return  : {metrics['avg_returns'][-1]:.2f}")


if __name__ == "__main__":
    env = load_environment()
    print_env_info(env)
    env.close()

    print("\n=== Training Deep Q-Network on Taxi-v3 ===")
    model, metrics = train_dqn(n_episodes=2_000)
    summarise_dqn(metrics)
    plot_dqn_metrics(metrics)

    with open("dqn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nDQN metrics saved → dqn_metrics.json")
    print("DQN model saved   → dqn_model.pth")
