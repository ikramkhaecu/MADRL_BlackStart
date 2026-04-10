#!/usr/bin/env python3
"""
Train RL agents on the Black-Start Restoration environment.
Produces training curves and episode replay JSON for the HTML visualiser.
"""
from __future__ import annotations
import sys, os, warnings, json, time, logging
os.environ['NUMBA_DISABLE_JIT'] = '1'
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
# Silence pandapower's numba spam before import
import pandapower as _pp
_pp.pp_dir = _pp.pp_dir  # trigger import side-effects
try:
    import numba  # noqa
except ImportError:
    pass  # expected – we just want the import warning gone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.envs import BlackStartRestorationEnv
from src.agents import DQNAgent, PPOAgent

# ──────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────
N_EPISODES   = 300
MAX_STEPS    = 15
EVAL_EVERY   = 30
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'snapshots')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def train_dqn():
    env = BlackStartRestorationEnv(max_steps=MAX_STEPS)
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dims=env.action_space.nvec.tolist(),
        lr=5e-4, gamma=0.99, eps_start=1.0, eps_end=0.08,
        eps_decay=0.997, buffer_size=30_000, batch_size=64,
        target_update=150, hidden=128,
    )

    history = {"episode": [], "reward": [], "lrr": [], "steps": [],
               "eps": [], "loss": [], "freq": [], "buses": []}
    best_reward = -1e9

    t0 = time.time()
    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_loss = 0.0

        for step in range(MAX_STEPS):
            action = agent.act(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.store(obs, action, reward, next_obs, term or trunc)
            loss = agent.learn()
            ep_loss += loss
            ep_reward += reward
            obs = next_obs
            if term or trunc:
                break

        lrr = info.get("lrr", 0.0)
        freq = info.get("freq", 50.0)
        n_buses = len(env.energised_buses)

        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["lrr"].append(lrr)
        history["steps"].append(step + 1)
        history["eps"].append(agent.eps)
        history["loss"].append(ep_loss / max(step, 1))
        history["freq"].append(freq)
        history["buses"].append(n_buses)

        if ep % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_r = np.mean(history["reward"][-EVAL_EVERY:])
            avg_lrr = np.mean(history["lrr"][-EVAL_EVERY:])
            print(f"Ep {ep:4d} | R={avg_r:7.2f} | LRR={avg_lrr:.3f} | "
                  f"eps={agent.eps:.3f} | buses={n_buses:2d} | "
                  f"freq={freq:.1f}Hz | {elapsed:.0f}s")

            if avg_r > best_reward:
                best_reward = avg_r
                agent.save(os.path.join(SNAPSHOT_DIR, "best_dqn.pt"))
                # Save a replay of the best episode
                env2 = BlackStartRestorationEnv(max_steps=MAX_STEPS)
                obs2, _ = env2.reset()
                for _ in range(MAX_STEPS):
                    act2 = agent.act(obs2)
                    obs2, _, t2, tr2, _ = env2.step(act2)
                    if t2 or tr2:
                        break
                env2.save_episode_log(os.path.join(SNAPSHOT_DIR, "best_episode.json"))

    return history


def train_ppo():
    env = BlackStartRestorationEnv(max_steps=MAX_STEPS)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dims=env.action_space.nvec.tolist(),
        lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        entropy_coef=0.02, ppo_epochs=4, batch_size=64, hidden=128,
    )

    history = {"episode": [], "reward": [], "lrr": [], "steps": [],
               "loss": [], "freq": [], "buses": []}
    best_reward = -1e9
    t0 = time.time()

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS):
            action, logp, val = agent.act(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.store(obs, action, reward, logp, val, term or trunc)
            ep_reward += reward
            obs = next_obs
            if term or trunc:
                break

        loss = agent.learn()
        lrr = info.get("lrr", 0.0)
        freq = info.get("freq", 50.0)
        n_buses = len(env.energised_buses)

        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["lrr"].append(lrr)
        history["steps"].append(step + 1)
        history["loss"].append(loss)
        history["freq"].append(freq)
        history["buses"].append(n_buses)

        if ep % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_r = np.mean(history["reward"][-EVAL_EVERY:])
            avg_lrr = np.mean(history["lrr"][-EVAL_EVERY:])
            print(f"Ep {ep:4d} | R={avg_r:7.2f} | LRR={avg_lrr:.3f} | "
                  f"buses={n_buses:2d} | freq={freq:.1f}Hz | {elapsed:.0f}s")

            if avg_r > best_reward:
                best_reward = avg_r
                agent.save(os.path.join(SNAPSHOT_DIR, "best_ppo.pt"))

    return history


def plot_training(histories: dict, save_path: str):
    """Generate training curve comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Black-Start Restoration — Training Progress", fontsize=14, weight='bold')

    colors = {"DQN": "#2196F3", "PPO": "#FF5722"}
    window = 30

    for name, h in histories.items():
        c = colors.get(name, "#333")
        eps = h["episode"]

        # Reward
        ax = axes[0, 0]
        raw = h["reward"]
        smooth = np.convolve(raw, np.ones(window)/window, mode='valid')
        ax.plot(eps[:len(smooth)], smooth, label=name, color=c, linewidth=1.5)
        ax.set_ylabel("Cumulative Reward"); ax.set_title("Episode Reward")
        ax.legend(); ax.grid(alpha=0.3)

        # LRR
        ax = axes[0, 1]
        raw = h["lrr"]
        smooth = np.convolve(raw, np.ones(window)/window, mode='valid')
        ax.plot(eps[:len(smooth)], smooth, label=name, color=c, linewidth=1.5)
        ax.set_ylabel("Load Recovery Ratio"); ax.set_title("Load Recovery")
        ax.legend(); ax.grid(alpha=0.3)

        # Buses energised
        ax = axes[1, 0]
        raw = h["buses"]
        smooth = np.convolve(raw, np.ones(window)/window, mode='valid')
        ax.plot(eps[:len(smooth)], smooth, label=name, color=c, linewidth=1.5)
        ax.set_ylabel("Buses Energised"); ax.set_title("Network Energisation")
        ax.set_xlabel("Episode"); ax.legend(); ax.grid(alpha=0.3)

        # Frequency
        ax = axes[1, 1]
        raw = h["freq"]
        smooth = np.convolve(raw, np.ones(window)/window, mode='valid')
        ax.plot(eps[:len(smooth)], smooth, label=name, color=c, linewidth=1.5)
        ax.axhline(50, color='grey', linestyle='--', alpha=0.5)
        ax.set_ylabel("Frequency (Hz)"); ax.set_title("System Frequency")
        ax.set_xlabel("Episode"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Training DQN Agent")
    print("=" * 60)
    h_dqn = train_dqn()

    print("\n" + "=" * 60)
    print("  Training PPO Agent")
    print("=" * 60)
    h_ppo = train_ppo()

    # Save training histories
    with open(os.path.join(SNAPSHOT_DIR, "training_history.json"), "w") as f:
        json.dump({"DQN": h_dqn, "PPO": h_ppo}, f)

    # Plot comparison
    plot_training({"DQN": h_dqn, "PPO": h_ppo},
                  os.path.join(SNAPSHOT_DIR, "training_curves.png"))

    print("\nDone. All outputs in:", SNAPSHOT_DIR)
