"""
Training script for the DQN Reinforcement-Learning signal controller.

Uses the TrafficSignalEnv simulator; saves the learned policy to
saved_models/rl_controller.pth.
"""

import os
import sys
import io
import json
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl_signal import DQN, TrafficSignalEnv, LANES

SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
STATE_DIM = 6  # 4 queues + time sin/cos
N_ACTIONS = 4


def train_rl(episodes=900, batch_size=64, lr=1e-3, target_update=150,
             gamma=0.99, eps_start=1.0, eps_end=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RL] Device: {device}", flush=True)

    env = TrafficSignalEnv(seed=7)
    model = DQN(STATE_DIM, hidden=128, n_actions=N_ACTIONS).to(device)
    target = DQN(STATE_DIM, hidden=128, n_actions=N_ACTIONS).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay = []
    replay_cap = 20000
    eps = eps_start
    eps_decay = (eps_start - eps_end) / (0.7 * episodes)

    rewards_hist = []

    print(f"[RL] Training DQN over {episodes} episodes...\n", flush=True)

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            if random.random() < eps:
                action = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    q = model(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = int(q.argmax(dim=1).item())

            next_state, reward, done = env.step(action)
            replay.append((state, action, reward, next_state, 1.0 - int(done)))
            if len(replay) > replay_cap:
                replay.pop(0)

            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                s = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
                a = torch.LongTensor([b[1] for b in batch]).unsqueeze(1).to(device)
                r = torch.FloatTensor([b[2] for b in batch]).to(device)
                ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
                d = torch.FloatTensor([b[4] for b in batch]).to(device)

                with torch.no_grad():
                    max_q = target(ns).max(1)[0]
                    target_q = r + gamma * max_q * d

                q_now = model(s).gather(1, a).squeeze(1)
                loss = loss_fn(q_now, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            ep_reward += reward

        eps = max(eps_end, eps - eps_decay)
        rewards_hist.append(float(ep_reward))

        if ep % target_update == 0:
            target.load_state_dict(model.state_dict())

        if ep % 100 == 0:
            print(f"  Episode {ep:4d}/{episodes} | mean reward(100): "
                  f"{np.mean(rewards_hist[-100:]):8.1f} | eps: {eps:.3f}", flush=True)

    _save(model, rewards_hist)

    # Quick evaluation against a random policy
    random_rewards = []
    policy_rewards = []
    for _ in range(30):
        s = env.reset()
        done, tot_r = False, 0.0
        while not done:
            s, r, done = env.step(random.randrange(N_ACTIONS))
            tot_r += r
        random_rewards.append(tot_r)
    for _ in range(30):
        s = env.reset()
        done, tot_r = False, 0.0
        while not done:
            with torch.no_grad():
                q = model(torch.FloatTensor(s).unsqueeze(0).to(device))
                act = int(q.argmax(dim=1).item())
            s, r, done = env.step(act)
            tot_r += r
        policy_rewards.append(tot_r)

    print(f"\n[RL] Random policy mean reward: {np.mean(random_rewards):.1f}")
    print(f"[RL] Learned policy mean reward: {np.mean(policy_rewards):.1f} "
          f"({(np.mean(policy_rewards)/max(np.mean(random_rewards),1e-6)-1)*100:.0f}% better)")
    return model, rewards_hist


def _save(model, rewards_hist):
    os.makedirs(SAVED_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVED_DIR, "rl_controller.pth"))
    meta = {
        "model_type": "DQN_SignalControl",
        "state_dim": STATE_DIM,
        "n_actions": N_ACTIONS,
        "hidden": 128,
        "lanes": LANES,
        "gamma": 0.99,
        "mean_reward_history": rewards_hist[-200:],
    }
    with open(os.path.join(SAVED_DIR, "rl_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [Saved] rl_controller.pth + rl_meta.json -> {SAVED_DIR}", flush=True)


if __name__ == "__main__":
    train_rl(episodes=900)