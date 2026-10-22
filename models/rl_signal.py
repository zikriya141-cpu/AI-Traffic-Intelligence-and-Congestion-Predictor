"""
Reinforcement Learning Signal Controller (DQN).

The controller is an AI agent that LEARNS the best junction policy instead of
using fixed rules. It balances:
  - minimizing total stopped vehicles (queues)
  - avoiding overflow on any single approach

State:  normalized queue per lane + time-of-day context
Actions: which approach gets green (4 actions)
Reward: -weighted sum of queues - heavy overflow penalties

Trained in a stochastic simulator that mimics Indian heterogeneous traffic
peak / off-peak / night patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LANES = ["Lane_N", "Lane_S", "Lane_E", "Lane_W"]


class DQN(nn.Module):
    """Deep Q-Network: state -> Q-values for each of the 4 lane actions."""

    def __init__(self, state_dim: int, hidden: int = 128, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class TrafficSignalEnv:
    """
    Simulated 4-approach junction.

    Each 'tick' is 15 s. Arrival rates vary by hour of day to reproduce
    morning/evening rush and night lulls. The chosen green approach is
    serviced (cars released), others accumulate.
    """

    def __init__(self, seed: int = 7):
        self.rng = np.random.RandomState(seed)
        self.queues = np.zeros(4, dtype=np.float32)
        self.tick = 0
        self.period = 96  # 96 ticks = 24 min per simulated day (scaled)
        self.service = 6.0   # cars released per green tick

    def _arrival_rate(self):
        h = (self.tick % self.period) / self.period * 24.0
        if 8 <= h <= 10:
            base = [3.2, 1.8, 2.6, 1.4]
        elif 17 <= h <= 20:
            base = [2.8, 1.6, 2.2, 1.3]
        elif h < 6:
            base = [0.4, 0.2, 0.3, 0.2]
        else:
            base = [1.4, 0.8, 1.1, 0.7]
        return np.array(base, dtype=np.float32)

    def reset(self):
        self.queues = self.rng.uniform(0, 8, 4).astype(np.float32)
        self.tick = 0
        return self._state()

    def _state(self):
        h = (self.tick % self.period) / self.period * 24.0
        return np.concatenate([
            self.queues / 30.0,
            [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)],
        ]).astype(np.float32)

    def step(self, action):
        rates = self._arrival_rate()
        arrivals = self.rng.poisson(rates).astype(np.float32)
        self.queues += arrivals
        # Service chosen approach
        self.queues[action] = max(0.0, self.queues[action] - self.service)

        overflow = np.maximum(0.0, self.queues - 40.0).sum()
        reward = -0.1 * self.queues.sum() - 2.0 * overflow
        self.tick += 1
        done = self.tick >= self.period
        return self._state(), float(reward), done


def select_greedy(model: DQN, state: np.ndarray, device="cpu"):
    """Pick the lane with the highest Q-value."""
    with torch.no_grad():
        q = model(torch.FloatTensor(state).unsqueeze(0).to(device))
    return int(q.argmax(dim=1).item())


def ai_lane_weights(model: DQN, lane_counts: dict, device="cpu"):
    """
    Softmax over DQN Q-values to produce AI-informed per-lane preference
    weights. Used to blend rule-based proportional allocation with the
    learned policy.
    """
    counts = np.array([lane_counts.get(l, 0) for l in LANES], dtype=np.float32)
    state = np.concatenate([counts / 200.0,
                            [np.sin(2 * np.pi * 0.0), np.cos(2 * np.pi * 0.0)]])
    with torch.no_grad():
        q = model(torch.FloatTensor(state).unsqueeze(0).to(device))
    w = F.softmax(q, dim=1)[0].cpu().numpy()
    return {lane: float(wi) for lane, wi in zip(LANES, w)}