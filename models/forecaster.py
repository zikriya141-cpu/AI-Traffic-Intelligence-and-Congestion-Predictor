"""
Spatial-Temporal Graph Forecaster (ST-GRU + Graph Fusion).

Advanced traffic forecasting that models:
  - Temporal dynamics per intersection (shared GRU encoder/decoder)
  - Spatial correlation ACROSS intersections (graph convolution over a
    weighted adjacency — nearby junctions influence each other)

Predicts multi-step future traffic: vehicle count + avg speed at
+15 / +30 / +45 / +60 minutes for every intersection in the network.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pk_config import build_adjacency as _build_adjacency, NODE_IDS as _NODE_IDS

FEATURE_NAMES = [
    "vehicle_count", "avg_speed",
    "time_sin", "time_cos",
    "is_weekend", "is_rush_hour",
]


class GraphConv(nn.Module):
    """Lightweight graph convolution layer using a normalized adjacency matrix."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        # x: (B, N, F) -> message passing: (B, N, F) @ A^T
        x = torch.matmul(adj.unsqueeze(0), x)
        return F.relu(self.bn(self.linear(x)))


class SpatialTemporalForecaster(nn.Module):
    """
    GRU encoder -> graph fusion -> GRU decoder -> multi-step head.

    Args:
        num_nodes: number of intersections in the network
        input_dim: features per time step
        hidden_dim: GRU hidden size
        num_layers: stacked GRU depth
        forecast_horizon: number of future steps to predict
    """

    def __init__(self, num_nodes=6, input_dim=6, hidden_dim=96, num_layers=2,
                 forecast_horizon=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.forecast_horizon = forecast_horizon

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True)
        self.gc1 = GraphConv(hidden_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.decoder = nn.GRUCell(hidden_dim, hidden_dim)

        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, adj):
        """
        x: (B, T, N, F) — per-node temporal sequences
        adj: (N, N) float adjacency matrix
        Returns:
            counts (B, N, H), speeds (B, N, H)
        """
        B, T, N, F = x.shape
        # Shared GRU over all nodes
        flat = x.reshape(B * N, T, F)
        _, h = self.encoder(flat)                      # (layers, B*N, hidden)
        h = h[-1].reshape(B, N, -1)                    # (B, N, hidden)

        # Spatial fusion (graph convolution)
        h = self.gc1(h, adj)
        h = self.gc2(h, adj)

        # Iterative multi-step decode
        counts, speeds = [], []
        state = h.reshape(B * N, -1)
        for _ in range(self.forecast_horizon):
            state = self.decoder(state, state)
            c = self.count_head(state)
            s = self.speed_head(state)
            counts.append(c)
            speeds.append(s)

        counts = torch.cat(counts, dim=1).reshape(B, N, -1)
        speeds = torch.cat(speeds, dim=1).reshape(B, N, -1)
        return counts, speeds

    def forecast(self, x, adj):
        """Convenience inference returning numpy arrays."""
        self.eval()
        with torch.no_grad():
            c, s = self.forward(x, adj)
        return c, s


def build_adjacency(node_ids=None):
    """Build normalized adjacency matrix (with self-loops) for given node order."""
    if node_ids is None:
        node_ids = _NODE_IDS
    return torch.FloatTensor(_build_adjacency(node_ids))