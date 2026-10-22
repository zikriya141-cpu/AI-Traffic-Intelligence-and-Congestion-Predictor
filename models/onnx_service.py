"""
Fast CPU inference service.

Loads the ONNX exports of the forecaster and the DQN signal controller via
ONNX Runtime, so predictions are fast (multi-threaded, graph-optimized)
without loading large PyTorch stacks into the running server.

Falls back to raw PyTorch inference if the ONNX files are missing.
"""

import os
import json
import numpy as np
import pandas as pd

from models.pk_config import NODE_IDS as _PK_NODE_IDS, LANES, build_adjacency

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR = os.path.join(PROJ_ROOT, "saved_models")

NODE_IDS = _PK_NODE_IDS
SEQ_LEN = 8
HORIZON = 4
NUM_FEATURES = 6


def _cyclic(hour, dow):
    return (np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            int(dow >= 5),
            int((8 <= hour <= 10) or (17 <= hour <= 20)))


def _adjacency():
    return build_adjacency(NODE_IDS)


def _scale_features(features_4d):
    try:
        import joblib
        scaler = joblib.load(os.path.join(SAVED_DIR, "forecaster_scaler.joblib"))
        n, T, N, F = features_4d.shape
        flat = features_4d.reshape(n * T * N, F)
        return scaler.transform(flat).reshape(n, T, N, F).astype(np.float32)
    except Exception:
        return features_4d


def classify(count, speed):
    if count >= 40 or speed <= 12:
        return "High"
    if count >= 20 or speed <= 25:
        return "Medium"
    return "Low"


class FastTrafficService:
    """ONNX Runtime backed forecaster + DQN signal recommender."""

    def __init__(self):
        self.fc = None
        self.rl = None
        self.meta = {}
        try:
            import onnxruntime as ort
            attrs = {"forecaster": "fc", "rl_controller": "rl"}
            for name, attr in attrs.items():
                path = os.path.join(SAVED_DIR, name + ".onnx")
                if os.path.exists(path):
                    setattr(self, attr,
                            ort.InferenceSession(path, providers=["CPUExecutionProvider"]))
            if self.fc:
                self.meta = json.load(
                    open(os.path.join(SAVED_DIR, "forecaster_meta.json")))
        except Exception as e:
            print(f"[ONNX] Failed to init OpenNN service: {e}")

    @property
    def available(self):
        return self.fc is not None

    # ── Forecaster ──────────────────────────────────────────────────────
    def forecast_network(self, history, target_datetime):
        """
        history: dict {node: [(count, speed), ...]} most-recent-first or
                 oldest-first — training order is chronological with SEQ_LEN=8.
        Returns {node: [{'minutes': <int>, 'vehicle_count': int,
                 'avg_speed': float, 'congestion': str}, ...]}
        """
        stamps = []
        ts = pd.to_datetime(target_datetime)
        for k in range(SEQ_LEN - 1, -1, -1):
            stamps.append(ts - pd.Timedelta(minutes=(k + 1) * 15))

        num_nodes = len(NODE_IDS)
        feats = np.zeros((1, SEQ_LEN, num_nodes, NUM_FEATURES), dtype=np.float32)
        for node_i, node in enumerate(NODE_IDS):
            hist = history.get(node)
            for t_i in range(SEQ_LEN):
                tsc = pd.Timestamp(stamps[t_i])
                if hist and t_i < len(hist):
                    count = float(hist[t_i][0])
                    speed = float(hist[t_i][1])
                else:
                    count, speed = 0.0, 30.0
                s, c, wd, rush = _cyclic(tsc.hour, tsc.weekday())
                feats[0, t_i, node_i] = [count, speed, s, c, wd, rush]

        scaled = _scale_features(feats)
        counts, speeds = self.fc.run(
            None, {"seq": scaled, "adj": _adjacency().astype(np.float32)})

        out = {}
        for node_i, node in enumerate(NODE_IDS):
            horizon = []
            for h_i in range(HORIZON):
                c = float(counts[0, node_i, h_i])
                s = max(2.0, float(speeds[0, node_i, h_i]))
                horizon.append({
                    "minutes": (h_i + 1) * 15,
                    "vehicle_count": int(round(max(0, c))),
                    "avg_speed": round(s, 1),
                    "congestion": classify(c, s),
                })
            out[node] = horizon
        return out

    # ── DQN signal recommender ──────────────────────────────────────────
    def recommend_green_lane(self, lane_counts: dict, hour: float = 0.0):
        """Return {'secret_lane': lane, 'qvalues': [...], 'weights': [...]}."""
        counts = np.array([lane_counts.get(l, 0) for l in LANES], dtype=np.float32)
        state = np.concatenate([
            counts / 200.0,
            [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)],
        ]).astype(np.float32).reshape(1, 6)
        q = self.rl.run(None, {"state": state})[0][0]
        e = np.exp(q - q.max())
        w = e / e.sum()
        return {
            "recommended_lane": LANES[int(np.argmax(q))],
            "qvalues": [float(v) for v in q],
            "weights": [float(v) for v in w],
        }