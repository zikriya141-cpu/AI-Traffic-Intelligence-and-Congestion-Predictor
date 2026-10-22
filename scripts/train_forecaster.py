"""
Training script for the Spatial-Temporal Graph Forecaster.

Builds per-intersection 15-minute traffic series from the CSV, constructs
multi-step forecasting windows (input 2h -> predict +1h at 15-min steps),
and trains the ST-GRU model with spatial fusion across the road network.

Speed optimizations for CPU:
  - uses ALL physical CPU cores via torch.set_num_threads
  - subsamples windows so training stays fast on consumer hardware

Outputs (saved_models/):
    forecaster.pth            — model weights
    forecaster_scaler.joblib  — feature scaler
    forecaster_meta.json      — model/network metadata
"""

import os
import sys
import io
import json
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim

# --- Use all physical CPU cores for torch ---
def best_threads():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        return os.cpu_count() or 1

torch.set_num_threads(best_threads())
print(f"[Forecaster] PyTorch CPU threads: {torch.get_num_threads()}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data_processor import load_and_engineer
from models.forecaster import (
    SpatialTemporalForecaster, build_adjacency, FEATURE_NAMES,
)
from models.pk_config import NODE_IDS

SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")

SEQ_LEN = 8    # 8 x 15min = 2 hours of history
HORIZON = 4    # +15, +30, +45, +60 minutes
STEP_MIN = 15
NODE_COUNT = len(NODE_IDS)
NUM_FEATURES = 6


def pd_to_dt(series):
    import pandas as pd
    return pd.to_datetime(series, errors="coerce")


def build_timeseries(df):
    """Aggregate CSV to per-intersection 15-min series."""
    df = df.copy()
    df["timestamp"] = pd_to_dt(df["timestamp"])
    df["ts15"] = df["timestamp"].dt.floor(f"{STEP_MIN}min")
    agg = df.groupby(["intersection_id", "ts15"]).agg(
        vehicle_count=("vehicle_count", "sum"),
        avg_speed=("avg_speed", "mean"),
    ).reset_index()
    return agg


def make_window_dataset(df_agg):
    """Build (n, SEQ+H, NODES, FEATURES) windows over ALL nodes at once."""
    import pandas as pd

    def cyclic(hour, dow):
        return (np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                int(dow >= 5),
                int((8 <= hour <= 10) or (17 <= hour <= 20)))

    pivot = df_agg.pivot(index="ts15", columns="intersection_id",
                         values=["vehicle_count", "avg_speed"])
    pivot = pivot.sort_index().ffill().bfill().dropna()
    # Reorder columns to the canonical NODE_IDS order
    pivot = pivot.reindex(columns=NODE_IDS, level=1)
    counts = pivot["vehicle_count"].values.astype(np.float32)   # (T, N)
    speeds = pivot["avg_speed"].values.astype(np.float32)       # (T, N)
    stamps = pivot.index.values
    n = len(pivot)

    out = []
    for i in range(n - SEQ_LEN - HORIZON + 1):
        seq = np.zeros((SEQ_LEN + HORIZON, NODE_COUNT, NUM_FEATURES), dtype=np.float32)
        for t in range(i, i + SEQ_LEN + HORIZON):
            ts = pd.Timestamp(stamps[t])
            s, c, wd, rush = cyclic(ts.hour, ts.dayofweek)
            seq[t - i, :, 0] = counts[t]
            seq[t - i, :, 1] = speeds[t]
            seq[t - i, :, 2] = s
            seq[t - i, :, 3] = c
            seq[t - i, :, 4] = wd
            seq[t - i, :, 5] = rush
        out.append(seq)
    return np.array(out, dtype=np.float32)


def train_forecaster(epochs=30, batch_size=512, lr=1e-3, max_windows=15000):
    import joblib
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import Dataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Forecaster] Device: {device}", flush=True)

    df = load_and_engineer()
    series = build_timeseries(df)

    arr = make_window_dataset(series)
    print(f"[Forecaster] Total windows: {len(arr):,}", flush=True)

    # Subsample for fast CPU training (keeps temporal spread via shuffle)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(arr))
    n_keep = min(len(arr), max_windows)
    keep = perm[:n_keep]
    arr = arr[keep]
    print(f"[Forecaster] Training windows: {len(arr):,}", flush=True)

    n_train = int(len(arr) * 0.8)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(arr))

    # Feature scaler (flatten per-sample features)
    scaler = StandardScaler()
    scaler.fit(arr[:, :SEQ_LEN, :, :].reshape(-1, NUM_FEATURES))

    def prep(sub_idx):
        data_x = arr[sub_idx, :SEQ_LEN, :, :].astype(np.float32)   # (n, T, N, F)
        data_y = arr[sub_idx, SEQ_LEN:, :, :2].astype(np.float32)  # (n, H, N, 2)
        n, T, Nn, F = data_x.shape
        flat = data_x.reshape(n * T * Nn, F)
        data_x = scaler.transform(flat).reshape(n, T, Nn, F)
        return data_x, data_y

    Xtr, Ytr = prep(train_idx)
    Xte, Yte = prep(test_idx)

    adj = build_adjacency(NODE_IDS).to(device)
    model = SpatialTemporalForecaster(
        num_nodes=NODE_COUNT, input_dim=6, hidden_dim=96, num_layers=2,
        forecast_horizon=HORIZON,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    class DS(Dataset):
        def __init__(self, x, y):
            self.x = torch.FloatTensor(x)
            self.y = torch.FloatTensor(y)
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return (self.x[i], self.y[i])

    tr_loader = DataLoader(DS(Xtr, Ytr), batch_size=batch_size, shuffle=True,
                           num_workers=0, drop_last=True)
    te_loader = DataLoader(DS(Xte, Yte), batch_size=batch_size, shuffle=False,
                           num_workers=0)

    print(f"[Forecaster] Train: {len(tr_loader.dataset):,} | Test: {len(te_loader.dataset):,}", flush=True)
    print(f"[Forecaster] Seq: {SEQ_LEN} steps | Horizon: {HORIZON} x 15min", flush=True)

    best_mae = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tot, tot_mae = 0.0, 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            counts, speeds = model(xb, adj)                    # (B, N, H)
            yc = yb[..., 0].permute(0, 2, 1)
            loss = mse(counts, yc) + 0.5 * mse(speeds, yb[..., 1].permute(0, 2, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item() * xb.size(0)
            tot_mae += torch.abs(counts.detach() - yc).sum().item()

        train_loss = tot / max(len(tr_loader.dataset), 1)

        model.eval()
        vtot, vmae = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(device), yb.to(device)
                counts, speeds = model(xb, adj)
                yc = yb[..., 0].permute(0, 2, 1)
                loss = mse(counts, yc) + 0.5 * mse(speeds, yb[..., 1].permute(0, 2, 1))
                vtot += loss.item() * xb.size(0)
                vmae += torch.abs(counts.detach() - yc).sum().item()

        val_loss = vtot / max(len(te_loader.dataset), 1)
        val_mae = vmae / max(len(te_loader.dataset), 1) / HORIZON / NODE_COUNT
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        improved = ""
        if val_mae < best_mae:
            best_mae = val_mae
            improved = " *BEST*"
            _save(model, scaler, {
                "model_type": "STGRU_GraphFusion",
                "num_nodes": NODE_COUNT,
                "input_dim": NUM_FEATURES,
                "hidden_dim": 96,
                "num_layers": 2,
                "seq_len": SEQ_LEN,
                "horizon": HORIZON,
                "step_minutes": STEP_MIN,
                "node_ids": NODE_IDS,
                "feature_names": FEATURE_NAMES,
                "best_val_mae": float(best_mae),
                "history": history,
            })

        print(f"  Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | "
              f"ValLoss: {val_loss:.4f} | ValMAE: {val_mae:.3f} veh{improved}",
              flush=True)

    print(f"\n[Forecaster] Done. Best MAE (per node per horizon): {best_mae:.3f} vehicles", flush=True)
    return model, history


def _save(model, scaler, meta):
    os.makedirs(SAVED_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVED_DIR, "forecaster.pth"))
    import joblib
    joblib.dump(scaler, os.path.join(SAVED_DIR, "forecaster_scaler.joblib"))
    with open(os.path.join(SAVED_DIR, "forecaster_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [Saved] forecaster.pth + scaler + meta -> {SAVED_DIR}", flush=True)


if __name__ == "__main__":
    train_forecaster(epochs=30, batch_size=512, lr=1e-3)