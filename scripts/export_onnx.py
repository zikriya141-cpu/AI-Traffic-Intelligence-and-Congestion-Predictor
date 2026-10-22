"""
Export trained PyTorch models to ONNX so the server can run FAST inference
on CPU via ONNX Runtime (uses all available cores + graph optimizations).

Exports:
    saved_models/forecaster.onnx     — spatial-temporal forecaster
    saved_models/rl_controller.onnx  — DQN signal controller
"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import numpy as np

SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")


def main():
    device = "cpu"
    torch.set_num_threads(os.cpu_count() or 1)

    # ---- Forecaster ----
    fc_path = os.path.join(SAVED_DIR, "forecaster.pth")
    if os.path.exists(fc_path):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.forecaster import SpatialTemporalForecaster, build_adjacency
        from models.pk_config import NODE_IDS
        meta = __import__("json").load(open(os.path.join(SAVED_DIR, "forecaster_meta.json")))
        node_ids = meta.get("node_ids", NODE_IDS)
        num_nodes = len(node_ids)
        seq_len = meta.get("seq_len", 8)
        model = SpatialTemporalForecaster(
            num_nodes=num_nodes, input_dim=6, hidden_dim=96, num_layers=2,
            forecast_horizon=meta.get("horizon", 4))
        model.load_state_dict(torch.load(fc_path, map_location="cpu", weights_only=True))
        model.eval()

        x = torch.randn(1, seq_len, num_nodes, 6)
        adj = build_adjacency(node_ids)
        torch.onnx.export(
            model, (x, adj),
            os.path.join(SAVED_DIR, "forecaster.onnx"),
            input_names=["seq", "adj"],
            output_names=["counts", "speeds"],
            opset_version=13,
            dynamo=False,
            dynamic_axes={"seq": {0: "batch"}},
        )
        print("[ONNX] forecaster.onnx exported", flush=True)
    else:
        print("[ONNX] forecaster.pth missing - skip", flush=True)

    # ---- RL controller ----
    rl_path = os.path.join(SAVED_DIR, "rl_controller.pth")
    if os.path.exists(rl_path):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.rl_signal import DQN
        model = DQN(state_dim=6, hidden=128, n_actions=4)
        model.load_state_dict(torch.load(rl_path, map_location="cpu", weights_only=True))
        model.eval()
        s = torch.randn(1, 6)
        torch.onnx.export(
            model, s,
            os.path.join(SAVED_DIR, "rl_controller.onnx"),
            input_names=["state"],
            output_names=["qvalues"],
            opset_version=13,
            dynamo=False,
            dynamic_axes={"state": {0: "batch"}},
        )
        print("[ONNX] rl_controller.onnx exported", flush=True)
    else:
        print("[ONNX] rl_controller.pth missing - skip", flush=True)


if __name__ == "__main__":
    main()