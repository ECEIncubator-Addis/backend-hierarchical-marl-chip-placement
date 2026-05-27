from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.serialization import safe_globals
from torch_geometric.data import Data

# Ensure the ML project's src/ is importable
ML_SRC = Path(__file__).resolve().parents[4] / "hierarchical-marl-chip-placement" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from models.gnn.model import GNNPlacementRegressor  # noqa: E402
from rl.algorithms import PPOAgent, A2CAgent, DQNAgent  # noqa: E402


# Root that holds the preprocessed real-connectivity graphs
GRAPHS_ROOT = ML_SRC / "data" / "preprocessed" / "real-connection"


def list_available_graphs() -> list[dict[str, str]]:
    """Return all design-PDK graph files under GRAPHS_ROOT."""
    results = []
    if not GRAPHS_ROOT.exists():
        return results
    for pt_file in sorted(GRAPHS_ROOT.rglob("*_graph.pt")):
        parts = pt_file.relative_to(GRAPHS_ROOT).parts
        if len(parts) >= 3:
            results.append({
                "design": parts[0],
                "pdk": parts[1],
                "path": str(pt_file),
            })
    return results


def default_graph_path() -> str | None:
    """Return the path of the first available preprocessed graph, or None."""
    graphs = list_available_graphs()
    return graphs[0]["path"] if graphs else None


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_dataset(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    """Load a raw graph .pt file without feature rebuilding."""
    with safe_globals([Data]):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"], data.get("metadata", {})
    if isinstance(data, Data):
        return data, {}
    raise ValueError(f"Unsupported .pt file format: {graph_path}")


def load_graph_with_features(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    """Load a graph and rebuild the 6-feature node matrix (mirrors simulator.py/load_graph)."""
    from configs.rl_envs.simulator import load_graph  # type: ignore[import]
    return load_graph(graph_path)


# ── GNN feature pipeline ─────────────────────────────────────────────────────

def build_gnn_features(
    graph: Data,
    checkpoint_data: dict[str, Any],
) -> torch.Tensor:
    """Build and normalise node features exactly as done in train_gnn.py.

    Uses ``include_position_features`` and ``include_node_type_features`` flags
    stored in the checkpoint.  Falls back to sensible defaults if absent.
    """
    include_positions = bool(checkpoint_data.get("include_position_features", False))
    include_node_type = bool(checkpoint_data.get("include_node_type_features", True))

    if graph.x is None:
        raise ValueError("graph.x is None — cannot build features")
    x = graph.x.float()

    if include_node_type and hasattr(graph, "node_type") and graph.node_type is not None:
        node_type_onehot = F.one_hot(graph.node_type.long(), num_classes=3).float()
        x = torch.cat([x, node_type_onehot], dim=1)

    if include_positions and hasattr(graph, "pos") and graph.pos is not None:
        x = torch.cat([x, graph.pos.float()], dim=1)

    mean = checkpoint_data.get("feature_mean")
    std = checkpoint_data.get("feature_std")
    if mean is not None and std is not None:
        mean_t = torch.as_tensor(mean, dtype=torch.float)
        std_t = torch.as_tensor(std, dtype=torch.float).clamp_min(1e-6)
        x = (x - mean_t) / std_t

    return x


# ── Model loading ─────────────────────────────────────────────────────────────

def load_gnn_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[GNNPlacementRegressor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GNNPlacementRegressor(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
        dropout=checkpoint.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def load_agent(checkpoint_path: str | Path, algorithm: str, device: str = "cpu") -> Any:
    algorithm = algorithm.lower()
    if algorithm == "ppo":
        return PPOAgent.load(checkpoint_path, device=device)
    if algorithm == "a2c":
        return A2CAgent.load(checkpoint_path, device=device)
    if algorithm == "dqn":
        return DQNAgent.load(checkpoint_path, device=device)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


# ── HPWL helper ───────────────────────────────────────────────────────────────

def compute_hpwl(graph: Data, macro_only: bool = True) -> float:
    """Compute half-perimeter wirelength from graph positions.

    When *macro_only* is True only counts edges where both endpoints are hard
    MACRO nodes (node_type == 0), giving a cleaner placement quality metric.
    """
    import numpy as np

    if not hasattr(graph, "edge_index") or graph.edge_index is None or graph.edge_index.numel() == 0:
        return 0.0

    canvas = graph.canvas_size.numpy() if hasattr(graph, "canvas_size") and graph.canvas_size is not None \
        else [400.0, 400.0]

    if graph.pos is None:
        return 0.0
    pos = graph.pos.numpy() * canvas  # absolute µm positions, shape [N, 2]
    src, dst = graph.edge_index.numpy()

    if macro_only and hasattr(graph, "node_type") and graph.node_type is not None:
        node_type = graph.node_type.numpy()
        macro_mask = (node_type[src] == 0) & (node_type[dst] == 0)
        src = src[macro_mask]
        dst = dst[macro_mask]

    if len(src) == 0:
        return 0.0

    dx = np.abs(pos[src, 0] - pos[dst, 0])
    dy = np.abs(pos[src, 1] - pos[dst, 1])
    return float(np.sum(dx + dy)) / 2
