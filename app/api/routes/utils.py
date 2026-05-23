import torch
from pathlib import Path
from typing import Any
from torch.serialization import safe_globals
from torch_geometric.data import Data
from models.gnn.model import GNNPlacementRegressor
from rl.algorithms import PPOAgent, A2CAgent, DQNAgent

import sys


def load_dataset(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    with safe_globals([Data]):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"], data.get("metadata", {})
    if isinstance(data, Data):
        return data, {}
    raise ValueError(f"Unsupported .pt file format: {graph_path}")


def load_gnn_model(checkpoint_path: str | Path, device: torch.device) -> tuple[GNNPlacementRegressor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
