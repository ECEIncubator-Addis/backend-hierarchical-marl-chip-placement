from __future__ import annotations

import torch
from fastapi import APIRouter

from app.api.routes.utils import (
    build_gnn_features,
    load_agent,
    load_gnn_model,
    load_graph_with_features,
)


router = APIRouter(tags=["model-requests"], prefix="/model-requests")


@router.get("/check")
async def check_health():
    return {"message": "success"}


@router.get("/gnn-embedding-available")
async def check_gnn_health(model_path: str):
    """Check whether a GNN checkpoint can be loaded successfully."""
    try:
        device = torch.device("cpu")
        load_gnn_model(model_path, device)
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "fault", "error": str(e)}


@router.get("/gnn-embedding")
async def get_gnn_embedding(model_path: str, graph_path: str):
    """Return GNN node embeddings for a graph using a trained checkpoint.

    The feature pipeline (node-type one-hot, position concat, mean/std
    normalisation) is reconstructed from flags stored in the checkpoint,
    so results are consistent with how the model was trained.

    Returns embeddings only for hard MACRO nodes when the graph has a
    ``placeable_mask`` attribute; otherwise returns embeddings for all nodes.
    """
    try:
        device = torch.device("cpu")
        model, checkpoint = load_gnn_model(model_path, device)

        # load_graph_with_features rebuilds the 6-feature x vector from w/h/edge_attr
        graph, _ = load_graph_with_features(graph_path)

        x = build_gnn_features(graph, checkpoint).to(device)
        if graph.edge_index is None:
            return {"status": "fault", "error": "graph has no edge_index"}
        edge_index = graph.edge_index.to(device)

        with torch.no_grad():
            embeddings = model.encoder(x, edge_index)  # [N, out_channels]

        # Return macro-only embeddings when the graph supports it
        placeable = getattr(graph, "placeable_mask", None)
        if placeable is not None:
            embeddings = embeddings[placeable.bool()]

        return {
            "embeddings": embeddings.cpu().tolist(),
            "num_nodes": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
        }
    except Exception as e:
        return {"status": "fault", "error": str(e)}


@router.get("/rl-policy-available")
async def check_rl_policy_health(model_path: str, algorithm: str = "ppo"):
    """Check whether an RL checkpoint can be loaded successfully."""
    try:
        load_agent(model_path, algorithm)
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "fault", "error": str(e)}


@router.get("/rl-policy-action")
async def get_rl_policy_action(model_path: str, algorithm: str, graph_path: str):
    """Return the action selected by a trained RL agent for the given graph.

    The agent receives the environment observation built from the graph's
    current positions and node features, exactly as the env produces it
    at each step.

    Returns:
        macro_index  : which macro to move (0-indexed among all nodes)
        direction    : movement direction index (0-3 or broader, agent-specific)
        action       : flat action integer used by the env's action space
    """
    try:
        import sys
        from pathlib import Path

        ml_src = Path(__file__).resolve().parents[4] / "hierarchical-marl-chip-placement" / "src"
        if str(ml_src) not in sys.path:
            sys.path.insert(0, str(ml_src))

        from models.macro_placement_env import MacroPlacementEnv  # type: ignore[import]

        agent = load_agent(model_path, algorithm)
        env = MacroPlacementEnv(graph_path)
        obs, _ = env.reset()

        with torch.no_grad():
            action_tuple = agent.act(obs, deterministic=True)

        # PPO/A2C return (action, log_prob, value); DQN may return just action
        flat_action = int(action_tuple[0]) if isinstance(action_tuple, tuple) else int(action_tuple)
        num_directions = getattr(agent, "num_directions", 4)
        macro_index = flat_action // num_directions
        direction = flat_action % num_directions

        return {
            "action": flat_action,
            "macro_index": macro_index,
            "direction": direction,
        }
    except Exception as e:
        return {"status": "fault", "error": str(e)}
