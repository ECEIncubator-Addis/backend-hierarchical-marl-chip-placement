import torch 
from fastapi import APIRouter
from app.api.routes.utils import load_dataset, load_gnn_model, load_agent





router = APIRouter(tags=["model-requests"], prefix="/model-requests")
@router.get("/check")
async def check_health():
    return {"message": "success"}



@router.get("/gnn-embedding-available")
async def check_gnn_health(model_path: str):
    """
        Args:
            model_path: a valid model location in the local filesystem
        Returns:
            status: healthy / fault
    """

    try:
        device = torch.device("cpu")
        model, _ = load_gnn_model(model_path, device)
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "fault", "error": str(e)}
    

@router.get("/gnn-embedding")
async def get_gnn_embedding(model_path: str, graph_path: str):
    """
        Args:
            model_path: a valid model location in the local filesystem
            graph_path: a valid graph location in the local filesystem
        Returns:
            embeddings: the GNN embeddings for the input graph
    """
    try:
        device = torch.device("cpu")
        model, _ = load_gnn_model(model_path, device)
        graph, _ = load_dataset(graph_path)
        with torch.no_grad():
            embeddings = model(graph.x, graph.edge_index)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        return {"status": "fault", "error": str(e)}
    
@router.get("/rl-policy-available")
async def check_rl_policy_health(model_path: str):
    """
        Args:
            model_path: a valid model location in the local filesystem
        Returns:    
            status: healthy / fault
    """

    try:
        device = torch.device("cpu")
        model, _ = load_gnn_model(model_path, device)
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "fault", "error": str(e)}
    
@router.get("/rl-policy-action")
async def get_rl_policy_action(model_path: str, algorithm: str, graph_path: str):
    """
        Args:
            model_path: a valid model location in the local filesystem
            algorithm: the RL algorithm to use (e.g., "ppo", "a2c", "dqn")
            graph_path: a valid graph location in the local filesystem
        Returns:
            action: the RL policy action for the input graph
    """
    try:
        model, _ = load_agent(model_path, algorithm)
        graph, _ = load_dataset(graph_path)
        with torch.no_grad():
            action = model.act(graph)
        return {"action": action.tolist()}
    except Exception as e:
        return {"status": "fault", "error": f"Failed to load agent: {str(e)}"}
    



