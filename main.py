from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import os
os.environ.setdefault("STARLETTE_MAX_UPLOAD_SIZE", "104857600")

from decouple import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.main import api_router
from app.core import db as app_db

# Ensure the ML project's src/ directory is importable by the entire process
_ML_SRC = Path(__file__).resolve().parent.parent / "hierarchical-marl-chip-placement" / "src"
if str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

DB_URL = config("DB_URL", cast=str)
DB_NAME = config("DB_NAME", cast=str)

# Default graph used when the request doesn't specify one
_DEFAULT_GRAPH = (
    _ML_SRC
    / "data" / "preprocessed" / "real-connection"
    / "ariane133" / "NanGate45" / "ariane133_NanGate45_graph.pt"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_db.init_db(uri=DB_URL, db_name=DB_NAME)
    print(f"Connected to MongoDB at {DB_URL}, using database '{DB_NAME}'")

    designs = await app_db.list_designs()
    if not designs:
        print("Database is empty. Loading default ariane133 design…")
        try:
            from app.api.routes.designs import extract_design_from_pt
            import json
            from app.models import Design

            pt_path = _DEFAULT_GRAPH
            json_path = pt_path.parent / f"metadata-ariane133-NanGate45.json"

            if pt_path.exists() and json_path.exists():
                with open(pt_path, "rb") as f:
                    pt_data = f.read()
                design_data = extract_design_from_pt(pt_data, pt_path.name)

                with open(json_path, "r") as f:
                    metadata = json.load(f)

                await app_db.create_design(Design(
                    name="ariane133_NanGate45_graph",
                    metadata=metadata,
                    data=design_data,
                ))
                print("Default design loaded.")
            else:
                print(f"Default graph not found at {pt_path}")
        except Exception as e:
            print(f"Failed to load default design: {e}")

    yield
    await app_db.close_db()


app = FastAPI(
    lifespan=lifespan,
    title="Hierarchical MARL Chip Placement Backend",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-process job registry ───────────────────────────────────────────────────

jobs: Dict[str, Any] = {}


class OptimizeRequest(BaseModel):
    algo: Optional[str] = "ppo"
    useGNN: Optional[bool] = True
    episodes: Optional[int] = 3
    graph_path: Optional[str] = None       # path to a preprocessed .pt graph
    checkpoint_path: Optional[str] = None  # path to a trained RL checkpoint


# ── Real optimization worker (runs in a thread pool) ─────────────────────────

def _run_optimization_sync(
    job_id: str,
    graph_path: str,
    algo: str,
    checkpoint_path: Optional[str],
    episodes: int,
) -> None:
    """Synchronous placement optimization executed in a thread-pool worker.

    Loads the graph, runs the RL agent (or a random policy if no checkpoint is
    supplied) for the requested number of episodes, and records real HPWL
    metrics in the shared *jobs* dict.
    """
    from app.api.routes.utils import compute_hpwl, load_agent, load_dataset
    from models.macro_placement_env import MacroPlacementEnv  # type: ignore[import]

    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 5

        # ── Load graph and compute initial HPWL ──────────────────────────
        graph, _ = load_dataset(graph_path)
        hpwl_before = compute_hpwl(graph, macro_only=True)
        jobs[job_id]["hpwlBefore"] = round(hpwl_before, 2)
        jobs[job_id]["progress"] = 10

        # ── Build environment ────────────────────────────────────────────
        env = MacroPlacementEnv(graph_path)

        # ── Load agent or fall back to random policy ─────────────────────
        agent = None
        if checkpoint_path:
            try:
                agent = load_agent(checkpoint_path, algo)
            except Exception as e:
                jobs[job_id]["warnings"] = [f"Checkpoint load failed ({e}); using random policy"]

        # ── Run episodes ─────────────────────────────────────────────────
        best_hpwl = hpwl_before
        total_reward = 0.0

        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                if agent is not None:
                    try:
                        action_out = agent.act(obs, deterministic=False)
                        action = int(action_out[0]) if isinstance(action_out, tuple) else int(action_out)
                    except Exception:
                        action = env.action_space.sample()
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += float(reward)

            # HPWL after this episode (macro-only, using live graph positions)
            ep_hpwl = compute_hpwl(env.graph, macro_only=True)
            if ep_hpwl < best_hpwl or best_hpwl == 0.0:
                best_hpwl = ep_hpwl
            total_reward += ep_reward

            progress = 10 + int(85 * (ep + 1) / episodes)
            jobs[job_id]["progress"] = progress

        # ── Final metrics ────────────────────────────────────────────────
        hpwl_after = compute_hpwl(env.graph, macro_only=True)
        improvement_pct = (
            (hpwl_before - hpwl_after) / hpwl_before * 100.0
            if hpwl_before > 0 else 0.0
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["metrics"] = {
            "reward": round(total_reward / max(episodes, 1), 4),
            "episodeLen": env.max_steps,
            "hpwlBefore": round(hpwl_before, 2),
            "hpwlAfter": round(hpwl_after, 2),
            "hpwlBest": round(best_hpwl, 2),
            "improvementPct": round(improvement_pct, 2),
            "episodes": episodes,
            "policy": "checkpoint" if agent is not None else "random",
        }

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["error"] = str(exc)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    """Submit a placement optimization job.

    Runs the RL agent (or a random policy when no checkpoint is supplied) on
    the specified graph and returns a job ID for polling via GET /api/jobs/{id}.
    """
    job_id = f"job_{int(time.time() * 1000):x}"

    graph_path = req.graph_path or str(_DEFAULT_GRAPH)
    if not Path(graph_path).exists():
        raise HTTPException(status_code=400, detail=f"Graph not found: {graph_path}")

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "metrics": None,
        "graph_path": graph_path,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_optimization_sync,
        job_id,
        graph_path,
        (req.algo or "ppo").lower(),
        req.checkpoint_path,
        max(1, req.episodes or 3),
    )

    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/graphs")
async def list_graphs():
    """List all preprocessed graphs available for optimization."""
    from app.api.routes.utils import list_available_graphs
    return list_available_graphs()


# Register the remaining API routes
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    print(f"Backend running on http://localhost:3001")
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
