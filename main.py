import asyncio
import time
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from decouple import config
from motor.motor_asyncio import AsyncIOMotorClient


from app.api.main import api_router
from app.core import db as app_db

from contextlib import asynccontextmanager

# Configure max upload size (100MB)
import os
os.environ.setdefault("STARLETTE_MAX_UPLOAD_SIZE", "104857600")


DB_URL = config("DB_URL", cast=str)
DB_NAME = config("DB_NAME", cast=str)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_db.init_db(uri=DB_URL, db_name=DB_NAME)
    print(f"Connected to MongoDB at {DB_URL}, using database '{DB_NAME}'")
    
    # Pre-load default design if database is empty
    designs = await app_db.list_designs()
    if not designs:
        print("Database is empty. Loading default ariane133 design...")
        try:
            from app.api.routes.designs import extract_design_from_pt
            import json
            from pathlib import Path
            from app.models import Design
            
            base_path = Path("/home/quantap/Documents/projects_/yearly-project/hierarchical-marl-chip-placement/src/data/preprocessed/real-connection/ariane133/Nangate45")
            pt_path = base_path / "ariane133_Nangate45_graph.pt"
            json_path = base_path / "metadata-ariane133-Nangate45.json"
            
            if pt_path.exists() and json_path.exists():
                with open(pt_path, "rb") as f:
                    pt_data = f.read()
                design_data = extract_design_from_pt(pt_data, "ariane133_Nangate45_graph.pt")
                
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                    
                design_payload = {
                    "name": "ariane133_Nangate45_graph",
                    "metadata": metadata,
                    "data": design_data,
                }
                
                await app_db.create_design(Design(**design_payload))
                print("Successfully loaded default design.")
            else:
                print("Default design files not found.")
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



jobs: Dict[str, Any] = {}

class OptimizeRequest(BaseModel):
    algo: Optional[str] = "PPO"
    useGNN: Optional[bool] = True
    timesteps: Optional[int] = 10000
    hpwlBefore: Optional[float] = 1000.0

async def background_optimization(job_id: str, hpwl_before: float):
    p = 0
    while p < 100:
        await asyncio.sleep(0.6)
        p += 4 + random.random() * 6
        if p >= 100:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["metrics"] = {
                "reward": round(random.random() * 6 + 7, 2),
                "episodeLen": 512,
                "hpwlBefore": hpwl_before,
                "hpwlAfter": hpwl_before * (1 - (random.random() * 0.1 + 0.1)),
            }
            break
        else:
            jobs[job_id]["progress"] = min(99, p)




@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    # Create a base36-like hex string for ID to match the old format somewhat closely
    job_id = f"job_{int(time.time() * 1000):x}"
    jobs[job_id] = {
        "id": job_id,
        "status": "running",
        "progress": 0,
        "metrics": None
    }
    asyncio.create_task(background_optimization(job_id, req.hpwlBefore or 1000.0))
    return {"id": job_id}

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

# Register API routes under /api
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    print("Mock backend running on http://localhost:3001")
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
