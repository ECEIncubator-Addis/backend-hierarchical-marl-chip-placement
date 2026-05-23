import asyncio
import time
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.api.main import api_router
from app import db as app_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    # initialize MongoDB (uses default localhost URI and database name)
    await app_db.init_db()


@app.on_event("shutdown")
async def shutdown_event():
    await app_db.close_db()

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

if __name__ == "__main__":
    import uvicorn
    print("Mock backend running on http://localhost:3001")
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
