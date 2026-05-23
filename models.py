from pydantic import BaseModel
from typing import Optional, Dict, Any



class JobStatus(BaseModel):
    id: str
    status: str
    progress: int
    metrics: Optional[Dict[str, Any]] = None


class OptimizeRequest(BaseModel):
    algo: Optional[str] = "PPO"
    useGNN: Optional[bool] = True
    timesteps: Optional[int] = 10000
    hpwlBefore: Optional[float] = 1000.0


    