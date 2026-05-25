from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from bson import ObjectId



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


class Design(BaseModel):
    """Pydantic model for designs stored in MongoDB.

    Fields:
      - id: optional string id (mapped to MongoDB _id)
      - name: human-friendly name
      - owner: optional owner identifier
      - created_at: timestamp of creation
      - metadata: arbitrary metadata
      - data: the actual design payload (layout/graph/placements)
    """
    id: Optional[str] = None
    name: str
    owner: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    data: Dict[str, Any]

    class Config:
        json_encoders = {
            ObjectId: lambda oid: str(oid)
        }


    