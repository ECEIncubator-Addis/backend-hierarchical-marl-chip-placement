
from typing import Optional, Any, Dict, List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from bson import ObjectId
from datetime import datetime, timezone
from app.models import Design

client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorDatabase] = None


async def init_db(uri: str = "mongodb://localhost:27017", db_name: str = "chip_placement") -> None:
    """Initialize global Motor client and database."""
    global client, db
    if client is None:
        client = AsyncIOMotorClient(uri)
        db = client[db_name]


def get_db() -> AsyncIOMotorDatabase:
    if db is None:
        raise RuntimeError("Database not initialized. Call init_db(...) before using.")
    return db


async def close_db() -> None:
    global client
    if client is not None:
        client.close()


async def create_design(design: Design) -> Dict[str, Any] | None:
    d = design.model_dump(exclude_none=True)
    # set created_at if missing
    if "created_at" not in d or d.get("created_at") is None:
        d["created_at"] = datetime.now(timezone.utc)
    # if caller provided an id, store as _id
    if d.get("id"):
        d["_id"] = d.pop("id")
    res = await get_db().designs.insert_one(d)
    # return saved document with string id
    try:
        saved = await get_db().designs.find_one({"_id": res.inserted_id})
        saved["id"] = str(saved.get("_id")) #type: ignore
        saved.pop("_id", None) #type: ignore
        return saved #type: ignore
    except Exception:
        return {"Exception": "Failed to retrieve saved design"}


async def list_designs() -> List[Dict[str, Any]]:
    out = []
    cursor = get_db().designs.find({})
    async for doc in cursor:
        doc["id"] = str(doc.get("_id"))
        doc.pop("_id", None)
        out.append(doc)
    return out


async def get_design(design_id: str) -> Optional[Dict[str, Any]]:
    # try ObjectId conversion, fallback to string match
    query = None
    try:
        query = {"_id": ObjectId(design_id)}
    except Exception:
        query = {"_id": design_id}
    doc = await get_db().designs.find_one(query)
    if not doc:
        return None
    doc["id"] = str(doc.get("_id"))
    doc.pop("_id", None)
    return doc
