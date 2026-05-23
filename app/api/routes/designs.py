from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.models import Design
from app import db as app_db

router = APIRouter(prefix="/designs", tags=["designs"])


@router.post("/", status_code=201)
async def upload_design(design: Design):
    saved = await app_db.create_design(design)
    return {"id": saved["id"]}


@router.get("/", response_model=List[dict])
async def list_all_designs():
    return await app_db.list_designs()


@router.get("/{design_id}")
async def get_design(design_id: str):
    doc = await app_db.get_design(design_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Design not found")
    return doc
