from fastapi import APIRouter


from app.api.routes import model_requests


api_router = APIRouter()

api_router.include_router(model_requests.router)


