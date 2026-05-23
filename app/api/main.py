from fastapi import APIRouter


from app.api.routes import model_requests, designs


api_router = APIRouter()

api_router.include_router(model_requests.router)
api_router.include_router(designs.router)


