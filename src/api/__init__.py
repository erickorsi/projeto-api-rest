from fastapi import APIRouter

from src.api.endpoints.predict import predict_router

router = APIRouter()

# POST routes
router.include_router(
    predict_router,
    prefix="/predict",
    tags="Machine Learning"
)
