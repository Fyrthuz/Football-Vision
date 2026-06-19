import torch
from fastapi import APIRouter, Depends

from app.schemas import HealthResponse
from app.services.detector import Detector

router = APIRouter(tags=["health"])


def get_detector() -> Detector:
    return Detector()


@router.get("/health", response_model=HealthResponse)
async def health(detector: Detector = Depends(get_detector)) -> HealthResponse:
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        models_loaded=detector.models_loaded,
    )
