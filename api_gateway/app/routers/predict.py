from fastapi import APIRouter, Request, Response
from .common import forward_request
from ..config import DISEASE_PREDICTION_SERVICE_URL, API_GATEWAY_BASE_PATH

router = APIRouter()
SERVICE_TARGET_URL = DISEASE_PREDICTION_SERVICE_URL
PATH_PREFIX_TO_STRIP = f"{API_GATEWAY_BASE_PATH}/predict" # e.g. /api/v1/predict

# Catch-all route for /predict/*
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def route_to_prediction_service(request: Request, path: str):
    # Construct the target path for the backend service by stripping the gateway's prefix
    # Ensure path starts with a '/' for the backend service
    backend_path = f"/{path}" 
    return await forward_request(request, SERVICE_TARGET_URL, backend_path)