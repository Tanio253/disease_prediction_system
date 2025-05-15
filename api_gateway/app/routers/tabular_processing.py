from fastapi import APIRouter, Request
from .common import forward_request
# Assuming TABULAR_PREPROCESSING_SERVICE_URL_GW is defined in api_gateway/app/config.py
# pointing to http://tabular_preprocessing_service:8003
from ..config import TABULAR_PREPROCESSING_SERVICE_URL_GW as SERVICE_TARGET_URL

router = APIRouter()

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route_to_tabular_processing_service(request: Request, path: str):
    backend_path = f"/{path}"
    return await forward_request(request, SERVICE_TARGET_URL, backend_path)