from fastapi import APIRouter, Request
from .common import forward_request
from ..config import DATA_INGESTION_SERVICE_URL, API_GATEWAY_BASE_PATH

router = APIRouter()
SERVICE_TARGET_URL = DATA_INGESTION_SERVICE_URL
PATH_PREFIX_TO_STRIP = f"{API_GATEWAY_BASE_PATH}/ingest"

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def route_to_ingestion_service(request: Request, path: str):
    backend_path = f"/{path}"
    return await forward_request(request, SERVICE_TARGET_URL, backend_path)