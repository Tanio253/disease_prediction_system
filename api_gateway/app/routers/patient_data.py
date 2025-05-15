from fastapi import APIRouter, Request
from .common import forward_request
from ..config import PATIENT_DATA_SERVICE_URL, API_GATEWAY_BASE_PATH

router = APIRouter()
SERVICE_TARGET_URL = PATIENT_DATA_SERVICE_URL
# For patient_data_service, it has distinct top-level paths like /patients/ and /studies/
# So the prefix to strip might be just the base API gateway prefix if we route them separately.
# Or, we can have /api/v1/patientdata/patients and /api/v1/patientdata/studies

# Example: if we want to route /api/v1/data/patients/* and /api/v1/data/studies/*
# to patient_data_service's /patients/* and /studies/* respectively.

@router.api_route("/patients/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def route_to_patient_data_patients(request: Request, path: str):
    backend_path = f"/patients/{path}" # Route to /patients/* in backend
    return await forward_request(request, SERVICE_TARGET_URL, backend_path)

@router.api_route("/studies/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def route_to_patient_data_studies(request: Request, path: str):
    backend_path = f"/studies/{path}" # Route to /studies/* in backend
    return await forward_request(request, SERVICE_TARGET_URL, backend_path)

# A simpler catch-all if patient_data_service uses paths directly without /patients or /studies prefix
# @router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
# async def route_to_patient_data_service(request: Request, path: str):
#     backend_path = f"/{path}"
#     return await forward_request(request, SERVICE_TARGET_URL, backend_path)