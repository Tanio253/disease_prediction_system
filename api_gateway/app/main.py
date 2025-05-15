from fastapi import FastAPI, Request, Response
import logging

from .config import API_GATEWAY_BASE_PATH
from .routers import predict, ingest, patient_data # Import your routers
from .routers import predict, ingest, patient_data, image_processing, tabular_processing 
# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disease Prediction API Gateway",
    description="Single entry point for all disease prediction backend services.",
    version="0.1.0"
)
app.include_router(image_processing.router, prefix=f"{API_GATEWAY_BASE_PATH}/image-preprocess", tags=["Image Preprocessing Service"])
app.include_router(tabular_processing.router, prefix=f"{API_GATEWAY_BASE_PATH}/tabular-preprocess", tags=["Tabular Preprocessing Service"])
@app.on_event("startup")
async def startup_event():
    logger.info("API Gateway starting up...")
    logger.info(f"All routes will be prefixed under: {API_GATEWAY_BASE_PATH} (or as defined by routers)")
    # You can log the loaded backend service URLs from config here for verification

# Include routers with their specific prefixes
app.include_router(predict.router, prefix=f"{API_GATEWAY_BASE_PATH}/predict", tags=["Prediction Service"])
app.include_router(ingest.router, prefix=f"{API_GATEWAY_BASE_PATH}/ingest", tags=["Data Ingestion Service"])
# For patient_data, requests to /api/v1/data/patients/* and /api/v1/data/studies/*
# will be handled by patient_data.router
app.include_router(patient_data.router, prefix=f"{API_GATEWAY_BASE_PATH}/data", tags=["Patient Data Service"])
# Add other routers here (e.g., for model_training_service if it has an API)
# app.include_router(training.router, prefix=f"{API_GATEWAY_BASE_PATH}/training", tags=["Model Training Service"])


@app.get("/gateway_health", tags=["Gateway Health"])
async def health_check():
    return {"status": "healthy", "service": "API Gateway"}

# Optional: A root path message for the gateway itself
@app.get("/", tags=["Gateway Root"])
async def gateway_root():
    return {"message": "Welcome to the Disease Prediction API Gateway. Access services via /api/v1/* paths."}

# Example of a middleware for basic request logging (optional)
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    logger.info(f"Gateway Incoming: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Gateway Outgoing status: {response.status_code} for {request.url.path}")
    return response