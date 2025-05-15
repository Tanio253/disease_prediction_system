import os
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "False").lower() == "true"

PATIENT_DATA_SERVICE_URL = os.getenv("PATIENT_DATA_SERVICE_URL", "http://patient_data_service:8000")

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api_gateway:8080") # Base URL of the gateway

IMAGE_PREPROCESSING_ENDPOINT = f"{API_GATEWAY_URL}/api/v1/image-preprocess/preprocess/"
TABULAR_NIH_PREPROCESSING_ENDPOINT = f"{API_GATEWAY_URL}/api/v1/tabular-preprocess/preprocess/nih-metadata/"
TABULAR_SENSOR_PREPROCESSING_ENDPOINT = f"{API_GATEWAY_URL}/api/v1/tabular-preprocess/preprocess/sensor-data/"

# MinIO Bucket Names (centralize if used in multiple places, but good here for clarity)
BUCKET_RAW_IMAGES = "raw-images"
BUCKET_RAW_SENSOR_DATA_PER_STUDY = "raw-sensor-data-per-study"
# We won't upload the full metadata/sensor CSVs again if we process them directly from UploadFile.
# If we were to save them first, these would be the buckets:
# BUCKET_RAW_TABULAR_UPLOADS = "raw-tabular-uploads"