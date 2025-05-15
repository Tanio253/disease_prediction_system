import os
from dotenv import load_dotenv

load_dotenv()

# These URLs will be the internal service names and ports defined in docker-compose
DISEASE_PREDICTION_SERVICE_URL = os.getenv("DISEASE_PREDICTION_SERVICE_URL_GW", "http://disease_prediction_service:8004")
DATA_INGESTION_SERVICE_URL = os.getenv("DATA_INGESTION_SERVICE_URL_GW", "http://data_ingestion_service:8001")
PATIENT_DATA_SERVICE_URL = os.getenv("PATIENT_DATA_SERVICE_URL_GW", "http://patient_data_service:8000")
# Add MODEL_TRAINING_SERVICE_URL if it has an API, e.g., http://model_training_service:PORT_FOR_TRAINING_API
IMAGE_PREPROCESSING_SERVICE_URL_GW = os.getenv("IMAGE_PREPROCESSING_SERVICE_URL_GW", "http://image_preprocessing_service:8002")
TABULAR_PREPROCESSING_SERVICE_URL_GW = os.getenv("TABULAR_PREPROCESSING_SERVICE_URL_GW", "http://tabular_preprocessing_service:8003")
# Define the base path for the API gateway itself if needed for structured logging or prefixing
API_GATEWAY_BASE_PATH = "/api/v1"