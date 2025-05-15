# tabular_preprocessing_service/app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "False").lower() == "true"

PATIENT_DATA_SERVICE_URL = os.getenv("PATIENT_DATA_SERVICE_URL", "http://patient_data_service:8000")

# Input Buckets (read by this service)
BUCKET_RAW_NIH_METADATA = "raw-tabular" # Assuming data_ingestion might put a master file here, or this service gets structured data from patient_service
BUCKET_RAW_SENSOR_DATA_PER_STUDY = "raw-sensor-data-per-study"

# Output Buckets (written by this service)
BUCKET_PROCESSED_NIH_TABULAR_FEATURES = "processed-nih-tabular-features"
BUCKET_PROCESSED_SENSOR_FEATURES = "processed-sensor-features"

# Feature engineering parameters (example)
SENSOR_COLUMNS_TO_AGGREGATE = [
    'HeartRate_bpm', 'RespiratoryRate_bpm', 'SpO2_percent',
    'Temperature_C', 'BPSystolic_mmHg', 'BPDiastolic_mmHg'
]
SENSOR_AGGREGATIONS = ['mean', 'std', 'min', 'max', 'median']

# For NIH metadata
NIH_CATEGORICAL_COLS = ['Patient Gender', 'View Position'] # Adjust if your actual column names differ
NIH_NUMERICAL_COLS = ['Patient Age_cleaned'] # Assuming age is cleaned and named this way