import os
from dotenv import load_dotenv

load_dotenv()

# MinIO Configuration (for loading the main fusion model)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "False").lower() == "true"
BUCKET_MODELS_STORE = os.getenv("BUCKET_MODELS_STORE", "models-store")
# Path to the trained fusion model within the bucket.
# For PoC, assume a fixed name. In prod, this could be versioned or point to 'latest'.
FUSION_MODEL_OBJECT_NAME = os.getenv("FUSION_MODEL_OBJECT_NAME", "trained_models/disease_fusion_model/disease_fusion_model_epoch_X.pt") # Replace X with actual epoch or use 'latest' if managed

# Image Processing Configuration (MUST match image_preprocessing_service and training)
IMG_SIZE = int(os.getenv("IMG_SIZE_PRED", "224"))
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
IMAGE_PRETRAINED_MODEL_NAME = os.getenv("IMAGE_PRETRAINED_MODEL_NAME_PRED", "resnet50") # For image feature extractor

# Tabular (NIH-like) Data Processing Config (MUST match tabular_preprocessing_service and training)
NIH_CATEGORICAL_COLS_PRED = ['Patient Gender', 'View Position']
NIH_NUMERICAL_COLS_PRED = ['Patient Age_cleaned'] # If age is the only numerical one
# Categories for OneHotEncoder - THIS IS CRITICAL and must match training
# Ideally, load a fitted encoder. For PoC, define categories.
EXAMPLE_CATEGORIES_PRED = {
    'Patient Gender': ['M', 'F', 'O'], # Must match EXACTLY what was used in tabular_preprocessing_service
    'View Position': ['PA', 'AP', 'LL', 'RL', 'XX'] # Add 'XX' or any other expected values. Add 'Missing' if you imputed with it.
}


# Sensor Data Processing Config (MUST match tabular_preprocessing_service and training)
SENSOR_COLUMNS_TO_AGGREGATE_PRED = [
    'HeartRate_bpm', 'RespiratoryRate_bpm', 'SpO2_percent',
    'Temperature_C', 'BPSystolic_mmHg', 'BPDiastolic_mmHg'
]
SENSOR_AGGREGATIONS_PRED = ['mean', 'std', 'min', 'max', 'median']

# Feature Dimensions (MUST match the trained fusion model's expected input)
# These are from model_training_service/scripts/config_training.py
IMAGE_FEATURE_DIM_PRED = int(os.getenv("IMAGE_FEATURE_DIM_PRED", "2048"))
NIH_TABULAR_FEATURE_DIM_PRED = int(os.getenv("NIH_TABULAR_FEATURE_DIM_PRED", "50")) # This needs to be precise after one-hot encoding
SENSOR_FEATURE_DIM_PRED = int(os.getenv("SENSOR_FEATURE_DIM_PRED", "30")) # num_sensor_cols * num_aggregations

# Model & Device
DEVICE_PRED = os.getenv("DEVICE_PRED", "cpu") # "cuda" or "cpu"
NUM_CLASSES_PRED = 15
ALL_DISEASE_CLASSES_PRED = [ # Must match order from training utils
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "No Finding"
]