import os
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "False").lower() == "true"

PATIENT_DATA_SERVICE_URL = os.getenv("PATIENT_DATA_SERVICE_URL", "http://patient_data_service:8000")

BUCKET_RAW_IMAGES = os.getenv("BUCKET_RAW_IMAGES", "raw-images")
BUCKET_PROCESSED_IMAGE_FEATURES = os.getenv("BUCKET_PROCESSED_IMAGE_FEATURES", "processed-image-features")

# Image Processing Config
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
NORMALIZATION_MEAN = [0.485, 0.456, 0.406] # ImageNet defaults
NORMALIZATION_STD = [0.229, 0.224, 0.225] # ImageNet defaults
DEVICE = os.getenv("DEVICE", "cpu") # "cuda" if GPU is available and configured

PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME", "resnet50") # e.g., resnet50, efficientnet_b0