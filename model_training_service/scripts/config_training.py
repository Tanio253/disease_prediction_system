# model_training_service/scripts/config_training.py
import torch
import os

# --- Environment/Service Configurations ---
# These should ideally be set via environment variables in docker-compose.yml for flexibility
PATIENT_DATA_SERVICE_URL = os.getenv("PATIENT_DATA_SERVICE_URL", "http://patient_data_service:8000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "False").lower() == "true"

# --- MinIO Bucket Names ---
BUCKET_PROCESSED_IMAGE_FEATURES = "processed-image-features"
BUCKET_PROCESSED_NIH_TABULAR_FEATURES = "processed-nih-tabular-features"
BUCKET_PROCESSED_SENSOR_FEATURES = "processed-sensor-features"
BUCKET_MODELS_STORE = "models-store"

# --- Feature Dimensions (CRITICAL - Must match output of preprocessing services) ---
# These are placeholders. You MUST determine these from your actual feature files.
# Example: If ResNet50 outputs 2048 features.
IMAGE_FEATURE_DIM = int(os.getenv("IMAGE_FEATURE_DIM_TRAIN", "2048"))

# Example: If NIH (age=1 + gender_OHE=3 + view_pos_OHE=5) = 9 features.
# Calculate this precisely based on your OneHotEncoder output in tabular_preprocessing_service.
NIH_TABULAR_FEATURE_DIM = int(os.getenv("NIH_TABULAR_FEATURE_DIM_TRAIN", "9")) # Placeholder

# Example: If 6 sensor types * 5 aggregation methods = 30 features.
SENSOR_FEATURE_DIM = int(os.getenv("SENSOR_FEATURE_DIM_TRAIN", "30")) # Placeholder

# --- Fusion Model Architecture Parameters ---
# Hidden layer dimensions for the fusion MLP
HIDDEN_DIMS_MLP = [1024, 512] # Example: two hidden layers
# NUM_CLASSES is defined in utils.py based on ALL_DISEASE_CLASSES
DROPOUT_RATE = 0.3

# --- Training Hyperparameters ---
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001")) # 1e-4
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "50")) # Start with a reasonable number
DEVICE = os.getenv("DEVICE_TRAIN", "cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42 # For reproducibility in splitting data, etc.
VALIDATION_SPLIT_RATIO = 0.2 # Proportion of data to use for validation

# --- Model Saving ---
MODEL_NAME_FUSION = "disease_fusion_model" # Base name for the saved model files

# --- Logging ---
LOG_LEVEL = "INFO"

# --- Data Loader ---
NUM_WORKERS_DATALOADER = int(os.getenv("NUM_WORKERS_DATALOADER", "0")) # 0 for main process, >0 for multiprocessing

# --- Column names (if needed, though utils.py has disease classes) ---
# If your NIH metadata processing relied on specific column names for categorical/numerical features,
# and if those were not directly passed but inferred from config, they could be here.
# However, it's better if the preprocessing services are self-contained or rely on a shared schema.
# For now, utils.py handles the main class definitions.

# Ensure device is set correctly
if DEVICE == "cuda" and not torch.cuda.is_available():
    print(f"Warning: DEVICE set to 'cuda' but CUDA is not available. Falling back to 'cpu'.")
    DEVICE = "cpu"

print(f"--- Training Configuration ---")
print(f"Device: {DEVICE}")
print(f"Image Feature Dimension: {IMAGE_FEATURE_DIM}")
print(f"NIH Tabular Feature Dimension: {NIH_TABULAR_FEATURE_DIM}") # CRITICAL
print(f"Sensor Feature Dimension: {SENSOR_FEATURE_DIM}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Num Epochs: {NUM_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"-----------------------------")