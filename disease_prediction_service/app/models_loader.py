import torch
import torchvision.models as models_tv
from minio import Minio
import io
import logging
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd # For dummy DataFrame for encoder fitting

# Assuming your FusionMLP definition is accessible
# If model_def.py from model_training_service is not in a shared package,
# you might need to copy/redefine FusionMLP here or make it part of a shared library.
# For this example, let's assume we redefine a compatible FusionMLP structure or import it.
#
# Placeholder: You'd need to ensure the FusionMLP class definition is available.
# One way is to copy model_training_service/scripts/model_def.py to this service's app directory
# and adjust imports if necessary. Let's assume model_def.py is copied to app/
try:
    from .model_def import FusionMLP # Assumes model_def.py is in the same directory
except ImportError:
    # Fallback or error if model_def.py is not found
    logger = logging.getLogger(__name__)
    logger.error("model_def.py (FusionMLP definition) not found. Predictions will fail.")
    # Define a dummy class to prevent import errors, but this won't work for actual prediction
    class FusionMLP(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc=torch.nn.Linear(1,1)
        def forward(self, *args): return self.fc(torch.randn(1,1))


from .config import (
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_USE_SSL,
    BUCKET_MODELS_STORE, FUSION_MODEL_OBJECT_NAME, DEVICE_PRED,
    IMAGE_PRETRAINED_MODEL_NAME, EXAMPLE_CATEGORIES_PRED, NIH_CATEGORICAL_COLS_PRED,
    IMAGE_FEATURE_DIM_PRED, NIH_TABULAR_FEATURE_DIM_PRED, SENSOR_FEATURE_DIM_PRED, NUM_CLASSES_PRED
)

logger = logging.getLogger(__name__)

# Global variables for loaded models and transformers
image_feature_extractor_model_g = None
fusion_model_g = None
nih_data_encoder_g = None # For OneHotEncoder
device_g = None
HIDDEN_DIMS_MLP_FOR_PRED = [1024, 512] # Must match training
DROPOUT_RATE_FOR_PRED = 0.3 # Must match training

def load_image_feature_extractor():
    global image_feature_extractor_model_g, device_g
    if image_feature_extractor_model_g is None:
        logger.info(f"Loading image feature extractor: {IMAGE_PRETRAINED_MODEL_NAME} for prediction.")
        try:
            if IMAGE_PRETRAINED_MODEL_NAME == "resnet50":
                weights = models_tv.ResNet50_Weights.IMAGENET1K_V2
                model = models_tv.resnet50(weights=weights)
                model.fc = torch.nn.Identity() # Remove classifier
            elif IMAGE_PRETRAINED_MODEL_NAME == "efficientnet_b0":
                weights = models_tv.EfficientNet_B0_Weights.IMAGENET1K_V1
                model = models_tv.efficientnet_b0(weights=weights)
                model.classifier = torch.nn.Identity() # Remove classifier
            else:
                raise ValueError(f"Unsupported image feature extractor model: {IMAGE_PRETRAINED_MODEL_NAME}")

            device_g = torch.device(DEVICE_PRED if torch.cuda.is_available() and DEVICE_PRED == "cuda" else "cpu")
            image_feature_extractor_model_g = model.to(device_g)
            image_feature_extractor_model_g.eval()
            logger.info(f"Image feature extractor '{IMAGE_PRETRAINED_MODEL_NAME}' loaded on {device_g}.")
        except Exception as e:
            logger.error(f"Error loading image feature extractor model: {e}", exc_info=True)
            raise # Critical for service function

def load_fusion_model():
    global fusion_model_g, device_g # device_g should be set by image extractor loading or here
    if fusion_model_g is None:
        logger.info(f"Loading fusion model from MinIO: {BUCKET_MODELS_STORE}/{FUSION_MODEL_OBJECT_NAME}")
        if device_g is None: # Ensure device is determined
            device_g = torch.device(DEVICE_PRED if torch.cuda.is_available() and DEVICE_PRED == "cuda" else "cpu")

        minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_USE_SSL)
        try:
            model_data = minio_client.get_object(BUCKET_MODELS_STORE, FUSION_MODEL_OBJECT_NAME)
            model_bytes = model_data.read()
            
            # Instantiate model architecture (ensure class definition matches saved model)
            # The FusionMLP __init__ needs to be compatible with the expected feature dimensions
            # This relies on IMAGE_FEATURE_DIM_PRED, NIH_TABULAR_FEATURE_DIM_PRED, SENSOR_FEATURE_DIM_PRED
            # being correctly set in config.py to match the model that was trained and saved.
            
            # Re-configure FusionMLP's internal dimensions based on _PRED config values.
            # This assumes your model_def.FusionMLP can be reconfigured or that the
            # config values used here match its saved state perfectly.
            # One way to handle this is for model_def.py to also read these from config.
            
            # For FusionMLP as defined in training (it reads dimensions from its own config import):
            # To make this robust, the FusionMLP class in model_def.py should accept these dims as __init__ args
            # or the config.py it imports should match the _PRED versions here.
            # Let's assume the imported FusionMLP class uses the same config mechanism or fixed dims.
            # We might need to override the config that FusionMLP (from model_def) sees:
            # This is tricky. Best if FusionMLP takes dims as args:
            # model_architecture = FusionMLP(img_dim=IMAGE_FEATURE_DIM_PRED, ...)

            # Assuming FusionMLP uses the config values from *its own imported config* which should match:
            model_architecture = FusionMLP(
    image_feature_dim=IMAGE_FEATURE_DIM_PRED,
    nih_tabular_feature_dim=NIH_TABULAR_FEATURE_DIM_PRED,
    sensor_feature_dim=SENSOR_FEATURE_DIM_PRED,
    hidden_dims_mlp=HIDDEN_DIMS_MLP_FOR_PRED, # From config or hardcoded to match training
    num_classes=NUM_CLASSES_PRED,
    dropout_rate=DROPOUT_RATE_FOR_PRED # From config or hardcoded
)

            # A better FusionMLP definition in model_def.py would be:
            # class FusionMLP(nn.Module):
            #     def __init__(self, img_dim, nih_dim, sensor_dim, hidden_dims, num_classes, dropout):
            #         ... self.total_input_dim = img_dim + nih_dim + sensor_dim ...
            # Then here:
            # from .config import HIDDEN_DIMS_MLP, DROPOUT_RATE # Assuming these are in current config
            # model_architecture = FusionMLP(
            #     img_dim=IMAGE_FEATURE_DIM_PRED,
            #     nih_dim=NIH_TABULAR_FEATURE_DIM_PRED, # This one especially needs to be accurate!
            #     sensor_dim=SENSOR_FEATURE_DIM_PRED,
            #     hidden_dims=HIDDEN_DIMS_MLP, # Need to get this from config too
            #     num_classes=NUM_CLASSES_PRED,
            #     dropout=DROPOUT_RATE # from config
            # )


            buffer = io.BytesIO(model_bytes)
            # Load state_dict. Ensure map_location for device flexibility.
            model_architecture.load_state_dict(torch.load(buffer, map_location=device_g))
            
            fusion_model_g = model_architecture.to(device_g)
            fusion_model_g.eval()
            logger.info(f"Fusion model loaded successfully from MinIO onto {device_g}.")

        except Exception as e:
            logger.error(f"Error loading fusion model: {e}", exc_info=True)
            raise # Critical for service function
        finally:
            if 'model_data' in locals():
                model_data.close()
                model_data.release_conn()

def load_nih_data_encoder():
    global nih_data_encoder_g
    if nih_data_encoder_g is None:
        logger.info("Initializing OneHotEncoder for NIH data prediction.")
        # THIS IS CRITICAL: The encoder must be identical to the one used in training/preprocessing.
        # Option 1 (Ideal): Load a saved, fitted encoder object (e.g., from joblib)
        # Option 2 (PoC): Re-initialize with the exact same categories.
        try:
            nih_data_encoder_g = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            # Create a dummy DataFrame with all possible categories to "fit" the encoder
            # This MUST match the categories and column order used in tabular_preprocessing_service
            dummy_data_for_encoder = {}
            valid_cols_for_fitting = []
            for col in NIH_CATEGORICAL_COLS_PRED: # These are the columns encoder expects
                if col in EXAMPLE_CATEGORIES_PRED:
                    dummy_data_for_encoder[col] = EXAMPLE_CATEGORIES_PRED[col]
                    valid_cols_for_fitting.append(col)
                else: # Should not happen if config is correct
                    logger.warning(f"No example categories defined in config for '{col}'. Encoder might be misconfigured.")
                    dummy_data_for_encoder[col] = ['UnknownCategoryEncountered'] # Placeholder
                    valid_cols_for_fitting.append(col)


            if not valid_cols_for_fitting:
                logger.error("No valid columns to fit OneHotEncoder based on NIH_CATEGORICAL_COLS_PRED. Check config.")
                # This is a critical failure.
                raise ValueError("Cannot fit OneHotEncoder due to missing column configurations.")

            dummy_df_encoder = pd.DataFrame(dummy_data_for_encoder)
            # Ensure the columns being fitted are only those in NIH_CATEGORICAL_COLS_PRED
            # and in the correct order if the original training encoder relied on it.
            # Here, we just use the list of columns for which we have categories.
            nih_data_encoder_g.fit(dummy_df_encoder[valid_cols_for_fitting]) # Fit only on relevant columns
            
            # Validate output dimension:
            # The number of output features from this encoder should match a portion of NIH_TABULAR_FEATURE_DIM_PRED
            # (the other part being numerical features like age).
            # This is complex to auto-validate here without knowing the numerical part.
            # The user MUST ensure NIH_TABULAR_FEATURE_DIM_PRED is correctly set in config.
            logger.info(f"OneHotEncoder for NIH data initialized and 'fitted' with example categories for columns: {valid_cols_for_fitting}.")
            logger.info(f"Encoder output features: {nih_data_encoder_g.get_feature_names_out(valid_cols_for_fitting)}")

        except Exception as e:
            logger.error(f"Error initializing/fitting OneHotEncoder for NIH data: {e}", exc_info=True)
            raise # Critical

def get_models_and_transformers():
    # This function ensures all necessary components are loaded.
    # It will be called by Depends in the API endpoint.
    if image_feature_extractor_model_g is None:
        load_image_feature_extractor()
    if fusion_model_g is None:
        load_fusion_model()
    if nih_data_encoder_g is None:
        load_nih_data_encoder()
    return image_feature_extractor_model_g, fusion_model_g, nih_data_encoder_g, device_g

# Call loading functions once at import time (if service structure allows for it, e.g. Uvicorn workers)
# Or, more robustly, use FastAPI's startup event.
# For simplicity in this file, we'll rely on get_models_and_transformers() being called.