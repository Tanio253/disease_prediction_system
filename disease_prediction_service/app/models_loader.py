import torch
import torchvision.models as models_tv
from minio import Minio
import io
import logging
import os
from sklearn.preprocessing import OneHotEncoder # Keep if you plan to load/save it
import pandas as pd
import json # Added


from .model_def import AttentionFusionMLP
from .config import settings # Your existing config for service settings

logger = logging.getLogger(__name__)

image_feature_extractor_model_g = None
fusion_model_g = None # This will now be an AttentionFusionMLP instance

nih_data_encoder_g = None
device_g = None


loaded_components = {
    "image_feature_extractor": None,
    "fusion_model": None,
    "nih_data_encoder": None, # Or more generally 'preprocessing_artifacts'
    "mlb": None, # MultiLabelBinarizer, should also be loaded from training
    "training_config": None, # To store loaded training config
    "device": None
}


def load_image_feature_extractor():
    # global image_feature_extractor_model_g, device_g # Use loaded_components
    if loaded_components["image_feature_extractor"] is None:
        logger.info(f"Loading image feature extractor: {settings.IMAGE_PRETRAINED_MODEL_NAME} for prediction.")
        try:
            if settings.IMAGE_PRETRAINED_MODEL_NAME == "resnet50":
                weights = models_tv.ResNet50_Weights.IMAGENET1K_V2
                model = models_tv.resnet50(weights=weights)
                model.fc = torch.nn.Identity()
            elif settings.IMAGE_PRETRAINED_MODEL_NAME == "efficientnet_b0":
                weights = models_tv.EfficientNet_B0_Weights.IMAGENET1K_V1
                model = models_tv.efficientnet_b0(weights=weights)
                model.classifier = torch.nn.Identity()
            else:
                raise ValueError(f"Unsupported image feature extractor model: {settings.IMAGE_PRETRAINED_MODEL_NAME}")

            current_device = torch.device(settings.DEVICE_PRED if torch.cuda.is_available() and settings.DEVICE_PRED == "cuda" else "cpu")
            loaded_components["image_feature_extractor"] = model.to(current_device)
            loaded_components["image_feature_extractor"].eval()
            loaded_components["device"] = current_device # Set device globally once
            logger.info(f"Image feature extractor '{settings.IMAGE_PRETRAINED_MODEL_NAME}' loaded on {current_device}.")
        except Exception as e:
            logger.error(f"Error loading image feature extractor model: {e}", exc_info=True)
            raise

def load_fusion_model_and_artifacts():
    if loaded_components["fusion_model"] is None:
        model_name = settings.FUSION_MODEL_NAME # Get model name from settings
        logger.info(f"Loading fusion model artifacts for '{model_name}' from MinIO bucket: {settings.MINIO_BUCKET_MODELS}")

        if loaded_components["device"] is None: # Ensure device is determined
            loaded_components["device"] = torch.device(settings.DEVICE_PRED if torch.cuda.is_available() and settings.DEVICE_PRED == "cuda" else "cpu")
        
        current_device = loaded_components["device"]

        minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_USE_SSL
        )

        training_config_path_minio = f"models/{model_name}/training_config.json"
        local_training_config_path = f"/tmp/{model_name}_training_config.json" # Ensure unique temp name
        try:
            logger.info(f"Attempting to load training config from: {training_config_path_minio}")
            minio_client.fget_object(settings.MINIO_BUCKET_MODELS, training_config_path_minio, local_training_config_path)
            with open(local_training_config_path, 'r') as f:
                training_config = json.load(f)
            loaded_components["training_config"] = training_config
            logger.info(f"Training config for '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load training_config.json for model '{model_name}' from MinIO: {e}", exc_info=True)
            raise RuntimeError(f"Could not load training_config.json for model '{model_name}': {e}")
        finally:
            if os.path.exists(local_training_config_path):
                os.remove(local_training_config_path)
        
        try:
            model_architecture = AttentionFusionMLP(
                img_feature_dim=training_config['img_feature_dim'],
                nih_feature_dim=training_config['nih_feature_dim'],
                sensor_feature_dim=training_config['sensor_feature_dim'],
                num_classes=training_config['num_classes'],
                embed_dim=training_config['embed_dim'],
                num_heads=training_config['num_heads'],
                dropout_rate=training_config.get('dropout_rate', 0.3) # Default if not in older config
            )
        except KeyError as e:
            logger.error(f"CRITICAL: Missing key {e} in loaded training_config.json. Cannot initialize model.")
            raise RuntimeError(f"Missing key {e} in training_config.json for model '{model_name}'.")

        
        model_state_dict_path_minio = f"models/{model_name}/{settings.FUSION_MODEL_STATE_DICT_FILENAME}" # e.g., "model.pt" or "best_model.pt"
        local_model_state_dict_path = f"/tmp/{model_name}_model_state.pt"
        try:
            logger.info(f"Attempting to load model state_dict from: {model_state_dict_path_minio}")
            minio_client.fget_object(settings.MINIO_BUCKET_MODELS, model_state_dict_path_minio, local_model_state_dict_path)
            model_architecture.load_state_dict(torch.load(local_model_state_dict_path, map_location=current_device))
            
            loaded_components["fusion_model"] = model_architecture.to(current_device)
            loaded_components["fusion_model"].eval()
            logger.info(f"Fusion model '{model_name}' state_dict loaded successfully onto {current_device}.")
        except Exception as e:
            logger.error(f"CRITICAL: Error loading fusion model state_dict for '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Could not load model state_dict for '{model_name}': {e}")
        finally:
            if os.path.exists(local_model_state_dict_path):
                os.remove(local_model_state_dict_path)
          

        mlb_path_minio = f"models/{model_name}/mlb.joblib" # Or .pkl, adjust as per saving
        local_mlb_path = f"/tmp/{model_name}_mlb.joblib"
        try:
            logger.info(f"Attempting to load MLB from: {mlb_path_minio}")
            minio_client.fget_object(settings.MINIO_BUCKET_MODELS, mlb_path_minio, local_mlb_path)
         
            import pickle # Example using pickle
            with open(local_mlb_path, 'rb') as f:
                loaded_components["mlb"] = pickle.load(f)
            logger.info(f"MLB for '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Warning: Could not load MLB for model '{model_name}' from MinIO: {e}. Check if it was saved correctly.", exc_info=True)
            
        if loaded_components.get("nih_data_encoder") is None and settings.LOAD_NIH_ENCODER_FROM_FILE is False: # Control via config
            load_nih_data_encoder_refit_logic() # Calls the re-fitting logic

def load_nih_data_encoder_refit_logic():
    """
    Re-fits OneHotEncoder based on config.
    WARNING: Prone to discrepancies if config doesn't perfectly match training.
    It's highly recommended to save and load the fitted encoder from training.
    """
    if loaded_components["nih_data_encoder"] is None:
        logger.warning("Attempting to re-initialize OneHotEncoder for NIH data. "
                       "It's STRONGLY recommended to load a saved, fitted encoder from training.")
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            dummy_data_for_encoder = {}
            valid_cols_for_fitting = []

            for col in settings.NIH_CATEGORICAL_COLS_PRED:
                if col in settings.EXAMPLE_CATEGORIES_PRED and settings.EXAMPLE_CATEGORIES_PRED[col]:
                    dummy_data_for_encoder[col] = settings.EXAMPLE_CATEGORIES_PRED[col]
                    valid_cols_for_fitting.append(col)
                else:
                    logger.warning(f"No example categories defined in config for '{col}' for OHE fitting. Using a placeholder.")
                    # Using a placeholder to allow fit, but this might not match training.
                    dummy_data_for_encoder[col] = [f"Placeholder_{col}"]
                    valid_cols_for_fitting.append(col)
            
            if not valid_cols_for_fitting:
                logger.error("No valid columns found to fit OneHotEncoder based on NIH_CATEGORICAL_COLS_PRED. Check config.")
                # This might be okay if there are no categorical columns expected.
                loaded_components["nih_data_encoder"] = None # Explicitly set to None if no fitting occurs
                return

            dummy_df_encoder = pd.DataFrame(dummy_data_for_encoder)
            ohe.fit(dummy_df_encoder[valid_cols_for_fitting])
            loaded_components["nih_data_encoder"] = ohe
            logger.info(f"OneHotEncoder for NIH data re-initialized and 'fitted' for columns: {valid_cols_for_fitting}.")
            logger.info(f"Encoder output features: {ohe.get_feature_names_out(valid_cols_for_fitting)}")

        except Exception as e:
            logger.error(f"Error re-initializing/fitting OneHotEncoder for NIH data: {e}", exc_info=True)
            loaded_components["nih_data_encoder"] = None


def get_model_components():
    """
    Ensures all necessary components are loaded and returns them.
    This function will be called, e.g., via FastAPI's Depends.
    """
    if loaded_components["device"] is None: # Try to determine device early
        loaded_components["device"] = torch.device(settings.DEVICE_PRED if torch.cuda.is_available() and settings.DEVICE_PRED == "cuda" else "cpu")

    if loaded_components["image_feature_extractor"] is None and settings.LOAD_IMAGE_EXTRACTOR: # Control via config
        load_image_feature_extractor()
    
    if loaded_components["fusion_model"] is None:
        load_fusion_model_and_artifacts() # This now loads model, config, mlb, etc.


    if loaded_components["fusion_model"] is None:
        raise RuntimeError("Fusion model could not be loaded. Prediction service cannot operate.")
    if loaded_components["mlb"] is None and settings.REQUIRE_MLB: # Control if MLB is strictly required
         logger.warning("MLB (MultiLabelBinarizer) is not loaded. Decoding predictions might fail or be inaccurate.")
         # raise RuntimeError("MLB is required but could not be loaded.")


    return loaded_components

from fastapi import FastAPI
app = FastAPI() # Assuming 'app' is your FastAPI instance, defined in main.py usually
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Pre-loading models and components...")
    try:
        get_model_components() # This will trigger all loading functions
        logger.info("Models and components pre-loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to preload models during startup: {e}", exc_info=True)
        # Depending on policy, you might want to prevent startup or allow it with degraded functionality.