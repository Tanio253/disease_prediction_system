from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from typing import List, Optional, Dict
import torch
import logging
import numpy as np # For sigmoid if not using torch.sigmoid

from .config import ALL_DISEASE_CLASSES_PRED, FUSION_MODEL_OBJECT_NAME # And others for logging
from .models_loader import get_models_and_transformers, FusionMLP # Assuming FusionMLP is correctly accessible
from .feature_preparation import (
    prepare_image_features_for_inference,
    prepare_nih_features_for_inference,
    prepare_sensor_features_for_inference
)
# If schemas.py is created for request/response models:
# from .schemas import PredictionResponse, DiseaseProbability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disease Prediction Service",
    description="Predicts diseases based on multi-modal patient data.",
    version="0.1.0"
)

# Dependency to ensure models are loaded
async def get_loaded_components():
    try:
        return get_models_and_transformers()
    except RuntimeError as e: # Catch critical loading errors from models_loader
        logger.error(f"Fatal error during model/transformer loading: {e}", exc_info=True)
        # This service cannot function if models don't load.
        # Uvicorn might restart it, or it will fail requests.
        # A more graceful shutdown or status reporting might be needed in production.
        raise HTTPException(status_code=503, detail=f"Service unavailable: Critical component failed to load - {str(e)}")


@app.on_event("startup")
async def startup_event():
    logger.info("Disease Prediction Service starting up...")
    try:
        # Trigger loading of all models and transformers at startup
        img_ext, fusion_m, nih_enc, dev = await get_loaded_components() # Use await if get_loaded_components becomes async
        logger.info("All models and transformers loaded successfully at startup.")
        logger.info(f"Expecting fusion model from: {FUSION_MODEL_OBJECT_NAME}")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load models/transformers during startup: {e}", exc_info=True)
        # Note: Uvicorn might still start, but /predict will fail until models load.
        # The Depends mechanism is better for per-request check if startup fails.


# Pydantic models for response (can be in schemas.py)
from pydantic import BaseModel
class DiseaseProbability(BaseModel):
    disease_name: str
    probability: float

class PredictionResponse(BaseModel):
    image_index_provided: Optional[str] = "N/A" # e.g., from image_file.filename
    predictions: List[DiseaseProbability]
    errors: Optional[List[str]] = None


@app.post("/predict/", response_model=PredictionResponse, summary="Predict diseases from multi-modal input")
async def predict_diseases(
    image_file: UploadFile = File(...),
    patient_age: int = Form(...),
    patient_gender: str = Form(...), # e.g., "M", "F", "O"
    view_position: str = Form(...), # e.g., "PA", "AP"
    sensor_data_csv: Optional[UploadFile] = File(None),
    loaded_components: tuple = Depends(get_loaded_components) # Ensure models are loaded
):
    image_feature_extractor, fusion_model, nih_encoder, device = loaded_components
    
    errors = []
    image_index_for_response = image_file.filename or "N/A"

    # 1. Prepare Image Features
    try:
        image_bytes = await image_file.read()
        img_features_tensor = prepare_image_features_for_inference(image_bytes, image_feature_extractor, device)
    except Exception as e:
        logger.error(f"Error processing image for {image_index_for_response}: {e}", exc_info=True)
        errors.append(f"Image processing failed: {str(e)}")
        # Return error or default features. For now, let's make it an error.
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


    # 2. Prepare NIH-like Tabular Features
    try:
        nih_features_tensor = prepare_nih_features_for_inference(
            patient_age, patient_gender, view_position, nih_encoder
        )
    except Exception as e:
        logger.error(f"Error processing NIH-like data for {image_index_for_response}: {e}", exc_info=True)
        errors.append(f"NIH data processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"NIH data processing failed: {str(e)}")

    # 3. Prepare Sensor Features
    sensor_bytes = None
    if sensor_data_csv:
        try:
            sensor_bytes = await sensor_data_csv.read()
        except Exception as e:
            logger.warning(f"Could not read uploaded sensor data CSV for {image_index_for_response}: {e}. Proceeding without it.")
            errors.append(f"Sensor data CSV read error (proceeding without): {str(e)}")
            sensor_bytes = None # Ensure it's None if read fails

    try:
        sensor_features_tensor = prepare_sensor_features_for_inference(sensor_bytes)
    except Exception as e: # Should be caught within prepare_sensor_features if it returns default
        logger.error(f"Unexpected error during sensor feature preparation for {image_index_for_response}: {e}", exc_info=True)
        errors.append(f"Sensor data processing failed unexpectedly: {str(e)}")
        # This path shouldn't be hit if prepare_sensor_features_for_inference handles its errors by returning zeros
        # But as a safeguard:
        from .config import SENSOR_FEATURE_DIM_PRED
        sensor_features_tensor = torch.zeros(SENSOR_FEATURE_DIM_PRED)


    # 4. Concatenate features and Perform Inference
    try:
        # Ensure all features are 1D tensors of correct float type for concatenation
        img_features_tensor = img_features_tensor.float().unsqueeze(0).to(device) # Add batch_dim, move to device
        nih_features_tensor = nih_features_tensor.float().unsqueeze(0).to(device)
        sensor_features_tensor = sensor_features_tensor.float().unsqueeze(0).to(device)
        
        # Fusion model expects features on its device
        with torch.no_grad():
            logits = fusion_model(img_features_tensor, nih_features_tensor, sensor_features_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy() # Remove batch, to CPU, to numpy

    except Exception as e:
        logger.error(f"Error during model inference for {image_index_for_response}: {e}", exc_info=True)
        errors.append(f"Model inference failed: {str(e)}")
        # Depending on policy, might return empty predictions or raise HTTP 500
        # For now, return empty predictions with error message
        return PredictionResponse(image_index_provided=image_index_for_response, predictions=[], errors=errors)


    # 5. Format and Return Response
    predictions_output: List[DiseaseProbability] = []
    for disease_name, prob in zip(ALL_DISEASE_CLASSES_PRED, probabilities):
        predictions_output.append(DiseaseProbability(disease_name=disease_name, probability=float(prob)))
        
    logger.info(f"Successfully generated predictions for {image_index_for_response}.")
    return PredictionResponse(
        image_index_provided=image_index_for_response,
        predictions=predictions_output,
        errors=errors if errors else None
    )

@app.get("/health")
async def health_check(loaded_components: tuple = Depends(get_loaded_components)): # Checks if models load
    img_ext, fusion_m, nih_enc, dev = loaded_components
    # Simple check to confirm they are not None
    if img_ext and fusion_m and nih_enc and dev:
        model_status = "All components loaded"
    else:
        model_status = "One or more components failed to load"
    return {"status": "healthy", "service": "Disease Prediction Service", "components_status": model_status}