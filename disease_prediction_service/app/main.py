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
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from .models_loader import get_model_components
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
import json
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


@router.post("/predict/", response_model=PredictionResponse, summary="Predict diseases from multi-modal input")
async def predict_diseases(
    # Inputs from the user
    image_file: Optional[UploadFile] = File(None), # Making image optional for flexibility
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None), # e.g., "M", "F", "O"
    view_position: Optional[str] = Form(None), # e.g., "PA", "AP"
    
    nih_data_json: Optional[str] = Form(None), # Can contain age, gender, view_position and other NIH fields
    sensor_data_csv: Optional[UploadFile] = File(None),
    loaded_components: Dict = Depends(get_model_components)
):
    fusion_model = loaded_components.get("fusion_model")
    image_feature_extractor = loaded_components.get("image_feature_extractor")
    mlb = loaded_components.get("mlb")
    device = loaded_components.get("device")
    training_config = loaded_components.get("training_config")

    if not fusion_model or not device or not training_config or not mlb:
        logger.error("Essential model components (fusion_model, device, training_config, mlb) not loaded.")
        raise HTTPException(status_code=500, detail="Model components not available. Service is not ready.")

    img_dim = fusion_model.img_feature_dim
    nih_dim = fusion_model.nih_feature_dim
    sensor_dim = fusion_model.sensor_feature_dim

    errors = []
    request_identifier = "prediction_request" # Could be a generated UUID

    img_tensor = torch.zeros(1, img_dim, device=device, dtype=torch.float32)
    img_mask = torch.tensor([True], device=device, dtype=torch.bool) # True = masked/missing

    nih_tensor = torch.zeros(1, nih_dim, device=device, dtype=torch.float32)
    nih_mask = torch.tensor([True], device=device, dtype=torch.bool)

    sensor_tensor = torch.zeros(1, sensor_dim, device=device, dtype=torch.float32)
    sensor_mask = torch.tensor([True], device=device, dtype=torch.bool)

    # 1. Prepare Image Features
    if image_file:
        try:
            image_bytes = await image_file.read()
            img_features_np = prepare_image_features_for_inference(image_bytes, image_feature_extractor, device)
            if img_features_np is not None:
                img_tensor[0] = torch.tensor(img_features_np, dtype=torch.float32)
                img_mask[0] = False # Not masked
            else:
                errors.append("Image feature extraction returned None.")
                logger.warning(f"Image feature extraction failed for {image_file.filename or request_identifier}")
        except Exception as e:
            logger.error(f"Error processing image for {image_file.filename or request_identifier}: {e}", exc_info=True)
            errors.append(f"Image processing failed: {str(e)}")

    nih_input_data = {}
    if nih_data_json:
        try:
            nih_input_data = json.loads(nih_data_json)
        except json.JSONDecodeError:
            errors.append("Invalid JSON format for NIH data.")
            logger.warning(f"Invalid JSON for NIH data in request {request_identifier}")
    # Override with individual form fields if provided and not in JSON, or if JSON is not provided
    if patient_age is not None and 'Patient Age' not in nih_input_data : nih_input_data['Patient Age'] = patient_age
    if patient_gender is not None and 'Patient Gender' not in nih_input_data: nih_input_data['Patient Gender'] = patient_gender
    if view_position is not None and 'View Position' not in nih_input_data: nih_input_data['View Position'] = view_position
    
    if nih_input_data: # Proceed if any NIH data is available
        try:

            nih_features_np = prepare_nih_features_for_inference(nih_input_data, loaded_components)
            if nih_features_np is not None:
                nih_tensor[0] = torch.tensor(nih_features_np, dtype=torch.float32)
                nih_mask[0] = False # Not masked
            else:
                errors.append("NIH feature preparation returned None.")
                logger.warning(f"NIH feature preparation failed for request {request_identifier}")
        except Exception as e:
            logger.error(f"Error processing NIH-like data for {request_identifier}: {e}", exc_info=True)
            errors.append(f"NIH data processing failed: {str(e)}")
    

    if sensor_data_csv:
        try:
            sensor_bytes = await sensor_data_csv.read()

            sensor_features_np = prepare_sensor_features_for_inference(sensor_bytes, loaded_components)
            if sensor_features_np is not None:
                sensor_tensor[0] = torch.tensor(sensor_features_np, dtype=torch.float32)
                sensor_mask[0] = False # Not masked
            else:
                errors.append("Sensor data provided but features could not be extracted.")
                logger.warning(f"Sensor feature extraction failed for {sensor_data_csv.filename or request_identifier}")
        except Exception as e:
            logger.error(f"Error processing sensor data for {request_identifier}: {e}", exc_info=True)
            errors.append(f"Sensor data processing failed: {str(e)}")
            # Continue, sensor will remain masked

    if img_mask[0].item() and nih_mask[0].item() and sensor_mask[0].item():
        detail_msg = "No valid features could be extracted from the provided inputs. " + "; ".join(errors)
        logger.error(f"Prediction failed for {request_identifier}: All modalities are masked. Errors: {errors}")
        raise HTTPException(status_code=400, detail=detail_msg)

    try:
        # Tensors are already on the correct device and have batch_dim=1
        with torch.no_grad():
            logits = fusion_model(img_tensor, nih_tensor, sensor_tensor,
                                  img_mask, nih_mask, sensor_mask)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    except Exception as e:
        logger.error(f"Error during model inference for {request_identifier}: {e}", exc_info=True)
        errors.append(f"Model inference failed: {str(e)}")
        # Return error or default predictions
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}. Errors from input processing: {'; '.join(errors)}")

    predictions_output: List[DiseaseProbability] = []
    
    class_names = mlb.classes_ if mlb and hasattr(mlb, 'classes_') else ALL_DISEASE_CLASSES_PRED
    if len(class_names) != len(probabilities):
        logger.error(f"Mismatch between number of classes in MLB/config ({len(class_names)}) and model output probabilities ({len(probabilities)}).")
        # Fallback or error
        raise HTTPException(status_code=500, detail="Internal error: Class name and probability mismatch.")


    for i, disease_name in enumerate(class_names):
        predictions_output.append(DiseaseProbability(disease_name=str(disease_name), probability=float(probabilities[i])))
        
    logger.info(f"Successfully generated predictions for {request_identifier}. Processed with errors: {errors if errors else 'None'}")
    return PredictionResponse(
        request_id=request_identifier, # Add a request_id to response
        predictions=predictions_output,
        errors=errors if errors else None
    )

@app.get("/health")
async def health_check(loaded_components: tuple = Depends(get_loaded_components)): # Checks if models load
    img_ext, fusion_m, nih_enc, dev = loaded_components
    if img_ext and fusion_m and nih_enc and dev:
        model_status = "All components loaded"
    else:
        model_status = "One or more components failed to load"
    return {"status": "healthy", "service": "Disease Prediction Service", "components_status": model_status}