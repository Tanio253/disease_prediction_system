from fastapi import FastAPI, HTTPException, Depends, Body
from minio import Minio
import logging
import io # For BytesIO
import torch # For saving tensor

from .config import BUCKET_RAW_IMAGES, BUCKET_PROCESSED_IMAGE_FEATURES
from .minio_utils import get_minio_client, SENSITIVE_check_and_create_bucket_util
from .patient_service_client import get_study_details, update_study_with_feature_path
from .image_processor import get_model_and_device, preprocess_image, extract_features
from .schemas import PreprocessImageRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Preprocessing Service",
    description="Preprocesses images and extracts features using a pre-trained CNN.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Image Preprocessing Service starting up...")
    try:
        # Initialize MinIO client and check/create output bucket
        minio = get_minio_client()
        SENSITIVE_check_and_create_bucket_util(minio, BUCKET_PROCESSED_IMAGE_FEATURES)
        logger.info(f"Ensured MinIO bucket '{BUCKET_PROCESSED_IMAGE_FEATURES}' exists.")
        
        # Load the model at startup to make first request faster and catch model loading errors early
        get_model_and_device() # This will load the model if not already loaded
        logger.info("Pre-trained model loaded/checked at startup.")
    except Exception as e:
        logger.error(f"Critical error during service startup: {e}", exc_info=True)
        # Depending on severity, might want to raise to prevent app from starting incorrectly
        # raise RuntimeError("Startup failed due to critical error") from e


@app.post("/preprocess/", summary="Preprocess an image and extract features by Image Index")
async def preprocess_image_endpoint(
    request_data: PreprocessImageRequest, # Use Pydantic model for request body
    minio: Minio = Depends(get_minio_client)
):
    image_index = request_data.image_index
    logger.info(f"Received request to preprocess image with Image Index: {image_index}")

    # 1. Get study details (including raw_image_path) from patient_data_service
    study_info = await get_study_details(image_index)
    if not study_info:
        logger.error(f"Study info not found for Image Index: {image_index}")
        raise HTTPException(status_code=404, detail=f"Study with Image Index '{image_index}' not found.")

    raw_image_path_in_bucket = study_info.get("raw_image_path")
    if not raw_image_path_in_bucket:
        logger.error(f"Raw image path not found in study info for Image Index: {image_index}")
        raise HTTPException(status_code=404, detail=f"Raw image path missing for study '{image_index}'.")

    # 2. Download raw image from MinIO
    try:
        image_data_object = minio.get_object(BUCKET_RAW_IMAGES, raw_image_path_in_bucket)
        image_bytes = image_data_object.read()
    except Exception as e:
        logger.error(f"Failed to download image '{raw_image_path_in_bucket}' from MinIO: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve image from storage for {image_index}.")
    finally:
        if 'image_data_object' in locals() and image_data_object:
            image_data_object.close()
            image_data_object.release_conn()

    # 3. Preprocess image and extract features
    try:
        model, device = get_model_and_device() # Ensures model is loaded
        processed_tensor = preprocess_image(image_bytes)
        feature_tensor = extract_features(processed_tensor, model, device)
    except Exception as e:
        logger.error(f"Error during image processing or feature extraction for {image_index}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image processing/feature extraction failed for {image_index}.")

    # 4. Save features to MinIO
    # We'll save as a PyTorch tensor file (.pt)
    feature_object_name = f"{image_index}/features.pt" 
    try:
        # Convert tensor to bytes
        buffer = io.BytesIO()
        torch.save(feature_tensor, buffer)
        buffer.seek(0) # Rewind buffer to the beginning
        feature_bytes = buffer.read()

        minio.put_object(
            BUCKET_PROCESSED_IMAGE_FEATURES,
            feature_object_name,
            io.BytesIO(feature_bytes), # Re-wrap for put_object if it re-reads
            length=len(feature_bytes),
            content_type='application/octet-stream' # Or a more specific PyTorch tensor MIME type if one exists
        )
        logger.info(f"Saved features for {image_index} to MinIO: {BUCKET_PROCESSED_IMAGE_FEATURES}/{feature_object_name}")
    except Exception as e:
        logger.error(f"Failed to save features for {image_index} to MinIO: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store processed features for {image_index}.")

    # 5. Update patient_data_service with the path to the features
    # The path stored should be relative to the bucket, i.e., just the object_name
    success_update = await update_study_with_feature_path(image_index, feature_object_name)
    if not success_update:
        # Log error, but maybe don't fail the entire request if features were saved.
        # Or, implement retry/compensation logic. For PoC, a warning is fine.
        logger.warning(f"Failed to update study {image_index} with feature path in patient_data_service. Manual update may be needed.")
        # Potentially raise HTTPException(status_code=500, detail=f"Failed to update feature path for {image_index} in database.")

    return {
        "message": "Image processed and features extracted successfully.",
        "image_index": image_index,
        "feature_path_minio_bucket": BUCKET_PROCESSED_IMAGE_FEATURES,
        "feature_path_minio_object": feature_object_name,
        "database_update_status": "success" if success_update else "failed"
    }

@app.get("/health")
async def health_check():
    # Could add a check for model availability here
    try:
        get_model_and_device() # Check if model loads without error
        model_status = "available"
    except Exception:
        model_status = "error"
    return {"status": "healthy", "service": "Image Preprocessing Service", "model_status": model_status}