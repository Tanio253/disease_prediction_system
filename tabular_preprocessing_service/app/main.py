# tabular_preprocessing_service/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Body
from minio import Minio
import logging
import io
import numpy as np
from pydantic import BaseModel

from .config import (
    BUCKET_RAW_SENSOR_DATA_PER_STUDY,
    BUCKET_PROCESSED_NIH_TABULAR_FEATURES,
    BUCKET_PROCESSED_SENSOR_FEATURES
)
from .minio_utils import get_minio_client, SENSITIVE_check_and_create_bucket_util
from .patient_service_client import get_study_details_for_tabular, update_study_with_tabular_feature_paths
from .nih_processor import process_nih_metadata
from .sensor_processor import process_sensor_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tabular Preprocessing Service",
    description="Preprocesses NIH metadata and sensor data to extract features.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Tabular Preprocessing Service starting up...")
    try:
        minio = get_minio_client()
        SENSITIVE_check_and_create_bucket_util(minio, BUCKET_PROCESSED_NIH_TABULAR_FEATURES)
        SENSITIVE_check_and_create_bucket_util(minio, BUCKET_PROCESSED_SENSOR_FEATURES)
        logger.info("Ensured MinIO output buckets exist.")
        # Any other startup logic, like loading pre-fitted scalers/encoders if used
    except Exception as e:
        logger.error(f"Critical error during tabular service startup: {e}", exc_info=True)

class TabularPreprocessRequest(BaseModel):
    image_index: str

@app.post("/preprocess/nih-metadata/", summary="Preprocess NIH patient metadata for a study")
async def preprocess_nih_metadata_endpoint(
    request_data: TabularPreprocessRequest,
    minio: Minio = Depends(get_minio_client)
):
    image_index = request_data.image_index
    logger.info(f"Received request to preprocess NIH metadata for Image Index: {image_index}")

    # 1. Get study details from patient_data_service
    # This should contain the structured NIH data like age, gender, view_position
    study_info = await get_study_details_for_tabular(image_index)
    if not study_info:
        logger.error(f"Study info not found for Image Index: {image_index}")
        raise HTTPException(status_code=404, detail=f"Study with Image Index '{image_index}' not found.")

    # 2. Process NIH metadata (which is now directly from study_info)
    try:
        nih_features_array = process_nih_metadata(study_info) # study_info is a dict
        if nih_features_array is None or nih_features_array.size == 0:
            logger.warning(f"NIH metadata processing returned no features for {image_index}.")
            # Decide if this is an error or an acceptable outcome (e.g., study with no processable metadata)
            # For now, let's allow it but not save/update path.
            return {"message": "NIH metadata processed, but no features were generated.", "image_index": image_index, "nih_feature_path": None}

    except Exception as e:
        logger.error(f"Error during NIH metadata processing for {image_index}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"NIH metadata processing failed for {image_index}: {str(e)}")

    # 3. Save features to MinIO (e.g., as .npy file)
    feature_object_name = f"{image_index}/nih_features.npy"
    try:
        buffer = io.BytesIO()
        np.save(buffer, nih_features_array, allow_pickle=False) # allow_pickle=False for security if not needed
        buffer.seek(0)
        feature_bytes = buffer.read()

        minio.put_object(
            BUCKET_PROCESSED_NIH_TABULAR_FEATURES,
            feature_object_name,
            io.BytesIO(feature_bytes),
            length=len(feature_bytes),
            content_type='application/octet-stream' # Or 'application/numpy'
        )
        logger.info(f"Saved NIH features for {image_index} to MinIO: {feature_object_name}")
    except Exception as e:
        logger.error(f"Failed to save NIH features for {image_index} to MinIO: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store processed NIH features for {image_index}.")

    # 4. Update patient_data_service
    success_update = await update_study_with_tabular_feature_paths(image_index, nih_features_path=feature_object_name)
    if not success_update:
        logger.warning(f"Failed to update study {image_index} with NIH feature path.")
        # Handle appropriately, maybe this is critical enough to be an error response

    return {
        "message": "NIH metadata processed and features stored successfully.",
        "image_index": image_index,
        "nih_feature_path": f"{BUCKET_PROCESSED_NIH_TABULAR_FEATURES}/{feature_object_name}",
        "database_update_status": "success" if success_update else "failed"
    }


@app.post("/preprocess/sensor-data/", summary="Preprocess raw sensor data for a study")
async def preprocess_sensor_data_endpoint(
    request_data: TabularPreprocessRequest,
    minio: Minio = Depends(get_minio_client)
):
    image_index = request_data.image_index
    logger.info(f"Received request to preprocess sensor data for Image Index: {image_index}")

    # 1. Get study details to find raw_sensor_data_path
    study_info = await get_study_details_for_tabular(image_index)
    if not study_info:
        logger.error(f"Study info not found for Image Index: {image_index}")
        raise HTTPException(status_code=404, detail=f"Study with Image Index '{image_index}' not found.")

    raw_sensor_path = study_info.get("raw_sensor_data_path")
    if not raw_sensor_path:
        logger.warning(f"Raw sensor data path not found for study {image_index}. Cannot process sensor data.")
        # This might be an acceptable case if some studies don't have sensor data
        return {"message": "Raw sensor data path not found. No sensor data processed.", "image_index": image_index, "sensor_feature_path": None}

    # 2. Download raw sensor data CSV from MinIO
    try:
        sensor_data_object = minio.get_object(BUCKET_RAW_SENSOR_DATA_PER_STUDY, raw_sensor_path)
        sensor_csv_bytes = sensor_data_object.read()
    except Exception as e:
        logger.error(f"Failed to download sensor data '{raw_sensor_path}' for {image_index} from MinIO: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sensor data from storage for {image_index}.")
    finally:
        if 'sensor_data_object' in locals() and sensor_data_object:
            sensor_data_object.close()
            sensor_data_object.release_conn()
            
    # 3. Process sensor data to extract features
    try:
        sensor_features_array = process_sensor_data(sensor_csv_bytes)
        if sensor_features_array is None or sensor_features_array.size == 0:
            logger.warning(f"Sensor data processing returned no features for {image_index}.")
            return {"message": "Sensor data processed, but no features were generated.", "image_index": image_index, "sensor_feature_path": None}
    except Exception as e:
        logger.error(f"Error during sensor data processing for {image_index}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sensor data processing failed for {image_index}: {str(e)}")

    # 4. Save features to MinIO
    feature_object_name = f"{image_index}/sensor_features.npy"
    try:
        buffer = io.BytesIO()
        np.save(buffer, sensor_features_array, allow_pickle=False)
        buffer.seek(0)
        feature_bytes = buffer.read()

        minio.put_object(
            BUCKET_PROCESSED_SENSOR_FEATURES,
            feature_object_name,
            io.BytesIO(feature_bytes),
            length=len(feature_bytes),
            content_type='application/octet-stream' # Or 'application/numpy'
        )
        logger.info(f"Saved sensor features for {image_index} to MinIO: {feature_object_name}")
    except Exception as e:
        logger.error(f"Failed to save sensor features for {image_index} to MinIO: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store processed sensor features for {image_index}.")

    # 5. Update patient_data_service
    success_update = await update_study_with_tabular_feature_paths(image_index, sensor_features_path=feature_object_name)
    if not success_update:
        logger.warning(f"Failed to update study {image_index} with sensor feature path.")

    return {
        "message": "Sensor data processed and features stored successfully.",
        "image_index": image_index,
        "sensor_feature_path": f"{BUCKET_PROCESSED_SENSOR_FEATURES}/{feature_object_name}",
        "database_update_status": "success" if success_update else "failed"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Tabular Preprocessing Service"}