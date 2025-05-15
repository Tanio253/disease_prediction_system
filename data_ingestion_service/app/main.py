from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks # Add BackgroundTasks
from typing import List
import pandas as pd
import logging
import io
import httpx # For making calls to other services

from .config import (
    BUCKET_RAW_IMAGES, BUCKET_RAW_SENSOR_DATA_PER_STUDY,
    IMAGE_PREPROCESSING_ENDPOINT, TABULAR_NIH_PREPROCESSING_ENDPOINT, TABULAR_SENSOR_PREPROCESSING_ENDPOINT # Import new endpoints
)
from .minio_client import get_minio_client, Minio
from .patient_service_client import create_or_get_patient, create_or_update_study
from .utils import clean_age_from_str, get_image_index_from_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Ingestion Service",
    description="Ingests raw NIH metadata, sensor data, and images for disease prediction and triggers preprocessing.",
    version="0.2.0" # Version bump
)

# --- Background Task for Preprocessing ---
async def trigger_preprocessing_tasks(image_index: str):
    """
    Asynchronously calls the preprocessing services for a given image_index.
    """
    payload = {"image_index": image_index}
    timeout_config = httpx.Timeout(10.0, read=120.0) # 10s connect, 120s read

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        # Trigger Image Preprocessing
        try:
            logger.info(f"Triggering image preprocessing for {image_index} at {IMAGE_PREPROCESSING_ENDPOINT}")
            response_img = await client.post(IMAGE_PREPROCESSING_ENDPOINT, json=payload)
            response_img.raise_for_status()
            logger.info(f"Image preprocessing triggered for {image_index}. Status: {response_img.status_code}. Response: {response_img.json()}")
        except Exception as e:
            logger.error(f"Failed to trigger image preprocessing for {image_index}: {e}", exc_info=True)

        # Trigger Tabular NIH Metadata Preprocessing
        if TABULAR_NIH_PREPROCESSING_ENDPOINT: # Check if configured
            try:
                logger.info(f"Triggering NIH tabular preprocessing for {image_index} at {TABULAR_NIH_PREPROCESSING_ENDPOINT}")
                response_nih = await client.post(TABULAR_NIH_PREPROCESSING_ENDPOINT, json=payload)
                response_nih.raise_for_status()
                logger.info(f"NIH tabular preprocessing triggered for {image_index}. Status: {response_nih.status_code}. Response: {response_nih.json()}")
            except Exception as e:
                logger.error(f"Failed to trigger NIH tabular preprocessing for {image_index}: {e}", exc_info=True)

        # Trigger Tabular Sensor Data Preprocessing
        if TABULAR_SENSOR_PREPROCESSING_ENDPOINT: # Check if configured
            try:
                logger.info(f"Triggering Sensor tabular preprocessing for {image_index} at {TABULAR_SENSOR_PREPROCESSING_ENDPOINT}")
                response_sensor = await client.post(TABULAR_SENSOR_PREPROCESSING_ENDPOINT, json=payload)
                response_sensor.raise_for_status()
                logger.info(f"Sensor tabular preprocessing triggered for {image_index}. Status: {response_sensor.status_code}. Response: {response_sensor.json()}")
            except Exception as e:
                logger.error(f"Failed to trigger Sensor tabular preprocessing for {image_index}: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    try:
        client = get_minio_client()
        logger.info("MinIO client accessible at startup.")
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client at startup: {e}")

@app.post("/ingest/batch/", summary="Ingest NIH metadata, sensor data, and images in batch and trigger preprocessing")
async def ingest_batch_data(
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    nih_metadata_file: UploadFile = File(..., description="CSV file with NIH metadata"),
    sensor_data_file: UploadFile = File(..., description="CSV file with generated sensor data"),
    image_files: List[UploadFile] = File(..., description="List of X-ray image files"),
    minio: Minio = Depends(get_minio_client)
):
    results_summary = {
        "total_studies_in_metadata": 0,
        "studies_ingested_successfully": 0, # Renamed for clarity
        "preprocessing_tasks_queued": 0,
        "patients_created_or_updated": 0,
        "images_uploaded_to_minio": 0,
        "sensor_subsets_uploaded_to_minio": 0,
        "errors": [],
        "warnings": [] # Added warnings for non-critical issues
    }

    # 1. Load metadata and sensor data
    try:
        nih_df_content = await nih_metadata_file.read()
        nih_df = pd.read_csv(io.BytesIO(nih_df_content), dtype={'Patient ID': str, 'Image Index': str})
        results_summary["total_studies_in_metadata"] = len(nih_df)

        sensor_df_content = await sensor_data_file.read()
        sensor_df = pd.read_csv(io.BytesIO(sensor_df_content), dtype={'ImageIndex': str, 'PatientID_Source': str})
    except Exception as e:
        logger.error(f"Error reading uploaded CSV files: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading CSV files: {str(e)}")
    finally:
        await nih_metadata_file.close()
        await sensor_data_file.close()


    # 2. Process and Upload Images
    image_paths_in_minio = {} # Maps Image Index (filename) to MinIO object name
    for image_file in image_files:
        if not image_file.filename:
            results_summary["warnings"].append({"file": "unknown", "issue": "Image file with no filename skipped."})
            await image_file.close() # Close even if skipped
            continue
        
        image_index_from_name = get_image_index_from_filename(image_file.filename)
        minio_image_object_name = f"{image_index_from_name}" # Object name is just the image index

        try:
            image_content = await image_file.read()
            minio.put_object(
                BUCKET_RAW_IMAGES,
                minio_image_object_name,
                io.BytesIO(image_content),
                length=len(image_content),
                content_type=image_file.content_type or 'image/png' # Default to image/png
            )
            image_paths_in_minio[image_index_from_name] = minio_image_object_name
            results_summary["images_uploaded_to_minio"] += 1
        except Exception as e:
            error_msg = f"Failed to upload image {image_index_from_name} to MinIO: {e}"
            logger.error(error_msg)
            results_summary["errors"].append({"file": image_index_from_name, "error": error_msg})
        finally:
            await image_file.close()

    # 3. Iterate through NIH Metadata
    required_nih_cols = ['Image Index', 'Patient ID', 'Patient Age', 'Finding Labels']
    if not all(col in nih_df.columns for col in required_nih_cols):
        missing_cols_str = ", ".join([col for col in required_nih_cols if col not in nih_df.columns])
        error_msg = f"NIH Metadata CSV must contain required columns. Missing: {missing_cols_str}."
        logger.error(error_msg)
        results_summary["errors"].append({"file": nih_metadata_file.filename or "NIH Metadata", "error": error_msg})
        # Return partial results or raise HTTP Exception
        return results_summary # Or raise HTTPException

    sensor_image_index_col = 'ImageIndex'
    if sensor_image_index_col not in sensor_df.columns:
        error_msg = f"Sensor Data CSV must contain '{sensor_image_index_col}' column."
        logger.error(error_msg)
        results_summary["errors"].append({"file": sensor_data_file.filename or "Sensor Data", "error": error_msg})
        return results_summary # Or raise HTTPException

    processed_patient_ids_source = set()

    for index, row in nih_df.iterrows():
        current_image_index = str(row['Image Index'])
        patient_id_source = str(row['Patient ID'])
        
        try:
            # a. Patient Data
            patient_age_str = str(row.get('Patient Age', ''))
            patient_gender = str(row.get('Patient Gender', 'Unknown')) # Handle missing if any
            cleaned_age = clean_age_from_str(patient_age_str)

            patient_payload = {
                "patient_id_source": patient_id_source, "age": cleaned_age, "gender": patient_gender
            }
            patient_response = await create_or_get_patient(patient_payload)
            if not patient_response or 'id' not in patient_response:
                results_summary["errors"].append({"image_index": current_image_index, "error": "Failed to create/get patient record."})
                continue
            
            if patient_id_source not in processed_patient_ids_source:
                results_summary["patients_created_or_updated"] += 1
                processed_patient_ids_source.add(patient_id_source)

            # b. Study-Specific Sensor Data
            # Ensure ImageIndex in sensor_df is also treated as string for matching
            study_sensor_df = sensor_df[sensor_df[sensor_image_index_col] == current_image_index]
            study_raw_sensor_path_in_minio = None
            if not study_sensor_df.empty:
                sensor_csv_bytes = study_sensor_df.to_csv(index=False).encode('utf-8')
                minio_sensor_object_name = f"{current_image_index}/sensor_readings.csv"
                minio.put_object(
                    BUCKET_RAW_SENSOR_DATA_PER_STUDY, minio_sensor_object_name,
                    io.BytesIO(sensor_csv_bytes), len(sensor_csv_bytes), content_type='text/csv'
                )
                study_raw_sensor_path_in_minio = minio_sensor_object_name
                results_summary["sensor_subsets_uploaded_to_minio"] += 1
            else:
                results_summary["warnings"].append({"image_index": current_image_index, "issue": "No sensor data found for this study."})

            # c. Raw Image Path
            raw_image_path_in_minio_bucket = image_paths_in_minio.get(current_image_index)
            if not raw_image_path_in_minio_bucket:
                 results_summary["warnings"].append({"image_index": current_image_index, "issue": f"Image file not found in upload for metadata entry {current_image_index}."})


            # d. Create/Update Study Record in DB
            study_payload = {
                "image_index": current_image_index,
                "patient_id_source": patient_id_source,
                "follow_up_number": int(row.get('Follow-up #', 0)),
                "view_position": str(row.get('View Position', '')),
                "finding_labels": str(row.get('Finding Labels', 'No Finding')),
                "raw_image_path": raw_image_path_in_minio_bucket,
                "raw_sensor_data_path": study_raw_sensor_path_in_minio,
                "original_image_width": int(row.get('OriginalImageWidth', 0)) if pd.notna(row.get('OriginalImageWidth')) else None,
                "original_image_height": int(row.get('OriginalImageHeight', 0)) if pd.notna(row.get('OriginalImageHeight')) else None,
                "original_pixel_spacing_x": float(row.get('OriginalImagePixelSpacing_x', 0.0)) if pd.notna(row.get('OriginalImagePixelSpacing_x')) else None,
                "original_pixel_spacing_y": float(row.get('OriginalImagePixelSpacing_y', 0.0)) if pd.notna(row.get('OriginalImagePixelSpacing_y')) else None,
            }
            
            study_db_response = await create_or_update_study(study_payload)
            if not study_db_response or 'id' not in study_db_response:
                results_summary["errors"].append({"image_index": current_image_index, "error": "Failed to create/update study record in DB."})
                continue
            
            results_summary["studies_ingested_successfully"] += 1
            
            # e. Add preprocessing tasks to background
            background_tasks.add_task(trigger_preprocessing_tasks, current_image_index)
            results_summary["preprocessing_tasks_queued"] += 1
            
        except Exception as e:
            error_msg = f"Error processing metadata row {index} (Image Index: {current_image_index}): {e}"
            logger.error(error_msg, exc_info=True)
            results_summary["errors"].append({"image_index": current_image_index, "error": str(e)})

    logger.info(f"Batch ingestion processing loop finished. Summary: {results_summary}")
    return results_summary

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Data Ingestion Service"}