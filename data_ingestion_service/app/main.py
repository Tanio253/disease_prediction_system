from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from typing import List
import pandas as pd
import logging
import io # For creating in-memory CSV

from .config import BUCKET_RAW_IMAGES, BUCKET_RAW_SENSOR_DATA_PER_STUDY
from .minio_client import get_minio_client, Minio
from .patient_service_client import create_or_get_patient, create_or_update_study
from .utils import clean_age_from_str, get_image_index_from_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Ingestion Service",
    description="Ingests raw NIH metadata, sensor data, and images for disease prediction.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    # Initialize MinIO client on startup to check connection early
    # Or simply let it be lazily loaded on first request via Depends
    try:
        client = get_minio_client()
        logger.info("MinIO client accessible at startup.")
        # Optionally check and create buckets if not handled by docker-compose
        # SENSITIVE_check_and_create_bucket(client, BUCKET_RAW_IMAGES)
        # SENSITIVE_check_and_create_bucket(client, BUCKET_RAW_SENSOR_DATA_PER_STUDY)
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client at startup: {e}")
        # Depending on policy, you might want to prevent startup
        # raise RuntimeError("MinIO client initialization failed") from e


@app.post("/ingest/batch/", summary="Ingest NIH metadata, sensor data, and images in batch")
async def ingest_batch_data(
    nih_metadata_file: UploadFile = File(..., description="CSV file with NIH metadata (your 5606 sample)"),
    sensor_data_file: UploadFile = File(..., description="CSV file with generated sensor data"),
    image_files: List[UploadFile] = File(..., description="List of X-ray image files (e.g., PNGs)"),
    minio: Minio = Depends(get_minio_client)
):
    results_summary = {
        "total_studies_in_metadata": 0,
        "studies_processed_successfully": 0,
        "patients_created_or_updated": 0,
        "studies_created_or_updated_in_db": 0,
        "images_uploaded": 0,
        "sensor_data_subsets_uploaded": 0,
        "errors": []
    }

    # 1. Load metadata and sensor data into Pandas DataFrames
    try:
        nih_df_content = await nih_metadata_file.read()
        nih_df = pd.read_csv(io.BytesIO(nih_df_content))
        results_summary["total_studies_in_metadata"] = len(nih_df)

        sensor_df_content = await sensor_data_file.read()
        sensor_df = pd.read_csv(io.BytesIO(sensor_df_content))
    except Exception as e:
        logger.error(f"Error reading uploaded CSV files: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading CSV files: {str(e)}")

    # 2. Process and Upload Images, creating a map of Image Index to MinIO path
    image_paths_in_minio = {}
    for image_file in image_files:
        if not image_file.filename:
            results_summary["errors"].append({"file": "unknown", "error": "Image file with no filename skipped."})
            continue
        
        image_index_from_name = get_image_index_from_filename(image_file.filename) # Assumes filename is Image Index
        minio_image_object_name = f"{image_index_from_name}" # Store at root of raw-images bucket for simplicity

        try:
            image_content = await image_file.read()
            minio.put_object(
                BUCKET_RAW_IMAGES,
                minio_image_object_name,
                io.BytesIO(image_content),
                length=len(image_content),
                content_type=image_file.content_type or 'application/octet-stream'
            )
            image_paths_in_minio[image_index_from_name] = minio_image_object_name # Path within bucket
            results_summary["images_uploaded"] += 1
            logger.info(f"Uploaded image '{image_index_from_name}' to MinIO bucket '{BUCKET_RAW_IMAGES}'.")
        except Exception as e:
            error_msg = f"Failed to upload image {image_index_from_name}: {e}"
            logger.error(error_msg)
            results_summary["errors"].append({"file": image_index_from_name, "error": error_msg})
        finally:
            await image_file.close()

    # 3. Iterate through NIH Metadata
    # Required columns from NIH metadata (adjust if your CSV has different names)
    # Based on your provided columns:
    # 'Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 
    # 'Patient Gender', 'View Position', 'OriginalImageWidth', 'OriginalImageHeight', 
    # 'OriginalImagePixelSpacing_x', 'OriginalImagePixelSpacing_y'
    
    # Ensure NIH DataFrame has 'Image Index' and 'Patient ID'
    if 'Image Index' not in nih_df.columns or 'Patient ID' not in nih_df.columns:
        error_msg = "NIH Metadata CSV must contain 'Image Index' and 'Patient ID' columns."
        logger.error(error_msg)
        results_summary["errors"].append({"file": nih_metadata_file.filename, "error": error_msg})
        raise HTTPException(status_code=400, detail=error_msg)

    # Ensure Sensor DataFrame has 'ImageIndex' (or map column name)
    sensor_image_index_col = 'ImageIndex' # From your sensor generation script
    if sensor_image_index_col not in sensor_df.columns:
        error_msg = f"Sensor Data CSV must contain '{sensor_image_index_col}' column."
        logger.error(error_msg)
        results_summary["errors"].append({"file": sensor_data_file.filename, "error": error_msg})
        raise HTTPException(status_code=400, detail=error_msg)


    processed_patient_ids = set()

    for index, row in nih_df.iterrows():
        try:
            image_index = str(row['Image Index'])
            patient_id_source = str(row['Patient ID'])
            
            # a. Patient Data
            patient_age_str = str(row.get('Patient Age', ''))
            patient_gender = str(row.get('Patient Gender', 'Unknown'))
            cleaned_age = clean_age_from_str(patient_age_str)

            patient_payload = {
                "patient_id_source": patient_id_source,
                "age": cleaned_age,
                "gender": patient_gender
            }
            patient_response = await create_or_get_patient(patient_payload)
            if not patient_response or 'id' not in patient_response:
                results_summary["errors"].append({"image_index": image_index, "error": "Failed to create/get patient record."})
                continue # Skip to next study if patient handling fails
            
            if patient_id_source not in processed_patient_ids:
                results_summary["patients_created_or_updated"] +=1
                processed_patient_ids.add(patient_id_source)

            # b. Study-Specific Sensor Data
            study_sensor_df = sensor_df[sensor_df[sensor_image_index_col] == image_index]
            study_raw_sensor_path_in_minio = None
            if not study_sensor_df.empty:
                sensor_csv_bytes = study_sensor_df.to_csv(index=False).encode('utf-8')
                minio_sensor_object_name = f"{image_index}/sensor_readings.csv"
                minio.put_object(
                    BUCKET_RAW_SENSOR_DATA_PER_STUDY,
                    minio_sensor_object_name,
                    io.BytesIO(sensor_csv_bytes),
                    length=len(sensor_csv_bytes),
                    content_type='text/csv'
                )
                study_raw_sensor_path_in_minio = minio_sensor_object_name # Path within bucket
                results_summary["sensor_data_subsets_uploaded"] += 1
                logger.info(f"Uploaded sensor data for '{image_index}' to MinIO bucket '{BUCKET_RAW_SENSOR_DATA_PER_STUDY}'.")
            else:
                logger.warning(f"No sensor data found for Image Index: {image_index}")
                results_summary["errors"].append({"image_index": image_index, "warning": "No sensor data found."})


            # c. Raw Image Path (already uploaded and mapped)
            raw_image_path_in_minio_bucket = image_paths_in_minio.get(image_index)
            if not raw_image_path_in_minio_bucket:
                results_summary["errors"].append({"image_index": image_index, "warning": f"Image file not found for metadata entry {image_index}."})
                # Decide if to continue or skip this study record
                # For now, we'll allow study record creation even if raw image is missing, path will be None.

            # d. Create/Update Study Record
            study_payload = {
                "image_index": image_index,
                "patient_id_source": patient_id_source, # patient_data_service uses this to link
                "follow_up_number": int(row.get('Follow-up #', 0)),
                "view_position": str(row.get('View Position', '')),
                "finding_labels": str(row.get('Finding Labels', 'No Finding')),
                "raw_image_path": raw_image_path_in_minio_bucket, # This is path *within* BUCKET_RAW_IMAGES
                "raw_sensor_data_path": study_raw_sensor_path_in_minio, # Path *within* BUCKET_RAW_SENSOR_DATA_PER_STUDY
                "original_image_width": int(row.get('OriginalImageWidth', 0)) if pd.notna(row.get('OriginalImageWidth')) else None,
                "original_image_height": int(row.get('OriginalImageHeight', 0)) if pd.notna(row.get('OriginalImageHeight')) else None,
                "original_pixel_spacing_x": float(row.get('OriginalImagePixelSpacing_x', 0.0)) if pd.notna(row.get('OriginalImagePixelSpacing_x')) else None,
                "original_pixel_spacing_y": float(row.get('OriginalImagePixelSpacing_y', 0.0)) if pd.notna(row.get('OriginalImagePixelSpacing_y')) else None,
            }
            
            study_response = await create_or_update_study(study_payload)
            if not study_response or 'id' not in study_response:
                results_summary["errors"].append({"image_index": image_index, "error": "Failed to create/update study record in DB."})
                continue
            
            results_summary["studies_created_or_updated_in_db"] += 1
            results_summary["studies_processed_successfully"] += 1

        except Exception as e:
            error_msg = f"Error processing row {index} (Image Index: {row.get('Image Index', 'N/A')}): {e}"
            logger.error(error_msg, exc_info=True) # Log with stack trace
            results_summary["errors"].append({"image_index": row.get('Image Index', 'N/A'), "error": str(e)})

    # Close uploaded files (FastAPI might do this automatically for UploadFile, but good practice)
    await nih_metadata_file.close()
    await sensor_data_file.close()

    logger.info(f"Batch ingestion summary: {results_summary}")
    return results_summary

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Data Ingestion Service"}