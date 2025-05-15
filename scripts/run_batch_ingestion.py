import httpx
import os
import glob
import logging
from typing import List, Tuple, Optional, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
API_GATEWAY_URL = "http://localhost:8080" # Your API Gateway's host and port
INGESTION_ENDPOINT = "/api/v1/ingest/ingest/batch/" # Full path via gateway

METADATA_CSV_PATH = "./data/raw/tabular/sampled_nih_metadata.csv"
SENSOR_DATA_CSV_PATH = "./data/raw/tabular/simulated_sensor_data.csv"
IMAGES_DIR_PATH = "./data/raw/images/" # Directory containing all 5606 .png images

# If you want to test with a smaller number of images first:
# MAX_IMAGES_TO_SEND = 10 # Set to None to send all found images
MAX_IMAGES_TO_SEND = 500 # Send all images matching the pattern

# Timeout for the HTTP request (in seconds)
# Ingesting many files can take time.
REQUEST_TIMEOUT = 300.0 # 5 minutes, adjust as needed

def find_image_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Finds all .png image files in the specified directory."""
    search_pattern = os.path.join(directory, "*.png") # Adjust if image format is different
    image_paths = glob.glob(search_pattern)
    logger.info(f"Found {len(image_paths)} images in '{directory}'.")
    if max_files is not None and len(image_paths) > max_files:
        logger.info(f"Limiting to {max_files} images for this run.")
        return image_paths[:max_files]
    return image_paths

def prepare_files_for_upload(
    metadata_path: str,
    sensor_data_path: str,
    image_paths: List[str]
) -> List[Tuple[str, Tuple[str, Any, str]]]:
    """
    Prepares the 'files' list for httpx.post in multipart/form-data format.
    Each item in the list is a tuple: (form_field_name, (filename, file_object, content_type))
    """
    files_payload = []

    # 1. Metadata CSV
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata CSV not found at: {metadata_path}")
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    files_payload.append(
        ("nih_metadata_file", (os.path.basename(metadata_path), open(metadata_path, "rb"), "text/csv"))
    )
    logger.info(f"Prepared metadata file: {metadata_path}")

    # 2. Sensor Data CSV
    if not os.path.exists(sensor_data_path):
        logger.error(f"Sensor data CSV not found at: {sensor_data_path}")
        raise FileNotFoundError(f"Sensor data CSV not found: {sensor_data_path}")
    files_payload.append(
        ("sensor_data_file", (os.path.basename(sensor_data_path), open(sensor_data_path, "rb"), "text/csv"))
    )
    logger.info(f"Prepared sensor data file: {sensor_data_path}")

    # 3. Image Files
    # The FastAPI endpoint expects 'image_files' as the field name for a list of files.
    # httpx handles multiple files with the same field name correctly.
    for img_path in image_paths:
        if not os.path.exists(img_path):
            logger.warning(f"Image file not found (will be skipped): {img_path}")
            continue
        files_payload.append(
            ("image_files", (os.path.basename(img_path), open(img_path, "rb"), "image/png")) # Assuming PNG
        )
    logger.info(f"Prepared {len([f for f in files_payload if f[0] == 'image_files'])} image files for upload.")
    
    return files_payload

def run_ingestion():
    """
    Finds data and image files and sends them to the data_ingestion_service.
    """
    logger.info("Starting data ingestion process...")
    
    image_file_paths = find_image_files(IMAGES_DIR_PATH, MAX_IMAGES_TO_SEND)
    if not image_file_paths:
        logger.error("No image files found to ingest. Aborting.")
        return

    try:
        files_for_request = prepare_files_for_upload(
            METADATA_CSV_PATH, SENSOR_DATA_CSV_PATH, image_file_paths
        )
    except FileNotFoundError:
        logger.error("One or more required data files not found. Aborting ingestion.")
        return
    
    if not files_for_request:
        logger.error("No files prepared for upload (metadata/sensor missing or no images). Aborting.")
        return

    target_url = f"{API_GATEWAY_URL.rstrip('/')}{INGESTION_ENDPOINT}"
    logger.info(f"Targeting ingestion endpoint: {target_url}")

    # Using synchronous httpx client for this script
    client = httpx.Client(timeout=REQUEST_TIMEOUT)
    
    opened_files = [f_obj for _, (_, f_obj, _) in files_for_request] # Keep track to close them

    try:
        logger.info(f"Sending {len(files_for_request)} total file parts (2 CSVs + {len(image_file_paths)} images) ... this may take a while.")
        start_time = time.time()
        
        # httpx will correctly format this as multipart/form-data
        # The 'files' parameter takes a list of tuples as prepared.
        response = client.post(target_url, files=files_for_request)
        
        end_time = time.time()
        logger.info(f"Request completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Response Status Code: {response.status_code}")
        
        try:
            response_json = response.json()
            logger.info("Response JSON:")
            import json # For pretty printing
            logger.info(json.dumps(response_json, indent=2))
            if response.status_code == 200 or response.status_code == 202 : # 202 if it's async
                 logger.info("Ingestion request successful (according to status code).")
                 # Further check response_json for detailed success/failure counts from the service
                 if response_json.get("studies_processed_successfully") == response_json.get("total_studies_in_metadata"):
                     logger.info("All studies appear to have been processed by the ingestion service.")
                 else:
                     logger.warning("There might be discrepancies in studies processed vs. total. Check response details.")
            else:
                logger.error("Ingestion request failed. See response JSON above for details.")

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON response. Raw response text:")
            logger.error(response.text)

    except httpx.ConnectError as e:
        logger.error(f"Connection Error: Could not connect to the API Gateway at {target_url}. Details: {e}")
    except httpx.ReadTimeout:
        logger.error(f"Read Timeout: The request to {target_url} timed out after {REQUEST_TIMEOUT} seconds.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during ingestion: {e}", exc_info=True)
    finally:
        for f_obj in opened_files:
            if hasattr(f_obj, 'close') and not f_obj.closed:
                f_obj.close()
        client.close()
        logger.info("All file objects closed and HTTP client closed.")

if __name__ == "__main__":
    # Before running, ensure:
    # 1. All services are up: `docker-compose up -d` (especially api_gateway and data_ingestion_service)
    # 2. Paths in CONFIGURATION section are correct.
    # 3. Your images are in IMAGES_DIR_PATH.
    run_ingestion()