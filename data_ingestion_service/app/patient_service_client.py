import httpx
from .config import PATIENT_DATA_SERVICE_URL
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Schemas for patient service interaction (can be more detailed, mirroring Pydantic models)
# For simplicity, using Dict[str, Any] for payloads now.
# In a more robust setup, these would be shared Pydantic models or a client library.

async def create_or_get_patient(patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        url = f"{PATIENT_DATA_SERVICE_URL}/patients/"
        try:
            response = await client.post(url, json=patient_data)
            response.raise_for_status() # Raises an exception for 4XX/5XX responses
            logger.info(f"Patient service call successful for patient_id_source: {patient_data.get('patient_id_source')}, status: {response.status_code}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling patient service for patient {patient_data.get('patient_id_source')}: {e.response.status_code} - {e.response.text}")
            # If it's a 409 or similar indicating "exists", we might want to fetch it,
            # but the patient_data_service endpoint should handle "create or get" logic.
            if e.response.status_code == 400 and "already registered" in e.response.text: # Based on patient_data_service logic
                 logger.warning(f"Patient {patient_data.get('patient_id_source')} likely already exists. Attempting to fetch.")
                 # Re-fetch logic if patient_service's POST isn't fully idempotent for updates
                 get_url = f"{PATIENT_DATA_SERVICE_URL}/patients/{patient_data.get('patient_id_source')}"
                 try:
                     get_response = await client.get(get_url)
                     get_response.raise_for_status()
                     return get_response.json()
                 except Exception as get_e:
                     logger.error(f"Failed to re-fetch patient {patient_data.get('patient_id_source')}: {get_e}")

            return None # Or re-raise if critical
        except httpx.RequestError as e:
            logger.error(f"Request error calling patient service for patient {patient_data.get('patient_id_source')}: {e}")
            return None

async def create_or_update_study(study_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        image_index = study_data.get("image_index")
        # The patient_data_service POST /studies/ is designed to be create-or-get for study by image_index
        url = f"{PATIENT_DATA_SERVICE_URL}/studies/"
        try:
            response = await client.post(url, json=study_data) # POST is used for create or get by image_index
            response.raise_for_status()
            logger.info(f"Study service POST call successful for image_index: {image_index}, status: {response.status_code}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error on POST /studies/ for image_index {image_index}: {e.response.status_code} - {e.response.text}")
            # If POST fails but indicates it exists (e.g. 409, or if POST only creates and PUT updates)
            # we might need a PUT call. But current patient_data_service /studies/ POST handles create-or-get
            # If patient_data_service's POST /studies/ returns the existing study on conflict, this is fine.
            # If it errors with 400/409 on conflict, we might try a PUT:
            # if e.response.status_code == 409 or (e.response.status_code == 400 and "already exists" in e.response.text):
            #     logger.warning(f"Study {image_index} likely already exists, attempting PUT to update.")
            #     put_url = f"{PATIENT_DATA_SERVICE_URL}/studies/image_index/{image_index}"
            #     # study_data for PUT should not contain patient_id_source or image_index in body for some designs
            #     update_payload = {k: v for k, v in study_data.items() if k not in ['patient_id_source', 'image_index']}
            #     try:
            #         put_response = await client.put(put_url, json=update_payload)
            #         put_response.raise_for_status()
            #         return put_response.json()
            #     except Exception as put_e:
            #         logger.error(f"Failed to PUT update study {image_index}: {put_e}")
            return None # Or re-raise if critical
        except httpx.RequestError as e:
            logger.error(f"Request error calling study service for image_index {image_index}: {e}")
            return None