# tabular_preprocessing_service/app/patient_service_client.py
import httpx
from .config import PATIENT_DATA_SERVICE_URL
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

async def get_study_details_for_tabular(image_index: str) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        url = f"{PATIENT_DATA_SERVICE_URL}/studies/image_index/{image_index}"
        try:
            response = await client.get(url)
            response.raise_for_status()
            details = response.json()
            logger.info(f"Successfully fetched study details for tabular processing (image_index: {image_index})")
            return details
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching study {image_index} for tabular: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error fetching study {image_index} for tabular: {e}")
            return None

async def update_study_with_tabular_feature_paths(
    image_index: str,
    nih_features_path: Optional[str] = None,
    sensor_features_path: Optional[str] = None
) -> bool:
    async with httpx.AsyncClient() as client:
        url = f"{PATIENT_DATA_SERVICE_URL}/studies/image_index/{image_index}"
        payload = {}
        if nih_features_path:
            payload["processed_nih_tabular_features_path"] = nih_features_path
        if sensor_features_path:
            payload["processed_sensor_features_path"] = sensor_features_path
        
        if not payload:
            logger.warning(f"No feature paths provided to update for study {image_index}")
            return False
            
        try:
            response = await client.put(url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully updated study {image_index} with tabular feature paths: {payload}")
            return True
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error updating study {image_index} with tabular paths: {e.response.status_code} - {e.response.text}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Request error updating study {image_index} with tabular paths: {e}")
            return False