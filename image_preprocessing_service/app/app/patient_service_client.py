import httpx
from .config import PATIENT_DATA_SERVICE_URL
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

async def get_study_details(image_index: str) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        url = f"{PATIENT_DATA_SERVICE_URL}/studies/image_index/{image_index}"
        try:
            response = await client.get(url)
            response.raise_for_status()
            logger.info(f"Successfully fetched study details for image_index: {image_index}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching study {image_index}: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error fetching study {image_index}: {e}")
            return None

async def update_study_with_feature_path(image_index: str, feature_path: str) -> bool:
    async with httpx.AsyncClient() as client:
        url = f"{PATIENT_DATA_SERVICE_URL}/studies/image_index/{image_index}"
        payload = {"processed_image_features_path": feature_path}
        try:
            response = await client.put(url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully updated study {image_index} with feature path: {feature_path}")
            return True
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error updating study {image_index}: {e.response.status_code} - {e.response.text}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Request error updating study {image_index}: {e}")
            return False