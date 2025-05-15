import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import requests # For synchronous calls to patient_data_service
import io
from minio import Minio
import logging
import os

from .config_training import (
    PATIENT_DATA_SERVICE_URL, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_USE_SSL,
    BUCKET_PROCESSED_IMAGE_FEATURES, BUCKET_PROCESSED_NIH_TABULAR_FEATURES, BUCKET_PROCESSED_SENSOR_FEATURES,
    IMAGE_FEATURE_DIM, NIH_TABULAR_FEATURE_DIM, SENSOR_FEATURE_DIM # For validation
)
from .utils import encode_labels # Assuming utils.py is in the same directory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FusionDataset(Dataset):
    def __init__(self, study_metadatas, minio_client):
        """
        Args:
            study_metadatas (list of dicts): List of study records from patient_data_service,
                                             each containing paths to processed features.
            minio_client (Minio): Initialized MinIO client.
        """
        self.study_metadatas = study_metadatas
        self.minio_client = minio_client
        logger.info(f"FusionDataset initialized with {len(self.study_metadatas)} studies.")

    def __len__(self):
        return len(self.study_metadatas)

    def _load_feature_from_minio(self, bucket_name, object_name):
        try:
            response = self.minio_client.get_object(bucket_name, object_name)
            feature_bytes = response.read()
            response.close()
            response.release_conn()

            if object_name.endswith('.pt'):
                feature_tensor = torch.load(io.BytesIO(feature_bytes))
            elif object_name.endswith('.npy'):
                feature_array = np.load(io.BytesIO(feature_bytes))
                feature_tensor = torch.from_numpy(feature_array).float() # Ensure float
            else:
                logger.error(f"Unsupported feature file type for {object_name} in bucket {bucket_name}")
                return None
            return feature_tensor
        except Exception as e:
            logger.error(f"Error loading feature {object_name} from bucket {bucket_name}: {e}", exc_info=True)
            return None

    def __getitem__(self, idx):
        study_info = self.study_metadatas[idx]
        image_index = study_info.get('image_index', f"unknown_index_{idx}") # For logging

        # Load Image Features
        img_feat_path = study_info.get('processed_image_features_path')
        if img_feat_path:
            image_features = self._load_feature_from_minio(BUCKET_PROCESSED_IMAGE_FEATURES, img_feat_path)
            if image_features is None: # Fallback if loading fails
                logger.warning(f"Image features failed to load for {image_index}, using zeros. Path: {img_feat_path}")
                image_features = torch.zeros(IMAGE_FEATURE_DIM) # Must match expected dim
            elif image_features.shape[0] != IMAGE_FEATURE_DIM: # Basic shape check
                logger.error(f"Mismatched image feature dim for {image_index}! Expected {IMAGE_FEATURE_DIM}, Got {image_features.shape}. Path: {img_feat_path}. Using zeros.")
                image_features = torch.zeros(IMAGE_FEATURE_DIM)
        else:
            logger.warning(f"Missing image feature path for {image_index}, using zeros.")
            image_features = torch.zeros(IMAGE_FEATURE_DIM)


        # Load NIH Tabular Features
        nih_feat_path = study_info.get('processed_nih_tabular_features_path')
        if nih_feat_path:
            nih_features = self._load_feature_from_minio(BUCKET_PROCESSED_NIH_TABULAR_FEATURES, nih_feat_path)
            if nih_features is None:
                logger.warning(f"NIH features failed to load for {image_index}, using zeros. Path: {nih_feat_path}")
                nih_features = torch.zeros(NIH_TABULAR_FEATURE_DIM)
            elif nih_features.shape[0] != NIH_TABULAR_FEATURE_DIM:
                 logger.error(f"Mismatched NIH feature dim for {image_index}! Expected {NIH_TABULAR_FEATURE_DIM}, Got {nih_features.shape}. Path: {nih_feat_path}. Using zeros.")
                 nih_features = torch.zeros(NIH_TABULAR_FEATURE_DIM)
        else:
            logger.warning(f"Missing NIH feature path for {image_index}, using zeros.")
            nih_features = torch.zeros(NIH_TABULAR_FEATURE_DIM)

        # Load Sensor Features
        sensor_feat_path = study_info.get('processed_sensor_features_path')
        if sensor_feat_path:
            sensor_features = self._load_feature_from_minio(BUCKET_PROCESSED_SENSOR_FEATURES, sensor_feat_path)
            if sensor_features is None:
                logger.warning(f"Sensor features failed to load for {image_index}, using zeros. Path: {sensor_feat_path}")
                sensor_features = torch.zeros(SENSOR_FEATURE_DIM)
            elif sensor_features.shape[0] != SENSOR_FEATURE_DIM:
                 logger.error(f"Mismatched sensor feature dim for {image_index}! Expected {SENSOR_FEATURE_DIM}, Got {sensor_features.shape}. Path: {sensor_feat_path}. Using zeros.")
                 sensor_features = torch.zeros(SENSOR_FEATURE_DIM)
        else:
            logger.warning(f"Missing sensor feature path for {image_index}, using zeros.")
            sensor_features = torch.zeros(SENSOR_FEATURE_DIM)

        # Encode Labels
        finding_labels_str = study_info.get('finding_labels', "")
        labels_encoded = encode_labels(finding_labels_str) # From utils.py
        labels_tensor = torch.from_numpy(labels_encoded).float() # Ensure float for BCEWithLogitsLoss

        return {
            'image_features': image_features.squeeze(), # Ensure 1D if loaded with extra dim
            'nih_features': nih_features.squeeze(),
            'sensor_features': sensor_features.squeeze(),
            'labels': labels_tensor,
            'image_index': image_index # For debugging or tracking if needed
        }

def fetch_training_ready_studies_from_service():
    """Fetches study metadata from patient_data_service."""
    url = f"{PATIENT_DATA_SERVICE_URL}/studies/training_ready/?limit=10000" # Adjust limit as needed
    try:
        response = requests.get(url, timeout=30) # Added timeout
        response.raise_for_status()
        studies = response.json()
        if not studies:
            logger.warning("No studies returned from patient_data_service for training.")
            return []
        logger.info(f"Fetched {len(studies)} studies marked as training_ready from patient service.")
        return studies
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching studies from patient data service: {e}", exc_info=True)
        return [] # Return empty list on error

def get_data_loaders(batch_size, test_split_ratio=0.2, num_workers=0): # num_workers=0 for simpler debugging
    """Creates and returns train and validation DataLoader instances."""
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL
    )

    all_studies = fetch_training_ready_studies_from_service()
    if not all_studies:
        logger.error("No studies available to create data loaders. Exiting.")
        return None, None # Or raise an error

    # Shuffle and split data (simple random split)
    np.random.shuffle(all_studies) # Shuffle in place
    split_idx = int(len(all_studies) * (1 - test_split_ratio))
    train_studies = all_studies[:split_idx]
    val_studies = all_studies[split_idx:]

    if not train_studies:
        logger.error("No training studies after split. Check data and split ratio.")
        return None, None
    if not val_studies: # It's possible to have no validation studies if dataset is too small
        logger.warning("No validation studies after split. Consider a smaller split ratio or more data.")


    logger.info(f"Training set size: {len(train_studies)}")
    logger.info(f"Validation set size: {len(val_studies)}")

    train_dataset = FusionDataset(train_studies, minio_client)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = None
    if val_studies: # Only create val_loader if there are validation studies
        val_dataset = FusionDataset(val_studies, minio_client)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == '__main__':
    # Test the data loader
    logger.info("Testing DataLoader functionality...")
    
    # Ensure your patient_data_service and MinIO are running and populated with some test data.
    # You'd need to have run ingestion and preprocessing for a few samples.
    
    # For local testing, you might need to mock patient_data_service response or MinIO
    # Or ensure services are up and accessible from where you run this.
    # This assumes MinIO client can connect and patient_data_service is reachable.
    
    # Override config for local test if needed (e.g., if patient_data_service is on localhost)
    # os.environ["PATIENT_DATA_SERVICE_URL"] = "http://localhost:8000" # If running service locally
    # from .config_training import PATIENT_DATA_SERVICE_URL # Re-import if changed via env
    # print(f"Test using Patient Service URL: {PATIENT_DATA_SERVICE_URL}")


    train_dl, val_dl = get_data_loaders(batch_size=4)

    if train_dl:
        logger.info(f"Train DataLoader created. Number of batches: {len(train_dl)}")
        for i, batch in enumerate(train_dl):
            logger.info(f"Train Batch {i+1}:")
            logger.info(f"  Image features shape: {batch['image_features'].shape}")
            logger.info(f"  NIH features shape: {batch['nih_features'].shape}")
            logger.info(f"  Sensor features shape: {batch['sensor_features'].shape}")
            logger.info(f"  Labels shape: {batch['labels'].shape}")
            logger.info(f"  Sample Image Index from batch: {batch['image_index'][0]}") # Print one image_index
            if i == 1: # Print info for 2 batches
                break
    else:
        logger.error("Failed to create Train DataLoader.")

    if val_dl:
        logger.info(f"Validation DataLoader created. Number of batches: {len(val_dl)}")
        for i, batch in enumerate(val_dl):
            logger.info(f"Validation Batch {i+1}: Image features shape: {batch['image_features'].shape}")
            if i == 0:
                break
    elif len(fetch_training_ready_studies_from_service()) > 0 and (len(fetch_training_ready_studies_from_service()) * 0.2 >=1) :
        logger.warning("Validation DataLoader not created, but there should be validation data.")
    else:
         logger.info("Validation DataLoader not created (likely due to small dataset or no validation split).")