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

# In model_training_service/scripts/data_loader.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import io
import random # Added
# from . import config_training # Assuming config_training.py is in the same directory or adjust path

class FusionDataset(Dataset):
    def __init__(self, patient_studies_df, minio_client, bucket_name,
                 # Expected feature dimensions (should come from config)
                 img_feature_dim, nih_feature_dim, sensor_feature_dim,
                 label_cols, training_mode=True, # Added training_mode
                 img_feat_path_col='image_features_path',
                 nih_feat_path_col='nih_features_path',
                 sensor_feat_path_col='sensor_features_path',
                 mask_image_prob=0.15, mask_nih_prob=0.15, mask_sensor_prob=0.15): # Masking probabilities

        self.patient_studies_df = patient_studies_df
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        # self.image_processor = image_processor # Not used here if features are pre-extracted

        self.img_feat_path_col = img_feat_path_col
        self.nih_feat_path_col = nih_feat_path_col
        self.sensor_feat_path_col = sensor_feat_path_col
        
        self.img_feature_dim = img_feature_dim
        self.nih_feature_dim = nih_feature_dim
        self.sensor_feature_dim = sensor_feature_dim

        self.label_cols = label_cols
        self.training_mode = training_mode # For enabling/disabling random masking
        self.mask_image_prob = mask_image_prob
        self.mask_nih_prob = mask_nih_prob
        self.mask_sensor_prob = mask_sensor_prob

    def __len__(self):
        return len(self.patient_studies_df)

    def _load_npy_from_minio(self, object_name):
        if pd.isna(object_name):
            return None
        try:
            response = self.minio_client.get_object(self.bucket_name, object_name)
            content = response.read()
            return np.load(io.BytesIO(content))
        except Exception as e:
            print(f"Error loading {object_name} from MinIO: {e}")
            return None
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()
                
    def __getitem__(self, idx):
        study_info = self.patient_studies_df.iloc[idx]

        img_features_np = self._load_npy_from_minio(study_info.get(self.img_feat_path_col))
        nih_features_np = self._load_npy_from_minio(study_info.get(self.nih_feat_path_col))
        sensor_features_np = self._load_npy_from_minio(study_info.get(self.sensor_feat_path_col))
        
        labels_np = study_info[self.label_cols].values.astype(np.float32)
        labels = torch.tensor(labels_np, dtype=torch.float32)

        # --- Image Features ---
        img_masked_flag = torch.tensor(False, dtype=torch.bool)
        if img_features_np is None:
            img_features = torch.zeros(self.img_feature_dim, dtype=torch.float32)
            img_masked_flag = torch.tensor(True, dtype=torch.bool)
        else:
            img_features = torch.tensor(img_features_np, dtype=torch.float32).squeeze() # Ensure 1D
            if self.training_mode and random.random() < self.mask_image_prob:
                img_features = torch.zeros_like(img_features)
                img_masked_flag = torch.tensor(True, dtype=torch.bool)
        
        # --- NIH Features ---
        nih_masked_flag = torch.tensor(False, dtype=torch.bool)
        if nih_features_np is None:
            nih_features = torch.zeros(self.nih_feature_dim, dtype=torch.float32)
            nih_masked_flag = torch.tensor(True, dtype=torch.bool)
        else:
            nih_features = torch.tensor(nih_features_np, dtype=torch.float32).squeeze() # Ensure 1D
            if self.training_mode and random.random() < self.mask_nih_prob:
                nih_features = torch.zeros_like(nih_features)
                nih_masked_flag = torch.tensor(True, dtype=torch.bool)

        # --- Sensor Features ---
        sensor_masked_flag = torch.tensor(False, dtype=torch.bool)
        if sensor_features_np is None:
            sensor_features = torch.zeros(self.sensor_feature_dim, dtype=torch.float32)
            sensor_masked_flag = torch.tensor(True, dtype=torch.bool)
        else:
            sensor_features = torch.tensor(sensor_features_np, dtype=torch.float32).squeeze() # Ensure 1D
            if self.training_mode and random.random() < self.mask_sensor_prob:
                sensor_features = torch.zeros_like(sensor_features)
                sensor_masked_flag = torch.tensor(True, dtype=torch.bool)
        
        # Ensure no modality is entirely masked if it's the only one present (optional safeguard)
        # For simplicity, current BERT-style masking allows any to be masked.
        # If all are masked, the model gets all zeros, which is a valid training signal.

        return {
            'img_features': img_features,
            'nih_features': nih_features,
            'sensor_features': sensor_features,
            'img_mask': img_masked_flag,
            'nih_mask': nih_masked_flag,
            'sensor_mask': sensor_masked_flag,
            'labels': labels
        }

def fusion_collate_fn(batch):
    
    img_features_batch = torch.stack([item['img_features'] for item in batch])
    nih_features_batch = torch.stack([item['nih_features'] for item in batch])
    sensor_features_batch = torch.stack([item['sensor_features'] for item in batch])
    
    img_mask_batch = torch.stack([item['img_mask'] for item in batch])
    nih_mask_batch = torch.stack([item['nih_mask'] for item in batch])
    sensor_mask_batch = torch.stack([item['sensor_mask'] for item in batch])
    
    labels_batch = torch.stack([item['labels'] for item in batch])

    return {
        'img_features': img_features_batch,
        'nih_features': nih_features_batch,
        'sensor_features': sensor_features_batch,
        'img_mask': img_mask_batch,
        'nih_mask': nih_mask_batch,
        'sensor_mask': sensor_mask_batch,
        'labels': labels_batch
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