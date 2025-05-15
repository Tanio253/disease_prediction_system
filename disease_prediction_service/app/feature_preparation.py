import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
import io
import logging
from sklearn.preprocessing import OneHotEncoder
from typing import Optional

from .config import (
    IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD,
    NIH_CATEGORICAL_COLS_PRED, NIH_NUMERICAL_COLS_PRED, # Patient Age_cleaned
    SENSOR_COLUMNS_TO_AGGREGATE_PRED, SENSOR_AGGREGATIONS_PRED,
    SENSOR_FEATURE_DIM_PRED # For default sensor features
)

logger = logging.getLogger(__name__)

# --- Image Feature Preparation ---
def prepare_image_features_for_inference(
    image_bytes: bytes,
    image_feature_extractor_model: torch.nn.Module,
    device: torch.device
) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
        ])
        img_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim, move to device

        with torch.no_grad():
            features = image_feature_extractor_model(img_tensor)
        return features.squeeze(0).cpu() # Remove batch dim, move to CPU
    except Exception as e:
        logger.error(f"Error preparing image features for inference: {e}", exc_info=True)
        raise # Re-raise to be caught by endpoint handler

# --- NIH-like Tabular Feature Preparation ---
def prepare_nih_features_for_inference(
    patient_age: int,
    patient_gender: str,
    view_position: str,
    encoder: OneHotEncoder # Pre-fitted OneHotEncoder
) -> torch.Tensor:
    try:
        # Create a DataFrame matching the structure expected by the encoder
        data = {
            'Patient Age_cleaned': [patient_age], # Assuming age is already cleaned
            'Patient Gender': [patient_gender],
            'View Position': [view_position]
        }
        # Ensure all categorical columns defined in config are present for encoder consistency
        for col in NIH_CATEGORICAL_COLS_PRED:
            if col not in data: # Should not happen if inputs are validated
                data[col] = ['Missing'] # Or a default category the encoder knows

        input_df = pd.DataFrame(data)

        # Apply OneHotEncoding to specified categorical columns
        # Ensure the columns for encoding are exactly those the encoder was fitted on
        # and are present in input_df
        cols_to_encode = [col for col in NIH_CATEGORICAL_COLS_PRED if col in input_df.columns]
        if not cols_to_encode:
            encoded_array = np.array([]).reshape(1,-1) # Empty 2D array if no categorical features
        else:
            encoded_array = encoder.transform(input_df[cols_to_encode]) # encoder expects 2D array
        
        encoded_features_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cols_to_encode))

        # Numerical features (just age in this simplified example)
        numerical_features_df = input_df[['Patient Age_cleaned']].copy() # Or select from NIH_NUMERICAL_COLS_PRED
        # Apply scaling if a scaler was used during training and loaded here

        # Combine numerical and encoded categorical features
        # Order must be consistent with training!
        if not numerical_features_df.empty and not encoded_features_df.empty:
            final_features_df = pd.concat([numerical_features_df.reset_index(drop=True),
                                           encoded_features_df.reset_index(drop=True)], axis=1)
        elif not encoded_features_df.empty:
            final_features_df = encoded_features_df
        elif not numerical_features_df.empty: # Only numerical
            final_features_df = numerical_features_df
        else: # Should not happen if input is validated
            logger.error("No NIH features generated. This is unexpected.")
            # This needs to match NIH_TABULAR_FEATURE_DIM_PRED
            # This is a critical error, means NIH_TABULAR_FEATURE_DIM_PRED won't be met.
            # Raise error or return a zero tensor of correct dimension
            raise ValueError("NIH feature preparation resulted in no features.")


        nih_features_np = final_features_df.to_numpy().squeeze() # Squeeze if single sample

        # CRITICAL: Ensure the output dimension matches NIH_TABULAR_FEATURE_DIM_PRED
        # This is a manual check point. If NIH_TABULAR_FEATURE_DIM_PRED is, say, 1 (age) + 7 (OHE) = 8,
        # then nih_features_np should have length 8.
        # Example: logger.info(f"Prepared NIH features shape: {nih_features_np.shape}")

        return torch.from_numpy(nih_features_np).float()
    except Exception as e:
        logger.error(f"Error preparing NIH features for inference: {e}", exc_info=True)
        raise


# --- Sensor Feature Preparation ---
def prepare_sensor_features_for_inference(
    sensor_data_csv_bytes: Optional[bytes]
) -> torch.Tensor:
    if sensor_data_csv_bytes is None:
        logger.info("No sensor data provided for inference, using default zero vector.")
        # Return a zero vector of the expected sensor feature dimension
        return torch.zeros(SENSOR_FEATURE_DIM_PRED)

    try:
        sensor_df = pd.read_csv(io.BytesIO(sensor_data_csv_bytes))
        if sensor_df.empty:
            logger.warning("Provided sensor data CSV is empty, using default zero vector.")
            return torch.zeros(SENSOR_FEATURE_DIM_PRED)

        aggregated_features = []
        # Ensure SENSOR_COLUMNS_TO_AGGREGATE_PRED are present in sensor_df
        for col in SENSOR_COLUMNS_TO_AGGREGATE_PRED:
            if col not in sensor_df.columns:
                logger.warning(f"Sensor column '{col}' not found in provided CSV. Aggregations for it will be zero.")
                # Append zeros for all aggregations of this missing column
                for _ in SENSOR_AGGREGATIONS_PRED:
                    aggregated_features.append(0.0)
                continue

            for agg_method in SENSOR_AGGREGATIONS_PRED:
                val = 0.0 # Default if aggregation fails or col missing
                try:
                    if agg_method == 'mean': val = sensor_df[col].mean()
                    elif agg_method == 'std': val = sensor_df[col].std(ddof=0)
                    elif agg_method == 'min': val = sensor_df[col].min()
                    elif agg_method == 'max': val = sensor_df[col].max()
                    elif agg_method == 'median': val = sensor_df[col].median()
                    
                    if pd.isna(val): # Handle NaNs from aggregation (e.g. std of 1 value)
                        val = 0.0
                except Exception as ex:
                    logger.warning(f"Could not compute {agg_method} for sensor column {col}: {ex}. Using 0.0.")
                aggregated_features.append(val)
        
        sensor_features_np = np.array(aggregated_features)
        # CRITICAL: Ensure dimension matches SENSOR_FEATURE_DIM_PRED
        if len(sensor_features_np) != SENSOR_FEATURE_DIM_PRED:
            logger.error(f"Mismatch in sensor feature dimension! Expected {SENSOR_FEATURE_DIM_PRED}, Got {len(sensor_features_np)}. Check SENSOR_COLUMNS_TO_AGGREGATE_PRED and SENSOR_AGGREGATIONS_PRED in config.")
            # Pad with zeros or truncate if necessary, or raise error. For now, pad/truncate.
            # This is risky and indicates config mismatch.
            if len(sensor_features_np) < SENSOR_FEATURE_DIM_PRED:
                padding = np.zeros(SENSOR_FEATURE_DIM_PRED - len(sensor_features_np))
                sensor_features_np = np.concatenate((sensor_features_np, padding))
            else:
                sensor_features_np = sensor_features_np[:SENSOR_FEATURE_DIM_PRED]
            logger.warning(f"Sensor features adjusted to dimension {SENSOR_FEATURE_DIM_PRED}.")

        return torch.from_numpy(sensor_features_np).float()
    except Exception as e:
        logger.error(f"Error preparing sensor features for inference: {e}", exc_info=True)
        # Fallback to zero vector on error
        return torch.zeros(SENSOR_FEATURE_DIM_PRED)