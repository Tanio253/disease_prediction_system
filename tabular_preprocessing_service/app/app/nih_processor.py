# tabular_preprocessing_service/app/nih_processor.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from typing import Dict, Any, Optional
import logging
from .config import NIH_CATEGORICAL_COLS, NIH_NUMERICAL_COLS # Adjust these in config.py

logger = logging.getLogger(__name__)

# In a real scenario, encoders/scalers should be fitted on training data and saved.
# For this PoC, we'll fit them on the fly or assume they are pre-fitted and loaded.
# For simplicity here, let's assume we are processing data row by row and fitting is not the focus.
# More robust: fit scalers/encoders on a representative dataset during a 'training setup' phase
# and then load and use them here.

# Global placeholder for fitted encoder/scaler
# These would ideally be loaded from artifacts (e.g., joblib files from MinIO)
# For this example, we'll create them, but they won't be properly "fitted" across a dataset.
# This is a known simplification for the PoC.
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for easier DataFrame integration
scaler = StandardScaler()

# Example: Suppose these were the categories seen during "training" for one-hot encoding
# This is a HACK for PoC. In reality, fit the encoder on your training dataset.
EXAMPLE_CATEGORIES = {
    'Patient Gender': ['M', 'F', 'O'], # From your data
    'View Position': ['PA', 'AP', 'LL', 'RL'] # From your data
}
# Fit the encoder with example categories. THIS IS A SIMPLIFICATION.
# A more robust approach is to fit the encoder on the full training set of NIH metadata.
try:
    # Create a dummy DataFrame with all possible categories to "fit" the encoder
    dummy_data_for_encoder = {}
    for col, cats in EXAMPLE_CATEGORIES.items():
        if col in NIH_CATEGORICAL_COLS: # Only use relevant columns
             dummy_data_for_encoder[col] = cats
    if dummy_data_for_encoder: # Check if there's anything to fit
        # Ensure all NIH_CATEGORICAL_COLS are present, even if with a single dummy value
        for col in NIH_CATEGORICAL_COLS:
            if col not in dummy_data_for_encoder:
                dummy_data_for_encoder[col] = ['dummy_value'] # Add a placeholder

        dummy_df_encoder = pd.DataFrame(dummy_data_for_encoder)
        encoder.fit(dummy_df_encoder[NIH_CATEGORICAL_COLS]) # Fit on relevant columns
        logger.info(f"OneHotEncoder 'fitted' with example categories for columns: {NIH_CATEGORICAL_COLS}")
    else:
        logger.warning("No example categories found to fit OneHotEncoder. It might not work as expected.")

except Exception as e:
    logger.error(f"Error 'fitting' OneHotEncoder with example categories: {e}. Using unfitted encoder.", exc_info=True)


def process_nih_metadata(study_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Processes structured NIH metadata from a study record.
    Assumes study_data contains fields like 'age', 'gender', 'view_position'.
    """
    logger.info(f"Processing NIH metadata for study related to image_index: {study_data.get('image_index')}")

    # Construct a DataFrame from the single study's data
    # The keys here must match what patient_data_service returns for a Study,
    # and what data_ingestion_service populated.
    data = {
        'Patient Age_cleaned': [study_data.get('age')], # Assuming 'age' is already cleaned integer
        'Patient Gender': [study_data.get('gender')],
        'View Position': [study_data.get('view_position')]
        # Add other relevant fields from study_data if needed
    }
    df = pd.DataFrame(data)

    # Fill NaNs for categorical columns before encoding (e.g., with a 'missing' string)
    for col in NIH_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Missing')
        else: # Add column with 'Missing' if not present to ensure encoder consistency
            df[col] = 'Missing'


    # 1. Encode Categorical Features
    encoded_features = None
    if any(col in df.columns for col in NIH_CATEGORICAL_COLS):
        try:
            # Ensure columns used for fitting are present, even if with default/missing values
            cols_for_encoding = [col for col in NIH_CATEGORICAL_COLS if col in df.columns]
            if not cols_for_encoding:
                 logger.warning("No categorical columns found in input data for encoding.")
            else:
                encoded_array = encoder.transform(df[cols_for_encoding])
                encoded_features = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cols_for_encoding))
        except Exception as e:
            logger.error(f"Error during one-hot encoding: {e}", exc_info=True)
            # Fallback: return no categorical features or handle error differently
            return None 
    else:
        logger.info("No categorical columns specified or found for NIH metadata processing.")

    # 2. Numerical Features (e.g., Age)
    # For this PoC, we assume age is the only numerical feature.
    # If scaler was fitted: df[NIH_NUMERICAL_COLS] = scaler.transform(df[NIH_NUMERICAL_COLS])
    numerical_features_df = pd.DataFrame()
    age_col_name = 'Patient Age_cleaned' # Must match data dict key and config
    if age_col_name in df.columns:
        # Simple imputation for age (e.g., fill with median if it were a larger dataset)
        df[age_col_name] = pd.to_numeric(df[age_col_name], errors='coerce').fillna(50) # default 50 if not numeric
        numerical_features_df = df[[age_col_name]]
    else:
        logger.warning(f"Numerical column '{age_col_name}' not found in NIH data for processing.")


    # 3. Combine features
    if encoded_features is not None and not numerical_features_df.empty:
        final_features_df = pd.concat([numerical_features_df.reset_index(drop=True),
                                       encoded_features.reset_index(drop=True)], axis=1)
    elif encoded_features is not None:
        final_features_df = encoded_features
    elif not numerical_features_df.empty:
        final_features_df = numerical_features_df
    else:
        logger.warning("No features (numerical or categorical) were processed for NIH metadata.")
        return None # Or an empty array: np.array([])

    logger.info(f"NIH metadata processed. Feature shape: {final_features_df.shape}")
    return final_features_df.to_numpy().squeeze() # Squeeze if it's a single row