# tabular_preprocessing_service/app/sensor_processor.py
import pandas as pd
import numpy as np
import io
import logging
from .config import SENSOR_COLUMNS_TO_AGGREGATE, SENSOR_AGGREGATIONS # Ensure these are correctly defined in config.py

logger = logging.getLogger(__name__)

def process_sensor_data(sensor_data_csv_bytes: bytes) -> np.ndarray:
    """
    Processes raw sensor data CSV (time-series) to extract summary features.

    Args:
        sensor_data_csv_bytes (bytes): The byte content of the CSV file containing sensor data.
                                       Expected columns are defined in SENSOR_COLUMNS_TO_AGGREGATE.

    Returns:
        np.ndarray: A 1D NumPy array of aggregated features. Returns an empty array
                    if processing fails or no features can be generated.
    """
    try:
        sensor_df = pd.read_csv(io.BytesIO(sensor_data_csv_bytes))
        if sensor_df.empty:
            logger.warning("Sensor data CSV is empty. No features to extract.")
            return np.array([]) # Return empty array if no data
        logger.info(f"Loaded sensor data for processing. Shape: {sensor_df.shape}, Columns: {sensor_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to read sensor data CSV from bytes: {e}", exc_info=True)
        return np.array([]) # Return empty array on error

    aggregated_features = []
    feature_names_debug = [] # For logging the feature order

    # Ensure SENSOR_COLUMNS_TO_AGGREGATE are present in sensor_df, process only available ones
    for col_to_agg in SENSOR_COLUMNS_TO_AGGREGATE:
        if col_to_agg not in sensor_df.columns:
            logger.warning(f"Sensor column '{col_to_agg}' not found in provided CSV. "
                           f"Aggregations for this column will be zero/default.")
            # Append default values (e.g., 0.0) for all aggregations of this missing column
            # to maintain consistent feature vector length
            for agg_method in SENSOR_AGGREGATIONS:
                aggregated_features.append(0.0)
                feature_names_debug.append(f"{col_to_agg}_{agg_method}_missing")
            continue # Move to the next column in SENSOR_COLUMNS_TO_AGGREGATE

        # Column exists, proceed with aggregations
        for agg_method in SENSOR_AGGREGATIONS:
            val = 0.0  # Default value if aggregation fails for any reason
            try:
                if sensor_df[col_to_agg].empty: # Handle empty series for a column
                    logger.warning(f"Sensor column '{col_to_agg}' is empty. Using 0.0 for {agg_method}.")
                elif agg_method == 'mean':
                    val = sensor_df[col_to_agg].mean()
                elif agg_method == 'std':
                    val = sensor_df[col_to_agg].std(ddof=0) # Use ddof=0 for consistency if needed
                elif agg_method == 'min':
                    val = sensor_df[col_to_agg].min()
                elif agg_method == 'max':
                    val = sensor_df[col_to_agg].max()
                elif agg_method == 'median':
                    val = sensor_df[col_to_agg].median()
                else:
                    logger.warning(f"Unknown aggregation method '{agg_method}' for column '{col_to_agg}'. Skipping.")
                    continue # Skip to next aggregation method

                # Handle NaN results from aggregations (e.g., std of a single value, or mean of all NaNs)
                if pd.isna(val):
                    logger.warning(f"Aggregation '{agg_method}' for column '{col_to_agg}' resulted in NaN. Using 0.0.")
                    val = 0.0
                
            except Exception as ex:
                logger.error(f"Error computing {agg_method} for sensor column {col_to_agg}: {ex}. Using 0.0.", exc_info=True)
                # val remains 0.0 as set initially

            aggregated_features.append(val)
            feature_names_debug.append(f"{col_to_agg}_{agg_method}")

    if not aggregated_features:
        logger.warning("No sensor features were generated despite processing columns.")
        return np.array([])

    final_features_array = np.array(aggregated_features, dtype=float)
    logger.info(f"Sensor data processed. Extracted {len(final_features_array)} features.")
    logger.debug(f"Generated sensor feature names (order is important): {feature_names_debug}")
    
    # CRITICAL: The length of this array must match SENSOR_FEATURE_DIM in training/prediction configs.
    # This should inherently be len(SENSOR_COLUMNS_TO_AGGREGATE) * len(SENSOR_AGGREGATIONS)
    # if all columns are present and all aggregations are performed.
    # The logic above already pads for missing columns to maintain length.

    return final_features_array

if __name__ == '__main__':
    # Example for local testing
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing sensor_processor.py...")

    # Create dummy config values if config.py is not directly runnable/importable here
    if 'SENSOR_COLUMNS_TO_AGGREGATE' not in globals():
        SENSOR_COLUMNS_TO_AGGREGATE = ['HeartRate_bpm', 'SpO2_percent', 'NonExistentColumn']
        SENSOR_AGGREGATIONS = ['mean', 'std', 'min', 'max']
        logger.info("Using dummy config for SENSOR_COLUMNS_TO_AGGREGATE and SENSOR_AGGREGATIONS for test.")


    # Test case 1: Valid data
    sample_csv_data1 = """Timestamp,HeartRate_bpm,SpO2_percent
2023-01-01 00:00:00,70,98
2023-01-01 00:01:00,72,97
2023-01-01 00:02:00,71,99
"""
    features1 = process_sensor_data(sample_csv_data1.encode('utf-8'))
    print(f"Test Case 1 - Features (shape {features1.shape}):\n{features1}")

    # Test case 2: Empty data
    sample_csv_data2 = "" # Empty file content
    features2 = process_sensor_data(sample_csv_data2.encode('utf-8'))
    print(f"\nTest Case 2 - Empty CSV - Features (shape {features2.shape}):\n{features2}")
    
    # Test case 3: CSV with only headers
    sample_csv_data3 = "Timestamp,HeartRate_bpm,SpO2_percent\n"
    features3 = process_sensor_data(sample_csv_data3.encode('utf-8'))
    print(f"\nTest Case 3 - Headers Only CSV - Features (shape {features3.shape}):\n{features3}")

    # Test case 4: Data with some NaNs within a column
    sample_csv_data4 = """Timestamp,HeartRate_bpm,SpO2_percent
2023-01-01 00:00:00,70,98
2023-01-01 00:01:00,,97
2023-01-01 00:02:00,71,
"""
    features4 = process_sensor_data(sample_csv_data4.encode('utf-8'))
    print(f"\nTest Case 4 - Data with NaNs - Features (shape {features4.shape}):\n{features4}")

    # Test case 5: Data with a column completely missing (as per SENSOR_COLUMNS_TO_AGGREGATE)
    # Assuming SENSOR_COLUMNS_TO_AGGREGATE includes 'NonExistentColumn'
    sample_csv_data5 = """Timestamp,HeartRate_bpm,SpO2_percent
2023-01-01 00:00:00,80,95
2023-01-01 00:01:00,82,96
"""
    features5 = process_sensor_data(sample_csv_data5.encode('utf-8'))
    print(f"\nTest Case 5 - Missing configured column ('NonExistentColumn') - Features (shape {features5.shape}):\n{features5}")
    # Expected length: (2 actual cols * 4 aggs) + (1 missing col * 4 aggs_filled_with_zero) = 8 + 4 = 12 (if SENSOR_COLUMNS_TO_AGGREGATE has 3 items & 4 aggs)
    expected_len = len(SENSOR_COLUMNS_TO_AGGREGATE) * len(SENSOR_AGGREGATIONS)
    print(f"Expected feature length based on config: {expected_len}")