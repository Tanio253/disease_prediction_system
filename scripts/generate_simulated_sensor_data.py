import pandas as pd
import numpy as np
from datetime import datetime, timedelta # Not strictly needed if we just use hour offsets
import argparse
import os
import re # For cleaning patient age
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define disease categories and their potential impact on sensor data
# These are simplified and illustrative. Real-world impacts are more complex and varied.
DISEASE_IMPACTS = {
    # Disease: {sensor: (mean_delta, std_multiplier_for_delta, min_impact, max_impact)}
    # std_multiplier_for_delta allows the delta itself to be variable.
    # min_impact, max_impact clip the *delta* before adding to baseline.
    "Atelectasis":        {"hr": (5, 2, 0, 10), "resp": (2, 1, 0, 4), "spo2": (-1, 0.5, -2, 0), "temp": (0.1, 0.1, 0, 0.3)},
    "Consolidation":      {"hr": (10, 3, 5, 15), "resp": (4, 2, 2, 6), "spo2": (-2.5, 1, -4, -1), "temp": (0.6, 0.2, 0.3, 1.0)},
    "Infiltration":       {"hr": (8, 2.5, 3, 12), "resp": (3, 1.5, 1, 5), "spo2": (-2, 0.8, -3.5, -0.5), "temp": (0.4, 0.15, 0.2, 0.8)},
    "Pneumothorax":       {"hr": (15, 5, 5, 25), "resp": (5, 2, 2, 8), "spo2": (-4, 1.5, -7, -2), "temp": (0, 0.1, -0.2, 0.2), "bp_sys": (-10, 5, -20, 0)},
    "Edema":              {"hr": (6, 2, 0, 10), "resp": (3, 1, 1, 5), "spo2": (-2.5, 1, -4, -1), "temp": (0.1, 0.1, 0, 0.3), "bp_sys": (5, 3, 0, 10)},
    "Emphysema":          {"hr": (2, 1, -2, 5), "resp": (1, 0.5, 0, 3), "spo2": (-3, 1, -5, -1.5), "temp": (0, 0.1, -0.1, 0.1)}, # Chronic, less acute changes
    "Fibrosis":           {"hr": (1, 1, -2, 4), "resp": (1, 0.5, 0, 2), "spo2": (-1.5, 0.5, -3, -0.5), "temp": (0, 0.1, -0.1, 0.1)}, # Chronic
    "Effusion":           {"hr": (7, 2, 2, 12), "resp": (2.5, 1, 1, 4), "spo2": (-1.5, 0.5, -3, -0.5), "temp": (0.2, 0.1, 0, 0.5)},
    "Pneumonia":          {"hr": (12, 4, 5, 20), "resp": (5, 2, 2, 7), "spo2": (-3.5, 1.2, -6, -1.5), "temp": (0.9, 0.3, 0.5, 1.5)},
    "Pleural_Thickening": {"hr": (0, 1, -2, 2), "resp": (0.5, 0.5, 0, 1.5), "spo2": (-0.5, 0.3, -1.5, 0), "temp": (0, 0.1, -0.1, 0.1)},
    "Cardiomegaly":       {"hr": (-3, 2, -8, 2), "resp": (1, 0.5, 0, 2), "spo2": (0, 0.2, -1, 0.5), "temp": (0, 0.1, -0.1, 0.1), "bp_sys": (5, 3, 0, 10)},
    "Nodule":             {}, # Often asymptomatic sensor-wise for these vitals
    "Mass":               {}, # Often asymptomatic
    "Hernia":             {}, # Unlikely to affect these vitals directly
    "No Finding":         {}  # No specific impact
}

# Baselines (mean, standard_deviation_for_noise, realistic_min, realistic_max)
BASELINES = {
    "HeartRate_bpm":      (75, 5, 40, 180),
    "RespiratoryRate_bpm": (16, 2, 8, 35),
    "SpO2_percent":       (97, 1, 85, 100),
    "Temperature_C":      (37.0, 0.2, 35.0, 41.5),
    "BPSystolic_mmHg":    (120, 8, 70, 220),
    "BPDiastolic_mmHg":   (80, 5, 40, 130)
}

def clean_age(age_str_val):
    """Cleans patient age from various string formats to an integer."""
    if pd.isna(age_str_val):
        return 50 # Default age for missing values
    if isinstance(age_str_val, (int, float)):
        return int(age_str_val)
    
    # Attempt to extract digits using regex
    age_str = str(age_str_val)
    match = re.search(r'(\d+)', age_str)
    if match:
        return int(match.group(1))
    
    logger.warning(f"Could not parse age string: '{age_str_val}'. Defaulting to 50.")
    return 50 # Default if parsing fails

def generate_sensor_value(base_mean, base_std_noise, disease_delta_sum, realistic_min, realistic_max):
    """Generates a single sensor reading with noise and disease impact."""
    value = np.random.normal(base_mean + disease_delta_sum, base_std_noise)
    return np.clip(value, realistic_min, realistic_max)

def generate_patient_sensor_data(patient_study_info, hours_to_simulate=24, readings_per_hour=1):
    """
    Generates time-series sensor data for a single patient study.
    """
    patient_id_source = patient_study_info['Patient ID']
    image_index = patient_study_info['Image Index'] # Crucial link
    # age = clean_age(patient_study_info['Patient Age']) # Age could be used for baseline adjustment later
    # gender = patient_study_info['Patient Gender'] # Gender could be used for baseline adjustment

    finding_labels = str(patient_study_info.get('Finding Labels', 'No Finding')).split('|')
    finding_labels = [label.strip() for label in finding_labels if label.strip()]
    if not finding_labels:
        finding_labels = ["No Finding"]

    # Calculate cumulative deltas from all present diseases
    cumulative_deltas = {sensor: 0.0 for sensor in BASELINES.keys()}
    # Map BASELINES keys to DISEASE_IMPACTS keys if they differ (e.g. bp_sys vs BPSystolic_mmHg)
    sensor_map = {
        "HeartRate_bpm": "hr", "RespiratoryRate_bpm": "resp", "SpO2_percent": "spo2",
        "Temperature_C": "temp", "BPSystolic_mmHg": "bp_sys", "BPDiastolic_mmHg": "bp_dia"
    }

    for disease in finding_labels:
        if disease in DISEASE_IMPACTS:
            impacts_for_disease = DISEASE_IMPACTS[disease]
            for vital_key, impact_params in impacts_for_disease.items():
                # vital_key is like "hr", "resp", etc.
                # impact_params is (mean_delta, std_multiplier_for_delta, min_impact, max_impact)
                
                # Find the corresponding full sensor name (e.g. "HeartRate_bpm" for "hr")
                full_sensor_name = next((bs_key for bs_key, map_key in sensor_map.items() if map_key == vital_key), None)

                if full_sensor_name and impact_params:
                    mean_delta, std_delta_mult, min_d, max_d = impact_params
                    # Generate a variable delta
                    variable_delta = np.random.normal(mean_delta, np.abs(mean_delta * std_delta_mult * 0.2)) # Make std_delta relative to mean_delta
                    clipped_delta = np.clip(variable_delta, min_d, max_d)
                    cumulative_deltas[full_sensor_name] += clipped_delta


    records = []
    num_readings = hours_to_simulate * readings_per_hour

    for i in range(num_readings):
        # Simulate a timestamp (e.g., hour offset from an arbitrary point)
        # For simplicity, we can just use an reading index if exact timestamps aren't critical for feature engineering
        # If using timestamps:
        # current_time = datetime.now() - timedelta(hours=(num_readings - 1 - i) / readings_per_hour)
        # record_timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        record_timestamp = f"T-{num_readings - 1 - i}" # Simple offset, e.g., T-23, T-22, ..., T-0

        record = {
            'PatientID_Source': patient_id_source,
            'ImageIndex': image_index,
            'ReadingOffset': i # 0 for earliest, up to num_readings-1 for latest
            # 'Timestamp': record_timestamp # Optional
        }

        hr_val = generate_sensor_value(*BASELINES["HeartRate_bpm"], cumulative_deltas["HeartRate_bpm"])
        bp_sys_val = generate_sensor_value(*BASELINES["BPSystolic_mmHg"], cumulative_deltas["BPSystolic_mmHg"])
        bp_dia_val = generate_sensor_value(*BASELINES["BPDiastolic_mmHg"], cumulative_deltas["BPDiastolic_mmHg"])
        
        # Ensure BP Diastolic <= BP Systolic - some physiological constraint
        if bp_dia_val >= bp_sys_val:
            bp_dia_val = bp_sys_val - np.random.uniform(5, 20) # Make it lower
            bp_dia_val = np.clip(bp_dia_val, BASELINES["BPDiastolic_mmHg"][2], BASELINES["BPDiastolic_mmHg"][3])


        record["HeartRate_bpm"] = round(hr_val)
        record["RespiratoryRate_bpm"] = round(generate_sensor_value(*BASELINES["RespiratoryRate_bpm"], cumulative_deltas["RespiratoryRate_bpm"]))
        record["SpO2_percent"] = round(generate_sensor_value(*BASELINES["SpO2_percent"], cumulative_deltas["SpO2_percent"]), 1)
        record["Temperature_C"] = round(generate_sensor_value(*BASELINES["Temperature_C"], cumulative_deltas["Temperature_C"]), 2)
        record["BPSystolic_mmHg"] = round(bp_sys_val)
        record["BPDiastolic_mmHg"] = round(bp_dia_val)
        
        records.append(record)
    
    return records


def main(metadata_file_path, output_file_path, hours_sim, readings_per_hr):
    if not os.path.exists(metadata_file_path):
        logger.error(f"Error: Metadata file not found at {metadata_file_path}")
        return

    logger.info(f"Loading NIH metadata from: {metadata_file_path}")
    try:
        # Specify dtype for columns known to cause issues if not string initially
        # For NIH Chest X-ray data, 'Patient ID' might be numeric but best read as str if it can have leading zeros or non-numeric parts
        # 'Finding Labels', 'Patient Age', 'Patient Gender' are often strings.
        nih_df = pd.read_csv(metadata_file_path, dtype={'Patient ID': str, 'Image Index': str})
    except Exception as e:
        logger.error(f"Error reading metadata CSV: {e}", exc_info=True)
        return
        
    logger.info(f"Loaded {len(nih_df)} records from metadata.")

    required_cols = ['Patient ID', 'Image Index', 'Patient Age', 'Finding Labels'] # 'Patient Gender' is also useful
    if not all(col in nih_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in nih_df.columns]
        logger.error(f"Error: Metadata CSV is missing required columns: {', '.join(missing)}")
        logger.info(f"Available columns are: {nih_df.columns.tolist()}")
        return
        
    all_sensor_records = []
    logger.info(f"Generating sensor data for {len(nih_df)} patient studies...")

    for index, row in nih_df.iterrows():
        if (index + 1) % 500 == 0: # Log progress
            logger.info(f"Processed {index + 1}/{len(nih_df)} studies for sensor data generation...")
        
        # Clean age for each row before passing (though not directly used in this version's generation logic, good practice)
        row_copy = row.copy() # Work on a copy to avoid SettingWithCopyWarning if modifying df directly
        row_copy['Patient Age'] = clean_age(row['Patient Age'])

        patient_study_sensor_records = generate_patient_sensor_data(row_copy, hours_sim, readings_per_hr)
        all_sensor_records.extend(patient_study_sensor_records)

    if not all_sensor_records:
        logger.warning("No sensor data was generated. Check metadata, disease impacts, or generation logic.")
        return

    sensor_df = pd.DataFrame(all_sensor_records)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    try:
        sensor_df.to_csv(output_file_path, index=False)
        logger.info(f"Successfully generated and saved {len(sensor_df)} simulated sensor readings to: {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving sensor data CSV: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated time-series sensor data based on NIH Chest X-ray metadata.")
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to the input NIH metadata CSV file (e.g., your sampled_nih_metadata.csv)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the generated sensor data CSV file (e.g., data/raw/tabular/simulated_sensor_data.csv)."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours of sensor data to simulate per patient study. (Default: 24)"
    )
    parser.add_argument(
        "--readings_per_hour",
        type=int,
        default=1,
        help="Number of sensor readings to generate per hour (e.g., 1 for hourly, 4 for every 15 mins). (Default: 1)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting simulated sensor data generation process...")
    main(args.metadata_csv, args.output_csv, args.hours, args.readings_per_hour)
    logger.info("Simulated sensor data generation process finished.")

    # Example usage from terminal (from the root of disease_prediction_system):
    # python scripts/generate_simulated_sensor_data.py \
    #   --metadata_csv data/raw/tabular/sampled_nih_metadata.csv \
    #   --output_csv data/raw/tabular/simulated_sensor_data.csv \
    #   --hours 24 \
    #   --readings_per_hour 1