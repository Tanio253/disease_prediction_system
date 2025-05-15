# model_training_service/scripts/utils.py
import numpy as np
import pandas as pd # Added for pd.isna check
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)

# Define the canonical list of all 15 classes (14 diseases + "No Finding")
# This order MUST BE CONSISTENT across training, label encoding, and model output interpretation.
ALL_DISEASE_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "No Finding"
] # Total 15 classes

# Initialize and fit the MultiLabelBinarizer globally
# This ensures that the transformation is consistent and uses all known classes.
try:
    mlb = MultiLabelBinarizer(classes=ALL_DISEASE_CLASSES)
    mlb.fit([ALL_DISEASE_CLASSES]) # Fit with a dummy list containing all classes to establish the order
    logger.info(f"MultiLabelBinarizer initialized and fitted with {len(mlb.classes_)} classes: {mlb.classes_}")
except Exception as e:
    logger.error(f"Failed to initialize or fit MultiLabelBinarizer: {e}", exc_info=True)
    # Fallback or raise error, as this is critical for label encoding
    raise RuntimeError("MultiLabelBinarizer initialization failed.") from e


def encode_labels(finding_labels_str: str) -> np.ndarray:
    """
    Converts a pipe-separated string of finding labels into a multi-hot binary vector.
    Uses the globally defined and fitted MultiLabelBinarizer.

    Args:
        finding_labels_str (str): Pipe-separated string of disease labels,
                                  e.g., "Atelectasis|Cardiomegaly".
                                  Can be None or empty.

    Returns:
        np.ndarray: A 1D NumPy array of shape (NUM_CLASSES,) representing the
                    multi-hot encoded labels.
    """
    if pd.isna(finding_labels_str) or not str(finding_labels_str).strip():
        # If input is NaN, None, empty, or whitespace only, default to "No Finding"
        labels_list = ["No Finding"]
    else:
        labels_list = str(finding_labels_str).split('|')
        # Optional: Sanitize labels if needed (e.g., strip whitespace from each label)
        labels_list = [label.strip() for label in labels_list if label.strip()]
        if not labels_list: # If after stripping, the list is empty
            labels_list = ["No Finding"]

    try:
        # The binarizer will ignore labels in labels_list that are not in its 'classes_' attribute.
        # This is okay if ALL_DISEASE_CLASSES is comprehensive.
        encoded_vector = mlb.transform([labels_list]) # transform expects a list of iterables
        return encoded_vector.squeeze() # Returns a 1D array, e.g., shape (15,)
    except Exception as e:
        logger.error(f"Error encoding labels for input string '{finding_labels_str}': {e}", exc_info=True)
        # Fallback to a "No Finding" vector or a zero vector of appropriate length
        # This ensures the data loader doesn't crash but might introduce incorrect labels for bad data.
        fallback_label_list = ["No Finding"]
        fallback_encoded = mlb.transform([fallback_label_list])
        return fallback_encoded.squeeze()

def get_all_classes() -> list:
    """Returns the canonical list of all disease classes."""
    return ALL_DISEASE_CLASSES

def get_num_classes() -> int:
    """Returns the total number of disease classes."""
    return len(ALL_DISEASE_CLASSES)

def get_label_encoder() -> MultiLabelBinarizer:
    """Returns the globally fitted MultiLabelBinarizer instance."""
    return mlb

if __name__ == '__main__':
    # Test the encoding functionality
    print(f"Testing label encoding. Total classes: {get_num_classes()}")
    print(f"MLB classes used for encoding: {get_label_encoder().classes_}")

    test_cases = [
        "Atelectasis|Cardiomegaly",
        "No Finding",
        "Pneumonia",
        "UnknownDisease|Edema", # "UnknownDisease" will be ignored by MLB if not in ALL_DISEASE_CLASSES
        "",          # Empty string
        None,        # None value
        "  Effusion  | Infiltration ", # Test with spaces
        "Invalid Label" # Another unknown label
    ]

    for i, label_str in enumerate(test_cases):
        encoded = encode_labels(label_str)
        print(f"Test {i+1}: Input='{label_str}'")
        print(f"  -> Encoded (sum={np.sum(encoded)}, shape={encoded.shape}): {encoded.tolist()}")
        # Optional: Decode back to see which labels were captured (for debugging)
        # print(f"  -> Decoded by MLB: {mlb.inverse_transform(encoded.reshape(1, -1))}")
        print("-" * 20)

    # Test a case where a label might be in ALL_DISEASE_CLASSES but not in the input string
    # (This is implicitly handled by the binarizer setting those positions to 0)