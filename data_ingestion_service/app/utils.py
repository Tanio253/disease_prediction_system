import re
import pandas as pd

def clean_age_from_str(age_str: str) -> int:
    if pd.isna(age_str) or not isinstance(age_str, str):
        return 50 # Default age or handle as error
    match = re.search(r'(\d+)', age_str)
    if match:
        return int(match.group(1))
    return 50 # Default if parsing fails

def get_image_index_from_filename(filename: str) -> str:
    # Assuming filename is like "00000001_000.png"
    # This might need to be more robust depending on actual filenames
    return filename # If filename is directly the Image Index