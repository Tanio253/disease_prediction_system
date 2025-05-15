from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class StudyBase(BaseModel):
    image_index: str = Field(..., description="Unique identifier for the image/study, from NIH 'Image Index'")
    follow_up_number: Optional[int] = None
    view_position: Optional[str] = None
    finding_labels: Optional[str] = None

    raw_image_path: Optional[str] = None
    processed_image_features_path: Optional[str] = None # Updated field name

    # raw_nih_metadata_row_path: Optional[str] = None # If storing snippets
    processed_nih_tabular_features_path: Optional[str] = None # Updated field name

    raw_sensor_data_path: Optional[str] = None
    processed_sensor_features_path: Optional[str] = None # Updated field name

    original_image_width: Optional[int] = None
    original_image_height: Optional[int] = None
    original_pixel_spacing_x: Optional[float] = None
    original_pixel_spacing_y: Optional[float] = None

class StudyCreate(StudyBase):
    patient_id_source: str # NIH Patient ID, used to find/create patient

class StudyUpdate(BaseModel): # For PATCH-like updates
    follow_up_number: Optional[int] = None
    view_position: Optional[str] = None
    finding_labels: Optional[str] = None
    raw_image_path: Optional[str] = None
    processed_image_features_path: Optional[str] = None
    # raw_nih_metadata_row_path: Optional[str] = None
    processed_nih_tabular_features_path: Optional[str] = None
    raw_sensor_data_path: Optional[str] = None
    processed_sensor_features_path: Optional[str] = None
    original_image_width: Optional[int] = None
    original_image_height: Optional[int] = None
    original_pixel_spacing_x: Optional[float] = None
    original_pixel_spacing_y: Optional[float] = None
    # No patient_id_source or image_index here, as those identify the record

class Study(StudyBase):
    id: int # Internal DB study ID
    patient_id: int # Internal DB patient ID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True # content_type renamed to from_attributes in Pydantic v2
        # from_attributes = True # For Pydantic V2

class PatientBase(BaseModel):
    patient_id_source: str # NIH Patient ID
    age: Optional[int] = None
    gender: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None

class Patient(PatientBase):
    id: int # Internal DB ID
    created_at: datetime
    updated_at: Optional[datetime] = None
    studies: List[Study] = []

    class Config:
        orm_mode = True
        # from_attributes = True # For Pydantic V2

# For returning internal study ID after creation or for specific lookup
class StudyLookupResponse(BaseModel):
    id: int
    image_index: str
    patient_id: int

    class Config:
        orm_mode = True
        # from_attributes = True