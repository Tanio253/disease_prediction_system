from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id_source = Column(String, unique=True, index=True, nullable=False) # The ID from the NIH dataset
    age = Column(Integer, nullable=True) # Store cleaned age as integer
    gender = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now()) # Add server_default for initial population

    studies = relationship("Study", back_populates="patient")

class Study(Base):
    __tablename__ = "studies"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    image_index = Column(String, unique=True, index=True, nullable=False) # Changed from study_source_id to image_index and made unique
    
    follow_up_number = Column(Integer, nullable=True)
    view_position = Column(String, nullable=True)
    study_date = Column(DateTime(timezone=True), nullable=True, server_default=func.now())

    # Image data paths
    raw_image_path = Column(String, nullable=True) # MinIO path, e.g., "patient_x/image_y.png"
    processed_image_features_path = Column(String, nullable=True) # MinIO path to extracted image features (e.g. .npy or .pt)

    # NIH metadata paths
    # We might not store the raw NIH metadata snippet per study if the main CSV is used by ingestion.
    # Instead, the patient_data_service will hold the structured info.
    # However, if a specific subset of the row IS stored, this would be its path:
    # raw_nih_metadata_row_path = Column(String, nullable=True)
    processed_nih_tabular_features_path = Column(String, nullable=True) # MinIO path to features from NIH metadata (age, gender etc.)

    # Sensor data paths
    raw_sensor_data_path = Column(String, nullable=True) # MinIO path to raw sensor data for this study
    processed_sensor_features_path = Column(String, nullable=True) # MinIO path to features from sensor data

    # Target variable
    finding_labels = Column(Text, nullable=True) # "Atelectasis|Cardiomegaly"

    # Original image technical details (can be useful for reference or specific preprocessing)
    original_image_width = Column(Integer, nullable=True)
    original_image_height = Column(Integer, nullable=True)
    original_pixel_spacing_x = Column(Float, nullable=True)
    original_pixel_spacing_y = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    patient = relationship("Patient", back_populates="studies")