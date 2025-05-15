from sqlalchemy.orm import Session
from . import models, schemas
from sqlalchemy.exc import IntegrityError

# --- Patient CRUD (mostly unchanged) ---
def get_patient_by_source_id(db: Session, patient_id_source: str):
    return db.query(models.Patient).filter(models.Patient.patient_id_source == patient_id_source).first()

def get_patient(db: Session, patient_id: int):
    return db.query(models.Patient).filter(models.Patient.id == patient_id).first()

def get_patients(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Patient).offset(skip).limit(limit).all()

def create_patient(db: Session, patient: schemas.PatientCreate):
    db_patient = models.Patient(
        patient_id_source=patient.patient_id_source,
        age=patient.age,
        gender=patient.gender
    )
    db.add(db_patient)
    try:
        db.commit()
        db.refresh(db_patient)
    except IntegrityError:
        db.rollback()
        # Could be that patient_id_source already exists, re-fetch
        existing_patient = get_patient_by_source_id(db, patient.patient_id_source)
        if existing_patient:
            return existing_patient # Return existing if creation failed due to unique constraint
        return None # Or raise a more specific error
    return db_patient

def update_patient(db: Session, patient_id_source: str, patient_update: schemas.PatientUpdate): # Changed to update by source_id
    db_patient = get_patient_by_source_id(db, patient_id_source)
    if not db_patient:
        return None
    update_data = patient_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_patient, key, value)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

# --- Study CRUD ---
def get_study_by_image_index(db: Session, image_index: str): # Changed from study_source_id
    return db.query(models.Study).filter(models.Study.image_index == image_index).first()

def get_study(db: Session, study_id: int): # Internal DB ID
    return db.query(models.Study).filter(models.Study.id == study_id).first()

def get_studies_for_patient_by_source_id(db: Session, patient_id_source: str, skip: int = 0, limit: int = 100):
    patient = get_patient_by_source_id(db, patient_id_source)
    if not patient:
        return []
    return db.query(models.Study).filter(models.Study.patient_id == patient.id).offset(skip).limit(limit).all()

def create_study_for_patient(db: Session, study_create_data: schemas.StudyCreate, patient_internal_id: int):
    # Check if study with this image_index already exists to prevent duplicates
    existing_study = get_study_by_image_index(db, image_index=study_create_data.image_index)
    if existing_study:
        # Decide on behavior: error out, or update existing. For now, let's assume error if trying to create duplicate.
        # This could also be an update if image_index is a key for updates.
        return existing_study # Or raise IntegrityError / custom exception

    db_study_data = study_create_data.model_dump(exclude={"patient_id_source"}) # Exclude this as we use internal id
    db_study = models.Study(**db_study_data, patient_id=patient_internal_id)
    db.add(db_study)
    try:
        db.commit()
        db.refresh(db_study)
    except IntegrityError as e:
        db.rollback()
        # Check if it was a duplicate image_index error
        existing_study = get_study_by_image_index(db, image_index=study_create_data.image_index)
        if existing_study:
            return existing_study # If somehow created in a race or if check above missed it
        raise e # Re-raise other integrity errors
    return db_study


def update_study_by_image_index(db: Session, image_index: str, study_update: schemas.StudyUpdate):
    db_study = get_study_by_image_index(db, image_index)
    if not db_study:
        return None
    update_data = study_update.model_dump(exclude_unset=True) # only update fields that are set
    for key, value in update_data.items():
        setattr(db_study, key, value)
    db.add(db_study)
    db.commit()
    db.refresh(db_study)
    return db_study

def get_all_studies_for_training(db: Session, skip: int = 0, limit: int = 10000):
    # Fetch studies that have all necessary processed paths for feature fusion model training
    return db.query(models.Study).filter(
        models.Study.processed_image_features_path != None,
        models.Study.processed_nih_tabular_features_path != None,
        models.Study.processed_sensor_features_path != None,
        models.Study.finding_labels != None # Ensure we have labels
    ).offset(skip).limit(limit).all()   