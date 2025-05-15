from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Dict

from . import crud, models, schemas
from .database import get_db # Removed SessionLocal, engine as get_db provides session

app = FastAPI(
    title="Patient Data Service",
    description="Manages patient and study metadata for disease prediction.",
    version="0.2.0" # Version bump
)

# --- Patient Endpoints ---
@app.post("/patients/", response_model=schemas.Patient, status_code=201, summary="Create a new patient or retrieve if exists")
def create_or_get_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    db_patient = crud.get_patient_by_source_id(db, patient_id_source=patient.patient_id_source)
    if db_patient:
        # If patient exists, update if new info provided, else just return
        # This logic can be more sophisticated based on requirements (e.g. disallow updates here)
        if (patient.age is not None and patient.age != db_patient.age) or \
           (patient.gender is not None and patient.gender != db_patient.gender):
            update_schema = schemas.PatientUpdate(age=patient.age, gender=patient.gender)
            return crud.update_patient(db, patient_id_source=patient.patient_id_source, patient_update=update_schema)
        return db_patient
    
    created_patient = crud.create_patient(db=db, patient=patient)
    if not created_patient: # Should be handled by IntegrityError in crud for duplicates
        raise HTTPException(status_code=500, detail="Patient could not be created.")
    return created_patient

@app.get("/patients/{patient_id_source}", response_model=schemas.Patient, summary="Get a patient by their source ID")
def read_patient_by_source_id(patient_id_source: str, db: Session = Depends(get_db)):
    db_patient = crud.get_patient_by_source_id(db, patient_id_source=patient_id_source)
    if db_patient is None:
        raise HTTPException(status_code=404, detail=f"Patient with source ID {patient_id_source} not found")
    return db_patient

@app.put("/patients/{patient_id_source}", response_model=schemas.Patient, summary="Update patient information by source ID")
def update_existing_patient(patient_id_source: str, patient_update: schemas.PatientUpdate, db: Session = Depends(get_db)):
    updated_patient = crud.update_patient(db, patient_id_source=patient_id_source, patient_update=patient_update)
    if updated_patient is None:
        raise HTTPException(status_code=404, detail=f"Patient with source ID {patient_id_source} not found or update failed")
    return updated_patient

# --- Study Endpoints ---
# The StudyCreate schema now includes patient_id_source
@app.post("/studies/", response_model=schemas.Study, status_code=201, summary="Create a new study or retrieve if image_index exists")
def create_or_get_study(study_data: schemas.StudyCreate, db: Session = Depends(get_db)):
    # Check if study already exists by image_index
    existing_study = crud.get_study_by_image_index(db, image_index=study_data.image_index)
    if existing_study:
        # Optionally, update the existing study if new data is provided.
        # For now, let's just return it to indicate it's already there.
        # Or raise HTTPException(status_code=409, detail=f"Study with Image Index {study_data.image_index} already exists.")
        return existing_study # Simple return for now

    # Find or create patient associated with this study
    db_patient = crud.get_patient_by_source_id(db, patient_id_source=study_data.patient_id_source)
    if not db_patient:
        # If patient doesn't exist, create a basic one from study_data if possible
        # (assuming StudyCreate might not have all patient details like age/gender,
        # so they might be None initially for the patient if created this way)
        patient_create_payload = schemas.PatientCreate(patient_id_source=study_data.patient_id_source)
        db_patient = crud.create_patient(db=db, patient=patient_create_payload)
        if not db_patient:
             raise HTTPException(status_code=500, detail=f"Failed to create associated patient {study_data.patient_id_source} for study.")
    
    new_study = crud.create_study_for_patient(db=db, study_create_data=study_data, patient_internal_id=db_patient.id)
    if new_study is existing_study: # if crud.create_study decided to return existing
        # This path might be taken if the initial get_study_by_image_index missed due to race condition,
        # and create_study_for_patient detected it via IntegrityError.
        # Return 200 OK or 201 Created based on whether it's truly new or was existing.
        # For simplicity, if it's returned, it's "done".
        return new_study 
    if not new_study:
        raise HTTPException(status_code=500, detail=f"Could not create study with Image Index {study_data.image_index}.")
    return new_study


@app.get("/studies/image_index/{image_index}", response_model=schemas.Study, summary="Get a study by its Image Index")
def read_study_by_image_index(image_index: str, db: Session = Depends(get_db)):
    db_study = crud.get_study_by_image_index(db, image_index=image_index)
    if db_study is None:
        raise HTTPException(status_code=404, detail=f"Study with Image Index {image_index} not found")
    return db_study

@app.get("/studies/db_id/{study_internal_id}", response_model=schemas.Study, summary="Get a study by its internal database ID")
def read_study_by_internal_id(study_internal_id: int, db: Session = Depends(get_db)):
    db_study = crud.get_study(db, study_id=study_internal_id)
    if db_study is None:
        raise HTTPException(status_code=404, detail=f"Study with internal ID {study_internal_id} not found")
    return db_study


@app.put("/studies/image_index/{image_index}", response_model=schemas.Study, summary="Update study information by Image Index")
def update_existing_study(image_index: str, study_update: schemas.StudyUpdate, db: Session = Depends(get_db)):
    # study_update should not contain image_index or patient_id_source
    updated_study = crud.update_study_by_image_index(db, image_index=image_index, study_update=study_update)
    if updated_study is None:
        raise HTTPException(status_code=404, detail=f"Study with Image Index {image_index} not found or update failed")
    return updated_study

@app.get("/studies/patient/{patient_id_source}", response_model=List[schemas.Study], summary="Get all studies for a patient by their source ID")
def read_studies_for_patient(patient_id_source: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    studies = crud.get_studies_for_patient_by_source_id(db, patient_id_source=patient_id_source, skip=skip, limit=limit)
    if not studies and not crud.get_patient_by_source_id(db, patient_id_source): # Check if patient even exists
        raise HTTPException(status_code=404, detail=f"Patient with source ID {patient_id_source} not found")
    return studies

@app.get("/studies/training_ready/", response_model=List[schemas.Study], summary="Get all studies ready for model training")
def read_studies_for_training(skip: int = 0, limit: int = 10000, db: Session = Depends(get_db)):
    studies = crud.get_all_studies_for_training(db, skip=skip, limit=limit)
    return studies


@app.get("/health")
async def health_check():
    # You could add a simple DB query here to check connection
    return {"status": "healthy"}