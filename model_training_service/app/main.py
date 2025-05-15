from fastapi import FastAPI, BackgroundTasks, HTTPException
from .schemas import TrainingJobRequest, TrainingJobResponse
from .training_manager import start_training_job_background, training_jobs # Assuming training_jobs is accessible
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="Model Training Service",
    description="Manages and executes model training pipelines.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Model Training Service (API Wrapper) starting up...")
    # Any API specific initializations

@app.post("/train/fusion_model/", response_model=TrainingJobResponse, status_code=202)
async def trigger_fusion_model_training(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks
):
    """
    Triggers the fusion model training pipeline asynchronously.
    """
    logger.info(f"Received training request: {request.model_dump()}")
    
    # Convert request to params dict if needed for training_manager
    params_for_job = request.model_dump()

    job_id = start_training_job_background(background_tasks, params_for_job)
    
    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        message="Fusion model training job has been queued."
    )

@app.get("/train/status/{job_id}", response_model=TrainingJobResponse)
async def get_training_job_status(job_id: str):
    """
    Retrieves the status of a specific training job.
    """
    job_info = training_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Training job not found.")
    
    # Potentially enrich with more details if available
    return TrainingJobResponse(job_id=job_id, **job_info)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Model Training Service API"}

# To run this FastAPI app, your docker-compose command would change to:
# command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"] # Assuming port 8005
# And expose the port in docker-compose.yml