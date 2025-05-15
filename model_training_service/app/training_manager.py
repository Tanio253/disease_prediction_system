import asyncio
import uuid
from fastapi import BackgroundTasks
import logging
import subprocess # To call the script
import sys

# Assuming your scripts are importable or callable via subprocess
# from ..scripts.train_fusion_model import main_training_pipeline # If scripts are structured as a package

logger = logging.getLogger(__name__)

# In-memory store for job statuses (for PoC, use DB/Redis for production)
training_jobs = {}

async def run_training_script_async(job_id: str, training_params: dict):
    training_jobs[job_id]["status"] = "running"
    logger.info(f"Starting training job {job_id} with params: {training_params}")
    
    try:
        # Option 1: If main_training_pipeline is importable and can accept params
        # main_training_pipeline(**training_params) # You'd need to modify script to accept params

        # Option 2: Run script as a subprocess (simpler for existing scripts)
        # Construct command. Ensure script path is correct relative to Docker WORKDIR /app
        # Example assumes train_fusion_model.py can be run with `python -m scripts.train_fusion_model`
        # You might need to pass hyperparameters via command-line arguments or env vars to the script
        cmd = [sys.executable, "-m", "scripts.train_fusion_model"] # sys.executable ensures using python in current env
        
        # Example of passing params (modify script to parse these or use env vars)
        # for key, value in training_params.items():
        #    cmd.append(f"--{key.replace('_','-')}") # e.g. num_epochs -> --num-epochs
        #    cmd.append(str(value))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["message"] = "Training completed successfully."
            # Parse stdout/stderr for model path or metrics if script prints them in a structured way
            logger.info(f"Training job {job_id} completed. STDOUT: {stdout.decode()}")
            if stderr:
                 logger.warning(f"Training job {job_id} STDERR: {stderr.decode()}")
        else:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["message"] = f"Training failed. Return code: {process.returncode}"
            logger.error(f"Training job {job_id} failed. STDERR: {stderr.decode()}, STDOUT: {stdout.decode()}")

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = f"Exception during training: {str(e)}"
        logger.error(f"Exception in training job {job_id}: {e}", exc_info=True)


def start_training_job_background(background_tasks: BackgroundTasks, params: dict) -> str:
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {"status": "queued", "message": "Training job queued."}
    
    # Extract relevant params for the script or main_training_pipeline function
    script_params = params.get("hyperparameters", {}) # Example
    
    background_tasks.add_task(run_training_script_async, job_id, script_params)
    logger.info(f"Queued training job {job_id}")
    return job_id