from pydantic import BaseModel, Field
from typing import Optional, Dict

class TrainingJobRequest(BaseModel):
    model_version_tag: Optional[str] = Field(None, description="A tag for the model version, e.g., 'v1.0-exp-featureX'")
    hyperparameters: Optional[Dict] = Field(None, description="Override default hyperparameters")
    # Add other relevant parameters, e.g., data subset filters

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    model_output_path: Optional[str] = None