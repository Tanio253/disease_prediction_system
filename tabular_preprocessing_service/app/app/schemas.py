# tabular_preprocessing_service/app/schemas.py
from pydantic import BaseModel

class TabularPreprocessRequest(BaseModel):
    image_index: str