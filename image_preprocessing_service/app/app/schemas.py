from pydantic import BaseModel

class PreprocessImageRequest(BaseModel):
    image_index: str