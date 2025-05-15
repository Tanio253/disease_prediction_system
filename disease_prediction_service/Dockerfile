FROM python:3.10-slim

WORKDIR /service_root 

# System dependencies for Pillow/OpenCV if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./disease_prediction_service/requirements.txt ./requirements.txt 
# Ensure PyTorch is installed correctly (CPU or GPU as per your setup)
# Example: RUN pip config set global.index-url https://download.pytorch.org/whl/cu118 # For CUDA
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip config unset global.index-url # If you set it

COPY ./disease_prediction_service/app ./app 
# Copy shared utility code if it's not packaged and installed as a library
# Example: If image_processor, nih_processor, sensor_processor, and model_def
# are needed directly and are in a shared location or copied here.
# For simplicity, we will re-implement or copy the necessary processing functions
# into this service's 'app' directory if they are small, or structure them as
# a shared local library if they are extensive.
# For now, let's assume relevant processing logic is within this service's app directory.

EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]