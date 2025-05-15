FROM python:3.10-slim

WORKDIR /service_root 
# Install system dependencies that might be needed by OpenCV or Pillow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./image_preprocessing_service/requirements.txt ./requirements.txt 
# Consider torch index URL if specific versions are needed or for faster downloads
# RUN pip config set global.index-url https://download.pytorch.org/whl/cu118 # Example for CUDA 11.8
RUN pip install --no-cache-dir -r requirements.txt

COPY ./image_preprocessing_service/app ./app 

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]