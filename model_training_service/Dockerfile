FROM python:3.10-slim

WORKDIR /app

# For PyTorch with CUDA (example for CUDA 11.8) - CHOOSE ONE or use CPU version
# Ensure your base OS image for Docker (if not this one) and host have compatible CUDA drivers.
# This section is highly dependent on your target deployment environment (CPU vs GPU)
# For CPU-only:
RUN pip config set global.index-url https://download.pytorch.org/whl/cpu
# For CUDA 11.8:
# RUN pip config set global.index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
# RUN pip config set global.index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision
RUN pip config unset global.index-url
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip config unset global.index-url # Unset after PyTorch install if you set it

COPY ./scripts /app/scripts
COPY ./app /app/app 

# Set PYTHONPATH if your imports are structured with a root package e.g. model_training_service.scripts
ENV PYTHONPATH /app

# Default command could be to run the training script,
# or to start the FastAPI server if that's the primary interface.
# For script-based:
# CMD ["python", "-m", "scripts.train_fusion_model"]
# If using FastAPI wrapper:
# EXPOSE 800X # The port your FastAPI app will run on
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "800X", "--reload"]
# For now, let's assume script-based execution as primary.
# The actual command will be in docker-compose.yml for flexibility.