FROM python:3.10-slim

WORKDIR /service_root 
COPY ./tabular_preprocessing_service/requirements.txt ./requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

COPY ./tabular_preprocessing_service/app ./app 

EXPOSE 8003
# Uvicorn now targets the 'app' package, then 'main' module, then 'app' instance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]