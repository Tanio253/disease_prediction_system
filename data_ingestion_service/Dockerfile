FROM python:3.10-slim

WORKDIR /service_root 

COPY ./data_ingestion_service/requirements.txt ./requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

COPY ./data_ingestion_service/app ./app 

EXPOSE 8001
# Python interpreter will be run from /service_root
# Uvicorn will look for package 'app', then module 'main', then object 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]