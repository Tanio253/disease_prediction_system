FROM python:3.10-slim

WORKDIR /service_root 

COPY ./api_gateway/requirements.txt ./requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api_gateway/app ./app 

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]