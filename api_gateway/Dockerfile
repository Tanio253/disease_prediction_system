FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

EXPOSE 8080 # External port for the gateway, defined in docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]