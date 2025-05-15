FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app # Copy the app.py and any other files

EXPOSE 7860 # Default Gradio port
CMD ["python", "app.py"]