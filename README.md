# AI-Powered Multi-Modal Disease Prediction System

This project implements an advanced AI system for predicting diseases by leveraging comprehensive health data, including medical imagery (Chest X-rays), tabular patient metadata, and simulated physiological sensor data. It uses a multi-modal fusion model to predict the likelihood of 14 common thoracic diseases.

## Key Features

* **Multi-Modal Data Fusion:** Integrates image, tabular, and sensor data for robust predictions.
* **Microservice Architecture:** Scalable and maintainable system with independent services.
* **End-to-End ML Pipeline:** Covers data ingestion, automated preprocessing, model training, and inference.
* **RESTful APIs & Interactive UI:** API-driven communication and a Gradio frontend for ease of use.
* **Persistent Storage:** Utilizes PostgreSQL for metadata and MinIO for data objects and models.
* **Containerized Deployment:** Employs Docker and Docker Compose for streamlined setup and operation.

## Technology Stack

* **Machine Learning:** PyTorch
* **Backend:** Python, FastAPI
* **Frontend:** Gradio (Python)
* **Database:** PostgreSQL
* **Object Storage:** MinIO
* **Operations:** Docker, Docker Compose
* **Data Handling:** Pandas, NumPy, Scikit-learn

## Architecture Overview

The system follows a microservice architecture. An API Gateway serves as the single entry point, routing requests to specialized backend services responsible for data ingestion, preprocessing (image, tabular, sensor), patient data management, model training, and disease prediction. A Gradio frontend provides an interactive user interface.

### Services
* **API Gateway:** Routes external requests to appropriate services.
* **Data Ingestion Service:** Handles intake and initial storage of raw multi-modal data. Triggers preprocessing.
* **Patient Data Service:** Manages patient and study metadata in PostgreSQL.
* **Image Preprocessing Service:** Processes raw images into features.
* **Tabular Preprocessing Service:** Processes NIH metadata and sensor data into features.
* **Model Training Service:** Trains the multi-modal fusion model.
* **Disease Prediction Service:** Performs inference using the trained model.
* **Frontend Service:** Provides a Gradio web UI for interaction.

## Data Flow Summary

1.  **Ingestion:** Raw data (images, tabular NIH data, sensor CSVs) is ingested via the API. Data is stored in MinIO, and metadata in PostgreSQL.
2.  **Preprocessing:** Automated background tasks process raw data into feature sets (image features, tabular features, sensor features), stored in MinIO.
3.  **Training:** The Model Training Service uses these features to train a multi-modal `AttentionFusionMLP`. Trained models are stored in MinIO.
4.  **Prediction:** The Disease Prediction Service loads the model and features to provide disease likelihoods, accessible via API or the Gradio UI.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tanio253/disease_prediction_system.git](https://github.com/tanio253/disease_prediction_system.git)
    cd disease_prediction_system
    ```

2.  **Configure Environment (if needed):**
    * Ensure `.env` files are present in service directories if they differ from `docker-compose.yml` defaults (most settings are pre-configured for local Docker).

3.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    This will build images, start all services, initialize MinIO buckets, and run database migrations.

## Usage

1.  **Data Ingestion (Batch Script):**
    * Place data in `./data/raw/images/` and `./data/raw/tabular/`.
    * Run the ingestion script:
        ```bash
        python scripts/run_batch_ingestion.py
        ```
    This populates the system and triggers preprocessing.

2.  **Model Training:**
    * Initiate training via the Model Training Service API endpoint (e.g., `POST /api/v1/training/start_training` via the API Gateway at `http://localhost:8080`).
    * Example:
        ```bash
        curl -X POST http://localhost:8080/api/v1/training/start_training -H "Content-Type: application/json" -d '{}'
        ```
       (Payload details depend on the `model_training_service` API; an empty JSON might trigger default training parameters).

3.  **Disease Prediction:**
    * **Frontend UI:** Access at `http://localhost:7860` (or mapped port). Upload data and get predictions.
    * **API:** Use endpoints on the API Gateway (e.g., `/api/v1/predict/predict_study/` or `/api/v1/predict/predict_modalities/`) to get predictions for existing or newly uploaded data.
        ```bash
        # Example: Predict for an already ingested and processed study
        curl -X POST http://localhost:8080/api/v1/predict/predict_study/ \
             -H "Content-Type: application/json" \
             -d '{"image_index": "00000013_005.png"}'
        ```

Refer to individual service API documentation (within their `main.py` or Swagger UI if enabled) for detailed request/response formats.
