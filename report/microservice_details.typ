#pagebreak()
= Microservice Details
#figure(
  image("assets/core_system.png", width: 100%),
  caption: [Core System Architecture Diagram. Shows the main microservices and their interactions, including the API Gateway, Data Ingestion Service, Patient Data Service, Image Preprocessing Service, Tabular Preprocessing Service, Model Training Service, Disease Prediction Service, and Frontend Service.]
)
== API Gateway (`api_gateway`)
The API Gateway serves as the unified entry point for all incoming requests to the system. It simplifies client interaction by providing a single interface and routes requests to the appropriate downstream microservices.

- *Technology:* FastAPI (Python)
- *Responsibilities:*
    - Request routing to `data_ingestion_service`, `disease_prediction_service`, `patient_data_service`, `image_preprocessing_service`, and `tabular_preprocessing_service`.
    - Aggregating responses (if necessary, though currently it primarily forwards requests).
    - Potentially handling cross-cutting concerns like authentication, rate limiting, and logging in a more mature version.
- *Key Files:* `main.py`, `config.py`, router files in `routers/` directory.
- *Endpoints Exposed (example via gateway):*
    - `/api/v1/ingest/...` (routes to Data Ingestion Service)
    - `/api/v1/predict/...` (routes to Disease Prediction Service)
    - `/api/v1/patients/...` (routes to Patient Data Service)
    - `/api/v1/studies/...` (routes to Patient Data Service)
    - `/api/v1/preprocess/image/...` (routes to Image Preprocessing Service)
    - `/api/v1/preprocess/tabular/...` (routes to Tabular Preprocessing Service)

== Data Ingestion Service (`data_ingestion_service`)
This service is responsible for handling the initial intake of various data types. It uploads raw data to MinIO and triggers downstream preprocessing tasks.

- *Technology:* FastAPI (Python), MinIO client, httpx.
- *Responsibilities:*
    - Receiving batch data uploads containing:
        - Chest X-ray images (PNG).
        - NIH metadata CSV (`sampled_nih_metadata.csv`).
        - Simulated sensor data CSV (`simulated_sensor_data.csv`).
    - Storing raw image files in the `raw-images` MinIO bucket.
    - Storing raw NIH metadata and sensor data CSVs in their respective MinIO buckets (`raw-tabular`, `raw-sensor-data-per-study`).
    - Interacting with the `patient_data_service` to create/update patient and study records.
    - Asynchronously triggering the `image_preprocessing_service` and `tabular_preprocessing_service` for each ingested study.
- *Key Files:* `main.py`, `minio_client.py`, `patient_service_client.py`, `utils.py`.
- *Endpoints:*
    - `POST /ingest/batch/`: Accepts lists of image files, a metadata CSV file, and a sensor data CSV file.
    // Add other ingestion endpoints if they are fully implemented and distinct
    // - `POST /ingest/image/`: [PLACEHOLDER: Endpoint for single image ingestion - if implemented or planned].
    // - `POST /ingest/tabular/nih/`: [PLACEHOLDER: Endpoint for NIH tabular data ingestion - if implemented or planned].
    // - `POST /ingest/tabular/sensor/`: [PLACEHOLDER: Endpoint for sensor data ingestion - if implemented or planned].
- *Tesing Endpoints:*
    #figure(
  image("assets/minIO.png", width: 70%),
  caption: [Data Ingested successfully to MinIO. This screenshot shows the MinIO web interface with the `raw-images` and `raw-sensor-data-per-study` buckets populated with files.]
)

#figure(
  image("assets/minIO2.png", width: 70%),
  caption: [After ingestion, the `data_ingestion_service` triggers the `image_preprocessing_service` and `tabular_preprocessing_service` to process the raw data. This screenshot shows the MinIO web interface with the `processed-sensor-features` buckets populated with processed files.]
)
== Patient Data Service (`patient_data_service`)
This service acts as a central repository for patient demographics, study information (including links to raw and processed data in MinIO), and disease labels.

- *Technology:* FastAPI (Python), PostgreSQL, SQLAlchemy, Alembic (for migrations).
- *Responsibilities:*
    - CRUD operations for Patients (`Patient` model): Stores patient ID (internal and source), age, gender.
    - CRUD operations for Studies (`Study` model): Stores image index, finding labels, view position, paths to raw image data, raw tabular data, raw sensor data, and paths to processed image, tabular, and sensor features in MinIO.
    - Storing extracted disease labels associated with each study.
    - Providing query capabilities for other services to retrieve patient/study metadata.
- *Key Files:* `main.py`, `crud.py`, `models.py`, `schemas.py`, `database.py`, Alembic migration files in `alembic/versions/`.
- *Database Schema:*
    - `patients` table: `id`, `patient_id_source`, `age_years`, `gender`.
    - `studies` table: `id`, `patient_id` (FK), `image_index`, `finding_labels`, `follow_up_number`, `view_position`, `image_raw_path`, `tabular_raw_path`, `sensor_raw_path`, `image_feature_path`, `nih_tabular_feature_path`, `sensor_tabular_feature_path`.
- *Endpoints:*
    - `POST /patients/`: Create or retrieve a patient.
    - `GET /patients/{patient_id_source}`: Get patient by source ID.
    - `POST /studies/`: Create or update a study.
    - `GET /studies/{image_index}`: Get study by image index.
    - `PUT /studies/{study_id}/image-feature-path`: Update image feature path for a study.
    - `PUT /studies/{study_id}/tabular-feature-paths`: Update tabular (NIH and sensor) feature paths for a study.
    - `GET /studies/`: Get all studies with optional query parameters.

== Image Preprocessing Service (`image_preprocessing_service`)
This service is dedicated to processing raw medical images (chest X-rays). It extracts features using a pre-trained Convolutional Neural Network (CNN) and stores them.

- *Technology:* FastAPI (Python), PyTorch, torchvision, MinIO client, Pillow.
- *Responsibilities:*
    - Retrieving raw images from the `raw-images` MinIO bucket based on a study ID.
    - Preprocessing images: resizing, normalization (as per `config.py`).
    - Extracting image features using a pre-trained model (e.g., ResNet50, as indicated in `image_processor.py`). The final layer of the CNN is typically used as the feature vector.
    - Storing the extracted image features (as PyTorch tensors) in the `processed-image-features` MinIO bucket.
    - Updating the corresponding study record in `patient_data_service` with the path to the processed image features.
- *Key Files:* `main.py`, `image_processor.py`, `minio_utils.py`, `patient_service_client.py`.
- *Endpoints:*
    - `POST /preprocess/`: Accepts a `study_id` to trigger preprocessing for the associated image.

== Tabular Preprocessing Service (`tabular_preprocessing_service`)
This service handles the preprocessing of two types of tabular data: NIH patient/study metadata and simulated sensor data.

- *Technology:* FastAPI (Python), Pandas, scikit-learn, NumPy, MinIO client.
- *Responsibilities:*
    - *NIH Metadata Processing (`nih_processor.py`):*
        - Retrieving raw NIH metadata CSV from MinIO.
        - Cleaning data (e.g., parsing age from string like "060Y").
        - Performing one-hot encoding for categorical features (e.g., `Patient Gender`, `View Position`).
        - Scaling numerical features.
        - Storing the processed NIH tabular features (as NumPy arrays or pickled DataFrames/encoders) in the `processed-tabular-nih-features` MinIO bucket.
    - *Sensor Data Processing (`sensor_processor.py`):*
        - Retrieving raw sensor data CSV (per study) from MinIO.
        - Aggregating time-series sensor data into summary statistics (mean, std, min, max, etc.) for relevant columns (e.g., `HeartRate_bpm`, `RespiratoryRate_bpm`).
        - Storing the processed sensor features in the `processed-sensor-features` MinIO bucket.
    - Updating the corresponding study record in `patient_data_service` with paths to these processed tabular features.
- *Key Files:* `main.py`, `nih_processor.py`, `sensor_processor.py`, `minio_utils.py`, `patient_service_client.py`.
- *Endpoints:*
    - `POST /preprocess/nih-metadata/`: Accepts `study_id` and `raw_file_path` to trigger NIH metadata preprocessing.
    - `POST /preprocess/sensor-data/`: Accepts `study_id` and `raw_file_path` to trigger sensor data preprocessing.

== Model Training Service (`model_training_service`)
This service is responsible for training the multi-modal fusion model that combines image, NIH tabular, and sensor features for disease prediction.

- *Technology:* FastAPI (Python), PyTorch, Pandas, NumPy, scikit-learn, MinIO client, subprocess.
- *Responsibilities:*
    - Exposing an endpoint to initiate a model training job.
    - Training jobs run as background tasks (`training_manager.py`).
    - The actual training logic is encapsulated in scripts (`scripts/train_fusion_model.py`):
        - Loading processed features (image, NIH tabular, sensor) and labels from MinIO/Patient Data Service via `data_loader.py`.
        - Defining the fusion model architecture (e.g., `AttentionFusionMLP` from `model_def.py`).
        - Executing the training loop, including loss calculation (e.g., BCEWithLogitsLoss for multi-label classification) and optimization.
        - Evaluating the model using metrics like ROC AUC, F1-score, Precision, Recall, Hamming Loss.
        - Saving the trained model components (fusion model state dict, image feature extractor state dict if fine-tuned, label binarizer, encoders/scalers) to the `models-store` MinIO bucket.
- *Key Files:* `main.py`, `training_manager.py`, `scripts/train_fusion_model.py`, `scripts/data_loader.py`, `scripts/model_def.py`, `scripts/config_training.py`.
- *Endpoints:*
    - `POST /train/`: Accepts a `TrainingJobRequest` to start a new training job. Returns a `TrainingJobResponse` with a job ID.
    - `GET /train/status/{job_id}`: Checks the status of a training job.

== Disease Prediction Service (`disease_prediction_service`)
This service is responsible for loading the trained multi-modal fusion model and making disease predictions based on provided patient data.

- *Technology:* FastAPI (Python), PyTorch, MinIO client, scikit-learn, Pandas, NumPy.
- *Responsibilities:*
    - Loading the trained fusion model (`AttentionFusionMLP`) and associated components (image feature extractor, encoders, scalers, label binarizer) from the `models-store` MinIO bucket during startup (`models_loader.py`).
    - Accepting input data for prediction:
        - Raw image file (for on-the-fly feature extraction).
        - NIH tabular data (as a JSON string or dictionary).
        - Sensor data (as a CSV string or structured format).
    - Preparing input data for the model using logic from `feature_preparation.py`, which mirrors the preprocessing steps from training. This includes:
        - Image feature extraction.
        - NIH tabular data encoding and scaling.
        - Sensor data aggregation and scaling.
    - Performing inference using the fusion model.
    - Returning the predicted disease probabilities for each class.
- *Key Files:* `main.py`, `models_loader.py`, `model_def.py`, `feature_preparation.py`, `config.py`.
- *Endpoints:*
    - `POST /predict/`: Accepts an image file, NIH tabular data (JSON string), and sensor data (CSV string) to make a prediction.

== Frontend Service (`frontend_service`)
Provides a web-based user interface for interacting with the disease prediction system.

- *Technology:* Gradio (Python), httpx (for API calls).
- *Responsibilities:*
    - Offering input fields for users to upload:
        - A chest X-ray image.
        - NIH tabular data (e.g., patient age, gender, view position - simplified for UI).
        - Sensor data (e.g., average heart rate, respiratory rate - simplified for UI).
    - Calling the `disease_prediction_service` (via the API Gateway) with the provided data.
    - Displaying the returned disease predictions (probabilities and predicted labels).
- *Key Files:* `app.py`.
- *Interface Components (based on typical Gradio usage):*
    - Image upload component.
    - Text/Number inputs for tabular and sensor data.
    - Output component to display prediction results (e.g., labels with probabilities).
