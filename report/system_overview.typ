#pagebreak()
= System Overview and Architecture
The disease prediction system is designed as a distributed collection of microservices that work in concert to deliver its functionalities. This architecture was chosen for its scalability, resilience, and ease of independent development and deployment of components.

The core components of the system include:
- *API Gateway:* Single entry point for all client requests, routing them to appropriate backend services.
- *Data Ingestion Service:* Responsible for receiving raw data (images, tabular CSVs, sensor readings) and initiating their processing.
- *Patient Data Service:* Manages patient demographic information, study metadata, and paths to processed features. It acts as the central metadata repository.
- *Image Preprocessing Service:* Processes raw medical images, extracts relevant features using a pre-trained deep learning model, and stores these features.
- *Tabular Preprocessing Service:* Processes raw tabular data (both NIH metadata and sensor data), performs cleaning, transformation, and feature engineering, and stores the processed features.
- *Model Training Service:* Orchestrates the training of the core multi-modal fusion model using the processed features from various sources.
- *Disease Prediction Service:* Serves the trained fusion model to make predictions based on input patient data (image features, tabular features, sensor features).
- *Frontend Service:* Provides a user-friendly web interface (built with Gradio) for users to input data and receive disease predictions.
- *Data Stores:*
    - *PostgreSQL:* Used by the Patient Data Service for storing structured metadata about patients and studies.
    - *MinIO:* Used as an object store for raw uploaded data (images, CSVs), processed features, and trained machine learning models.

// Remember to create this diagram and place it in the images folder
// #figure(
//   image("images/system_architecture_diagram.png", width: 90%),
//   caption: [High-Level System Architecture Diagram. Shows microservices, API Gateway, Frontend, MinIO, and PostgreSQL, with primary data/request flows.]
// )
// [PLACEHOLDER: High-Level System Architecture Diagram. This diagram should show all microservices, the API Gateway, Frontend, MinIO, and PostgreSQL, with arrows indicating primary data/request flows. For example, Frontend -> API Gateway -> Disease Prediction Service -> Model in MinIO & Features from MinIO (via Patient Data Service lookup).]

The services are containerized using Docker and managed via Docker Compose for local development and deployment orchestration.