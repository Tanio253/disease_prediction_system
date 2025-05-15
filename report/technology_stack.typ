#pagebreak()
= Technology Stack Summary
The system leverages a modern, scalable technology stack:
- *Backend Framework:* FastAPI (Python) for all microservices requiring HTTP APIs.
- *Machine Learning Framework:* PyTorch for model definition, training, and inference. Torchvision for pre-trained image models and image transformations.
- *Data Handling & Manipulation:* Pandas, NumPy.
- *Data Preprocessing (ML):* Scikit-learn (for OneHotEncoder, StandardScaler, MultiLabelBinarizer).
- *Frontend UI:* Gradio (Python) for rapid web UI development.
- *Database (Structured Data):* PostgreSQL, accessed via SQLAlchemy ORM with Alembic for schema migrations.
- *Object Storage (Unstructured Data & Models):* MinIO.
- *Containerization & Orchestration:* Docker and Docker Compose.
- *API Gateway:* FastAPI (could be replaced by dedicated gateway solutions like Kong or Traefik in a production environment).
- *Asynchronous Task Handling:* FastAPI's `BackgroundTasks` for triggering preprocessing and model training.
- *HTTP Client:* `httpx` for inter-service communication.
- *Configuration Management:* `python-dotenv` for managing environment variables.
- *Logging:* Python's built-in `logging` module.

This stack was chosen to meet the project requirements, emphasizing Python-based solutions, robust API capabilities, efficient ML model handling, and standard operational practices with containerization.