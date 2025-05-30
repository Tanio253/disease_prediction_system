#pagebreak()
= Deployment
The system is designed for containerized deployment using Docker. The `docker-compose.yml` file defines and orchestrates all the microservices, including the MinIO object store and PostgreSQL database.

- Each microservice has its own Dockerfile (e.g., `api_gateway/Dockerfile`, `data_ingestion_service/Dockerfile`, etc. These are defined in their respective service directories and referenced in the `docker-compose.yml`).
- Environment variables are used extensively for configuration (e.g., service URLs, MinIO credentials, database connection strings), managed through `.env` files and the `docker-compose.yml` environments section.
- Data persistence for MinIO and PostgreSQL is handled using Docker volumes (`minio_data`, `postgres_data`) as defined in `docker-compose.yml`.
- Health checks are defined in `docker-compose.yml` for MinIO and PostgreSQL to ensure they are ready before dependent services start.

To deploy the system locally:
1.  Ensure Docker and Docker Compose are installed.
2.  Create a `.env` file based on `.env.example` if provided, or ensure variables are set.
3.  Build the Docker images for each service using `docker-compose build`.
4.  Start all services using `docker-compose up -d`.
