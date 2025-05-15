FROM python:3.10-slim

WORKDIR /app # This is key!

# Install poetry (optional, but good for dependency management)
# RUN pip install poetry
# COPY poetry.lock pyproject.toml /app/
# RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

# Or using requirements.txt
COPY requirements.txt .
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Install Alembic globally or ensure it's in requirements.txt
# RUN pip install alembic psycopg2-binary # Already in your requirements.txt

COPY . . 

# The command to run migrations and then the app will be in docker-compose.yml
# EXPOSE 8000