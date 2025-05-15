from minio import Minio
from .config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_USE_SSL
import logging

logger = logging.getLogger(__name__)

minio_client = None

def get_minio_client():
    global minio_client
    if minio_client is None:
        try:
            minio_client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_USE_SSL
            )
            logger.info(f"MinIO client initialized for endpoint: {MINIO_ENDPOINT}")
            # You might want to check if buckets exist here or create them,
            # but docker-compose already handles default bucket creation.
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise
    return minio_client

def SENSITIVE_check_and_create_bucket(client: Minio, bucket_name: str):
    """Helper to check if bucket exists and create if not."""
    try:
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            logger.info(f"Bucket '{bucket_name}' created.")
        else:
            logger.info(f"Bucket '{bucket_name}' already exists.")
    except Exception as e:
        logger.error(f"Error with bucket '{bucket_name}': {e}")
        raise