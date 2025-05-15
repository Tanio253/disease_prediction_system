# tabular_preprocessing_service/app/minio_utils.py
from minio import Minio
from .config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_USE_SSL
import logging

logger = logging.getLogger(__name__)
minio_client_instance = None

def get_minio_client():
    global minio_client_instance
    if minio_client_instance is None:
        try:
            minio_client_instance = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_USE_SSL
            )
            logger.info(f"MinIO client initialized for tabular preprocessing: {MINIO_ENDPOINT}")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise
    return minio_client_instance

def SENSITIVE_check_and_create_bucket_util(client: Minio, bucket_name: str):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        logger.info(f"Bucket '{bucket_name}' created.")
    else:
        logger.info(f"Bucket '{bucket_name}' already exists.")