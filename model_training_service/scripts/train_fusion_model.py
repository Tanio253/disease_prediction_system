import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, hamming_loss
import numpy as np
import logging
import time
import os
import json
from minio import Minio

from .data_loader import get_data_loaders
from .model_def import FusionMLP
from .utils import get_all_classes
from .config_training import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE,
    MODEL_NAME_FUSION, BUCKET_MODELS_STORE,
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_USE_SSL
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(data_loader):
        img_features = batch['image_features'].to(device)
        nih_features = batch['nih_features'].to(device)
        sensor_features = batch['sensor_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(img_features, nih_features, sensor_features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # Store predictions and labels for epoch-level metrics
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        if batch_idx % 50 == 0: # Log every 50 batches
            logger.info(f"  Batch {batch_idx+1}/{len(data_loader)}, Batch Loss: {loss.item():.4f}")


    avg_epoch_loss = epoch_loss / len(data_loader)
    
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    
    return avg_epoch_loss, all_preds_np, all_labels_np


def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            img_features = batch['image_features'].to(device)
            nih_features = batch['nih_features'].to(device)
            sensor_features = batch['sensor_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(img_features, nih_features, sensor_features)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            all_preds.append(torch.sigmoid(outputs).cpu().numpy()) # Sigmoid to get probabilities
            all_labels.append(labels.cpu().numpy())

    avg_epoch_loss = epoch_loss / len(data_loader)
    
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    
    return avg_epoch_loss, all_preds_np, all_labels_np

def calculate_metrics(predictions_proba, true_labels, threshold=0.5):
    """Calculates multi-label classification metrics."""
    metrics = {}
    num_classes = true_labels.shape[1]
    class_names = get_all_classes()

    # Convert probabilities to binary predictions based on threshold
    predictions_binary = (predictions_proba >= threshold).astype(int)

    metrics['hamming_loss'] = hamming_loss(true_labels, predictions_binary)
    metrics['f1_micro'] = f1_score(true_labels, predictions_binary, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(true_labels, predictions_binary, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(true_labels, predictions_binary, average='weighted', zero_division=0)
    metrics['f1_samples'] = f1_score(true_labels, predictions_binary, average='samples', zero_division=0)
    
    metrics['precision_micro'] = precision_score(true_labels, predictions_binary, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(true_labels, predictions_binary, average='micro', zero_division=0)

    # Per-class metrics
    per_class_auc = {}
    per_class_f1 = {}
    for i in range(num_classes):
        try: # AUC requires at least one positive instance in true labels for a class
            auc = roc_auc_score(true_labels[:, i], predictions_proba[:, i])
            per_class_auc[class_names[i]] = auc
        except ValueError:
            per_class_auc[class_names[i]] = np.nan # Or 0.5 if preferred for undefined cases
        
        f1 = f1_score(true_labels[:, i], predictions_binary[:, i], zero_division=0)
        per_class_f1[class_names[i]] = f1

    metrics['per_class_auc'] = per_class_auc
    metrics['per_class_f1'] = per_class_f1
    
    # Average AUC across classes (macro)
    valid_aucs = [auc for auc in per_class_auc.values() if not np.isnan(auc)]
    metrics['auc_macro'] = np.mean(valid_aucs) if valid_aucs else np.nan

    return metrics


def save_model_to_minio(model, model_name, epoch, minio_client, metrics=None):
    model_filename = f"{model_name}_epoch_{epoch+1}.pt"
    model_path_local = f"/tmp/{model_filename}" # Temporary local path
    
    # Save model state_dict
    torch.save(model.state_dict(), model_path_local)
    
    # Upload to MinIO
    try:
        minio_object_name = f"trained_models/{model_name}/{model_filename}"
        minio_client.fput_object(
            BUCKET_MODELS_STORE,
            minio_object_name,
            model_path_local
        )
        logger.info(f"Model saved to MinIO: {BUCKET_MODELS_STORE}/{minio_object_name}")

        # Optionally save metrics alongside the model
        if metrics:
            metrics_filename = f"{model_name}_epoch_{epoch+1}_metrics.json"
            metrics_object_name = f"trained_models/{model_name}/{metrics_filename}"
            metrics_bytes = json.dumps(metrics, indent=4).encode('utf-8')
            minio_client.put_object(
                BUCKET_MODELS_STORE,
                metrics_object_name,
                io.BytesIO(metrics_bytes),
                len(metrics_bytes),
                content_type='application/json'
            )
            logger.info(f"Metrics saved to MinIO: {BUCKET_MODELS_STORE}/{metrics_object_name}")

    except Exception as e:
        logger.error(f"Error saving model/metrics to MinIO: {e}", exc_info=True)
    finally:
        if os.path.exists(model_path_local):
            os.remove(model_path_local)


def main_training_pipeline():
    logger.info(f"Starting training pipeline on device: {DEVICE}")
    start_time = time.time()

    # 1. Get DataLoaders
    train_loader, val_loader = get_data_loaders(batch_size=BATCH_SIZE)
    if train_loader is None:
        logger.error("Could not obtain DataLoader. Aborting training.")
        return

    # 2. Initialize Model, Criterion, Optimizer
    model = FusionMLP().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() # Suitable for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    minio_client = Minio(
        MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_USE_SSL
    )
    # Ensure model bucket exists
    if not minio_client.bucket_exists(BUCKET_MODELS_STORE):
        minio_client.make_bucket(BUCKET_MODELS_STORE)
        logger.info(f"Created MinIO bucket: {BUCKET_MODELS_STORE}")


    best_val_metric = 0.0 # Example: using F1-micro or AUC_macro
    best_model_epoch = -1

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        logger.info(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
        train_metrics = calculate_metrics(train_preds, train_labels)
        logger.info(f"Epoch {epoch+1} Training Metrics: F1-micro: {train_metrics['f1_micro']:.4f}, AUC-macro: {train_metrics.get('auc_macro', float('nan')):.4f}")


        if val_loader:
            val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, criterion, DEVICE)
            logger.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            
            val_metrics = calculate_metrics(val_preds, val_labels)
            logger.info(f"Epoch {epoch+1} Validation Metrics: F1-micro: {val_metrics['f1_micro']:.4f}, AUC-macro: {val_metrics.get('auc_macro', float('nan')):.4f}")
            
            # Simple model saving strategy: save if current val_metric is better
            current_val_metric = val_metrics.get('f1_micro', 0.0) # Or use 'auc_macro'
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_model_epoch = epoch
                logger.info(f"New best validation F1-micro: {best_val_metric:.4f} at epoch {epoch+1}. Saving model...")
                save_model_to_minio(model, MODEL_NAME_FUSION, epoch, minio_client, val_metrics)
        else:
            # If no validation loader, save model periodically or at the end
            logger.info("No validation loader. Saving model based on epoch.")
            if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS: # Save every 10 epochs or last epoch
                 save_model_to_minio(model, MODEL_NAME_FUSION, epoch, minio_client, train_metrics)


    end_time = time.time()
    logger.info(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")
    if val_loader:
        logger.info(f"Best model saved from epoch {best_model_epoch+1} with validation F1-micro: {best_val_metric:.4f}")
    else:
        logger.info(f"Final model from epoch {NUM_EPOCHS} saved.")

if __name__ == "__main__":
    # This allows running the training script directly
    # Example: python -m model_training_service.scripts.train_fusion_model
    # (if model_training_service is in PYTHONPATH or you are in its parent dir)
    main_training_pipeline()