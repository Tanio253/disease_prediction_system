import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader # Ensure DataLoader is imported
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, hamming_loss
import numpy as np
import logging
import time
import os
import json
import io # Required for saving metrics json to MinIO
from minio import Minio

from .data_loader import FusionDataset, fusion_collate_fn 
from .model_def import AttentionFusionMLP 
from .utils import get_all_classes, load_processed_study_data_df, save_mlb_to_minio, load_mlb_from_minio # Assuming these exist
from . import config_training as C 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine device
device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
use_amp = True if device.type == 'cuda' else False # Enable AMP only for CUDA
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds_proba = []
    all_labels_list = [] # Renamed to avoid conflict

    for batch_idx, batch in enumerate(data_loader):
        img_features = batch['img_features'].to(device)
        nih_features = batch['nih_features'].to(device)
        sensor_features = batch['sensor_features'].to(device)
        
        img_mask = batch['img_mask'].to(device)
        nih_mask = batch['nih_mask'].to(device)
        sensor_mask = batch['sensor_mask'].to(device)
        
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(img_features, nih_features, sensor_features,
                            img_mask, nih_mask, sensor_mask) # Pass masks
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        
        all_preds_proba.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels_list.append(labels.detach().cpu().numpy())
        
        if batch_idx % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(data_loader)}, Batch Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(data_loader)
    all_preds_np = np.concatenate(all_preds_proba, axis=0)
    all_labels_np = np.concatenate(all_labels_list, axis=0)
    
    return avg_epoch_loss, all_preds_np, all_labels_np


def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_preds_proba = []
    all_labels_list = [] # Renamed

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            img_features = batch['img_features'].to(device)
            nih_features = batch['nih_features'].to(device)
            sensor_features = batch['sensor_features'].to(device)

            img_mask = batch['img_mask'].to(device) # Get masks for validation too
            nih_mask = batch['nih_mask'].to(device)
            sensor_mask = batch['sensor_mask'].to(device)
            
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(img_features, nih_features, sensor_features,
                                img_mask, nih_mask, sensor_mask) # Pass masks
                loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            all_preds_proba.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())

    avg_epoch_loss = epoch_loss / len(data_loader)
    all_preds_np = np.concatenate(all_preds_proba, axis=0)
    all_labels_np = np.concatenate(all_labels_list, axis=0)
    
    return avg_epoch_loss, all_preds_np, all_labels_np

def calculate_metrics(predictions_proba, true_labels, threshold=0.5):
    metrics = {}
    num_classes = true_labels.shape[1]
    class_names = get_all_classes() # Ensure this function is available and correct

    predictions_binary = (predictions_proba >= threshold).astype(int)

    metrics['hamming_loss'] = hamming_loss(true_labels, predictions_binary)
    metrics['f1_micro'] = f1_score(true_labels, predictions_binary, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(true_labels, predictions_binary, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(true_labels, predictions_binary, average='weighted', zero_division=0)
    metrics['f1_samples'] = f1_score(true_labels, predictions_binary, average='samples', zero_division=0)
    
    metrics['precision_micro'] = precision_score(true_labels, predictions_binary, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(true_labels, predictions_binary, average='micro', zero_division=0)

    per_class_auc = {}
    per_class_f1 = {}
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
        try:
            auc = roc_auc_score(true_labels[:, i], predictions_proba[:, i])
            per_class_auc[class_name] = auc
        except ValueError:
            per_class_auc[class_name] = np.nan
        
        f1 = f1_score(true_labels[:, i], predictions_binary[:, i], zero_division=0)
        per_class_f1[class_name] = f1

    metrics['per_class_auc'] = per_class_auc
    metrics['per_class_f1'] = per_class_f1
    
    valid_aucs = [auc for auc in per_class_auc.values() if not np.isnan(auc)]
    metrics['auc_macro'] = np.mean(valid_aucs) if valid_aucs else np.nan

    return metrics


def save_model_artifacts(model, model_name, epoch, minio_client, training_config, metrics=None):
    """Saves model state_dict, training_config.json, and optionally metrics.json to MinIO."""
    model_basename = f"{model_name}_epoch_{epoch+1}"
    
    # Save model state_dict
    model_filename_local = f"/tmp/{model_basename}.pt"
    torch.save(model.state_dict(), model_filename_local)
    model_object_name_minio = f"models/{model_name}/{model_basename}.pt" # Changed path prefix
    try:
        minio_client.fput_object(
            C.BUCKET_MODELS_STORE, model_object_name_minio, model_filename_local
        )
        logger.info(f"Model state_dict saved to MinIO: {C.BUCKET_MODELS_STORE}/{model_object_name_minio}")
    except Exception as e:
        logger.error(f"Error saving model state_dict to MinIO: {e}", exc_info=True)
    finally:
        if os.path.exists(model_filename_local):
            os.remove(model_filename_local)

    # Save training_config.json
    config_filename_local = f"/tmp/{model_name}_training_config.json" # Save one config per model_name
    config_object_name_minio = f"models/{model_name}/training_config.json"
    try:
        with open(config_filename_local, 'w') as f:
            json.dump(training_config, f, indent=4)
        minio_client.fput_object(
            C.BUCKET_MODELS_STORE, config_object_name_minio, config_filename_local
        )
        logger.info(f"Training config saved to MinIO: {C.BUCKET_MODELS_STORE}/{config_object_name_minio}")
    except Exception as e:
        logger.error(f"Error saving training_config.json to MinIO: {e}", exc_info=True)
    finally:
        if os.path.exists(config_filename_local):
            os.remove(config_filename_local)

    # Save metrics if provided
    if metrics:
        metrics_filename_local = f"/tmp/{model_basename}_metrics.json"
        metrics_object_name_minio = f"models/{model_name}/{model_basename}_metrics.json" # Changed path prefix
        try:
            with open(metrics_filename_local, 'w') as f:
                 json.dump(metrics, f, indent=4) # Save metrics locally first
            minio_client.fput_object(
                C.BUCKET_MODELS_STORE, metrics_object_name_minio, metrics_filename_local
            )
            logger.info(f"Metrics saved to MinIO: {C.BUCKET_MODELS_STORE}/{metrics_object_name_minio}")
        except Exception as e:
            logger.error(f"Error saving metrics to MinIO: {e}", exc_info=True)
        finally:
            if os.path.exists(metrics_filename_local):
                os.remove(metrics_filename_local)


def main_training_pipeline():
    logger.info(f"Starting training pipeline on device: {device}")
    start_time = time.time()

    minio_client = Minio(
        C.MINIO_ENDPOINT, access_key=C.MINIO_ACCESS_KEY, secret_key=C.MINIO_SECRET_KEY, secure=C.MINIO_USE_SSL
    )
    if not minio_client.bucket_exists(C.BUCKET_MODELS_STORE):
        minio_client.make_bucket(C.BUCKET_MODELS_STORE)
        logger.info(f"Created MinIO bucket: {C.BUCKET_MODELS_STORE}")
    if not minio_client.bucket_exists(C.MINIO_PROCESSED_BUCKET): # Ensure processed data bucket exists too
        logger.error(f"MinIO bucket for processed data {C.MINIO_PROCESSED_BUCKET} not found. Please create it and ensure data is present.")
        return

    patient_studies_df = load_processed_study_data_df(minio_client, C.MINIO_PROCESSED_BUCKET, C.PROCESSED_STUDIES_CSV)
    if patient_studies_df is None or patient_studies_df.empty:
        logger.error("Failed to load processed study data. Aborting.")
        return

    train_df = patient_studies_df.sample(frac=0.8, random_state=42)
    val_df = patient_studies_df.drop(train_df.index)
    logger.info(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    labels_list = get_all_classes() # From utils.py

    img_feature_dim = C.EXPECTED_IMG_FEATURE_DIM
    nih_feature_dim = C.EXPECTED_NIH_FEATURE_DIM
    sensor_feature_dim = C.EXPECTED_SENSOR_FEATURE_DIM

    train_dataset = FusionDataset(
        train_df, minio_client, C.MINIO_PROCESSED_BUCKET, # Use processed bucket
        img_feature_dim, nih_feature_dim, sensor_feature_dim,
        label_cols=labels_list, training_mode=True,
        mask_image_prob=C.MASK_IMAGE_PROB,
        mask_nih_prob=C.MASK_NIH_PROB,
        mask_sensor_prob=C.MASK_SENSOR_PROB,
        # Ensure path columns in config_training match CSV headers from preprocessing
        img_feat_path_col=C.IMG_FEATURES_PATH_COL, 
        nih_feat_path_col=C.NIH_FEATURES_PATH_COL,
        sensor_feat_path_col=C.SENSOR_FEATURES_PATH_COL
    )
    val_dataset = FusionDataset(
        val_df, minio_client, C.MINIO_PROCESSED_BUCKET, # Use processed bucket
        img_feature_dim, nih_feature_dim, sensor_feature_dim,
        label_cols=labels_list, training_mode=False, # No random masking for validation
        img_feat_path_col=C.IMG_FEATURES_PATH_COL,
        nih_feat_path_col=C.NIH_FEATURES_PATH_COL,
        sensor_feat_path_col=C.SENSOR_FEATURES_PATH_COL
    )

    train_loader = DataLoader(train_dataset, batch_size=C.BATCH_SIZE, shuffle=True, collate_fn=fusion_collate_fn, num_workers=C.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=C.BATCH_SIZE, shuffle=False, collate_fn=fusion_collate_fn, num_workers=C.NUM_WORKERS, pin_memory=True)


    # 3. Initialize Model, Criterion, Optimizer
    model = AttentionFusionMLP(
        img_feature_dim=img_feature_dim,
        nih_feature_dim=nih_feature_dim,
        sensor_feature_dim=sensor_feature_dim,
        num_classes=C.NUM_CLASSES,
        embed_dim=C.MODEL_EMBED_DIM,
        num_heads=C.MODEL_NUM_HEADS,
        dropout_rate=C.MODEL_DROPOUT_RATE
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    
    # Prepare training_config to save
    training_config_to_save = {
        "model_name": C.MODEL_NAME_FUSION,
        "img_feature_dim": img_feature_dim,
        "nih_feature_dim": nih_feature_dim,
        "sensor_feature_dim": sensor_feature_dim,
        "num_classes": C.NUM_CLASSES,
        "embed_dim": C.MODEL_EMBED_DIM,
        "num_heads": C.MODEL_NUM_HEADS,
        "dropout_rate": C.MODEL_DROPOUT_RATE,
        "learning_rate": C.LEARNING_RATE,
        "batch_size": C.BATCH_SIZE,
        "num_epochs": C.NUM_EPOCHS,
        "mask_image_prob": C.MASK_IMAGE_PROB,
        "mask_nih_prob": C.MASK_NIH_PROB,
        "mask_sensor_prob": C.MASK_SENSOR_PROB,
        # Add other relevant parameters from config_training that define the model/training
    }



    best_val_metric = 0.0 
    best_model_epoch = -1

    # 4. Training Loop
    for epoch in range(C.NUM_EPOCHS):
        logger.info(f"\n--- Epoch {epoch+1}/{C.NUM_EPOCHS} ---")
        
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        train_metrics = calculate_metrics(train_preds, train_labels)
        logger.info(f"Epoch {epoch+1} Training Metrics: F1-micro: {train_metrics['f1_micro']:.4f}, AUC-macro: {train_metrics.get('auc_macro', float('nan')):.4f}")

        if val_loader and len(val_loader) > 0 : # Ensure val_loader is not empty
            val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, criterion, device)
            logger.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            val_metrics = calculate_metrics(val_preds, val_labels)
            logger.info(f"Epoch {epoch+1} Validation Metrics: F1-micro: {val_metrics['f1_micro']:.4f}, AUC-macro: {val_metrics.get('auc_macro', float('nan')):.4f}")
            
            current_val_metric = val_metrics.get(C.BEST_MODEL_METRIC, 0.0) 
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_model_epoch = epoch
                logger.info(f"New best validation {C.BEST_MODEL_METRIC}: {best_val_metric:.4f} at epoch {epoch+1}. Saving model artifacts...")
                save_model_artifacts(model, C.MODEL_NAME_FUSION, epoch, minio_client, training_config_to_save, val_metrics)
        else:
            logger.info("No validation data or loader. Saving model from this epoch.")
            save_model_artifacts(model, C.MODEL_NAME_FUSION, epoch, minio_client, training_config_to_save, train_metrics)
            
    logger.info(f"Saving final training_config.json for model {C.MODEL_NAME_FUSION}")
    config_filename_local = f"/tmp/{C.MODEL_NAME_FUSION}_training_config.json"
    config_object_name_minio = f"models/{C.MODEL_NAME_FUSION}/training_config.json"
    try:
        with open(config_filename_local, 'w') as f:
            json.dump(training_config_to_save, f, indent=4)
        minio_client.fput_object(
            C.BUCKET_MODELS_STORE, config_object_name_minio, config_filename_local
        )
    finally:
        if os.path.exists(config_filename_local): os.remove(config_filename_local)


    end_time = time.time()
    logger.info(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")
    if val_loader and len(val_loader) > 0 and best_model_epoch != -1:
        logger.info(f"Best model saved from epoch {best_model_epoch+1} with validation {C.BEST_MODEL_METRIC}: {best_val_metric:.4f}")
    else:
        logger.info(f"Final model from epoch {C.NUM_EPOCHS} saved (or no validation performed).")

if __name__ == "__main__":
    main_training_pipeline()