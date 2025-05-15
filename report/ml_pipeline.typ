#pagebreak()
= Machine Learning Pipeline

== Feature Preparation (for Inference)
The `disease_prediction_service` prepares features for inference similarly to the training pipeline (`feature_preparation.py`):
- *Image Features:* Raw image is preprocessed (resized, normalized) and fed through the loaded image feature extractor model.
- *NIH Tabular Features:* Input JSON data is converted to a DataFrame, categorical columns are one-hot encoded using the loaded encoder, and numerical columns are scaled using the loaded scaler. Missing values are handled (e.g., by imputation or default values).
- *Sensor Features:* Input CSV data is aggregated (mean, std, etc.) and then potentially scaled using loaded sensor data scalers if applicable (current implementation seems to use raw aggregated values directly for prediction, but scaling would be a good practice if done during training).

== Model Architecture
The core prediction model is a multi-modal fusion network, specifically an `AttentionFusionMLP` (defined in `model_training_service/scripts/model_def.py` and `disease_prediction_service/app/model_def.py`).
- It takes concatenated features from the three modalities (image, NIH tabular, sensor) as input.
- The `AttentionFusionMLP` model includes an attention mechanism to weigh the importance of different feature modalities before feeding them into subsequent MLP layers. Input features from image, NIH tabular data, and sensor data are first processed by separate linear layers. The outputs are then concatenated and passed through an attention layer. The attention weights are learned to emphasize more relevant modalities for a given prediction. The attended features are then passed through further MLP layers. [NOTE: Adapt this description if your `AttentionFusionMLP` implementation in `model_def.py` has different specifics. The code defines `modal_processors` and an `attention` layer, so this description aligns with a common approach to modality attention.]
- The MLP consists of several hidden layers with activation functions (e.g., ReLU) and dropout for regularization.
- The output layer uses a sigmoid activation function for each class, suitable for multi-label disease prediction. The number of output neurons corresponds to the number of disease classes (14 diseases + "No Finding" as defined in `config_training.py`).

// Remember to create this diagram and place it in the images folder
// #figure(
//   image("images/fusion_model_architecture_diagram.png", width: 80%),
//   caption: [Diagram of the Attention Fusion Model Architecture. Shows input feature vectors, modality-specific processing, attention mechanism, MLP layers, and multi-label output.]
// )
// [PLACEHOLDER: Diagram of the Fusion Model Architecture. This should show the three input feature vectors, how they are concatenated/fused (e.g., simple concatenation or attention mechanism), the MLP layers, and the multi-label output layer.]

== Model Training (`model_training_service`)
The training process is orchestrated by the `model_training_service` using scripts:
1.  *Data Loading (`data_loader.py`):*
    - Fetches study metadata (including feature paths and labels) from the `patient_data_service`.
    - Creates a custom PyTorch `Dataset` (`FusionDataset`) that loads the processed image features, NIH tabular features, and sensor features from their respective MinIO buckets for each study.
    - Labels (`Finding Labels`) are multi-hot encoded using `MultiLabelBinarizer` from scikit-learn. The `ALL_DISEASE_CLASSES` list in `config_training.py` ensures consistent class ordering.
    - Uses PyTorch `DataLoader` for batching and shuffling.
2.  *Training Loop (`train_fusion_model.py`):*
    - Initializes the `AttentionFusionMLP` model, loss function (BCEWithLogitsLoss), and optimizer (e.g., AdamW).
    - Iterates through epochs and batches.
    - For each batch:
        - Performs forward pass.
        - Calculates loss.
        - Performs backward pass and optimizer step.
    - Uses AMP (Automatic Mixed Precision) for potentially faster training.
    - Validation is performed periodically to monitor performance on a hold-out set.
3.  *Evaluation:* Metrics such as ROC AUC (per-class and macro/micro averaged), F1-score, Precision, Recall, and Hamming Loss are calculated.
4.  *Model Saving:*
    - The trained fusion model's state dictionary is saved.
    - The image feature extractor's state dictionary (if fine-tuned, though current setup seems to use it as a fixed extractor) is saved.
    - The `MultiLabelBinarizer` (MLB) instance is saved using `joblib` or `pickle` and stored in MinIO.
    - Other preprocessing artifacts like one-hot encoders and scalers for tabular data are also saved to MinIO.
    - Training metrics and configuration are logged and potentially saved as JSON to MinIO.

== Model Deployment and Serving (`disease_prediction_service`)
- The `disease_prediction_service` loads the trained model components from MinIO upon startup (`models_loader.py`).
- This includes the main fusion model (`AttentionFusionMLP`), the image feature extractor (pre-trained ResNet50), the NIH data OneHotEncoder, and the MultiLabelBinarizer for interpreting outputs.
- The service then exposes an API endpoint (`/predict/`) to receive new patient data (image, NIH tabular, sensor data) and returns disease predictions.