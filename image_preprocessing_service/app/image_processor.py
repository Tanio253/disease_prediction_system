import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import io
import logging
from .config import IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD, DEVICE, PRETRAINED_MODEL_NAME

logger = logging.getLogger(__name__)

# Global model variable to load it only once
feature_extractor_model = None
device_to_use = None

def get_model_and_device():
    global feature_extractor_model, device_to_use
    if feature_extractor_model is None:
        logger.info(f"Loading pre-trained model: {PRETRAINED_MODEL_NAME} on device: {DEVICE}")
        try:
            if PRETRAINED_MODEL_NAME == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V2 # Using updated weights API
                model = models.resnet50(weights=weights)
            elif PRETRAINED_MODEL_NAME == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                model = models.efficientnet_b0(weights=weights)
            # Add other models as needed
            else:
                logger.error(f"Unsupported model name: {PRETRAINED_MODEL_NAME}. Defaulting to ResNet50.")
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                model = models.resnet50(weights=weights)

            # Remove the final classification layer (specific to model architecture)
            if hasattr(model, 'fc'): # For ResNet
                model.fc = torch.nn.Identity()
            elif hasattr(model, 'classifier'): # For EfficientNet, VGG, etc.
                if isinstance(model.classifier, torch.nn.Sequential):
                     # For EfficientNet, classifier is Sequential(Dropout, Linear)
                    model.classifier = torch.nn.Identity() # Or model.classifier[0] if only Linear needs removing
                else: # Single Linear layer
                    model.classifier = torch.nn.Identity()
            else:
                logger.warning(f"Could not automatically remove classification layer for {PRETRAINED_MODEL_NAME}. Output might be class scores.")

            device_to_use = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
            feature_extractor_model = model.to(device_to_use)
            feature_extractor_model.eval()
            logger.info(f"Model {PRETRAINED_MODEL_NAME} loaded successfully on {device_to_use}.")
        except Exception as e:
            logger.error(f"Error loading model {PRETRAINED_MODEL_NAME}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model {PRETRAINED_MODEL_NAME}") from e
            
    return feature_extractor_model, device_to_use

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
        ])
        tensor = transform(image)
        return tensor.unsqueeze(0) # Add batch dimension
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}", exc_info=True)
        raise

def extract_features(image_tensor: torch.Tensor, model, device) -> torch.Tensor:
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            features = model(image_tensor)
        return features.squeeze(0).cpu() # Remove batch dim and move to CPU
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}", exc_info=True)
        raise