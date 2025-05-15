# In model_def.py (shared or copied to both services)
import torch
import torch.nn as nn
# from .config_training import NUM_CLASSES, DROPOUT_RATE # If config is specific to training context
# OR pass these as args

class FusionMLP(nn.Module):
    def __init__(self, image_feature_dim: int, nih_tabular_feature_dim: int, sensor_feature_dim: int,
                 hidden_dims_mlp: list, num_classes: int, dropout_rate: float):
        super(FusionMLP, self).__init__()
        self.total_input_dim = image_feature_dim + nih_tabular_feature_dim + sensor_feature_dim
        
        layers = []
        current_dim = self.total_input_dim
        
        for h_dim in hidden_dims_mlp:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, image_features, nih_tabular_features, sensor_features):
        x = torch.cat((image_features, nih_tabular_features, sensor_features), dim=1)
        return self.network(x)