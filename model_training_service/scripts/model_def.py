import torch
import torch.nn as nn


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
    
class AttentionFusionMLP(nn.Module):
    def __init__(self, img_feature_dim, nih_feature_dim, sensor_feature_dim,
                 num_classes, embed_dim, num_heads, dropout_rate=0.3):
        super(AttentionFusionMLP, self).__init__()

        self.img_feature_dim = img_feature_dim
        self.nih_feature_dim = nih_feature_dim
        self.sensor_feature_dim = sensor_feature_dim
        self.embed_dim = embed_dim

        self.img_projection = nn.Linear(img_feature_dim, embed_dim)
        self.nih_projection = nn.Linear(nih_feature_dim, embed_dim)
        self.sensor_projection = nn.Linear(sensor_feature_dim, embed_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=False)

        self.norm1 = nn.LayerNorm(embed_dim)
        

        self.classifier_input_dim = embed_dim 
        self.fc1 = nn.Linear(self.classifier_input_dim, self.classifier_input_dim // 2)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout_rate) 
        self.fc2 = nn.Linear(self.classifier_input_dim // 2, num_classes)

    def forward(self, img_features, nih_features, sensor_features,
                img_mask, nih_mask, sensor_mask):
        """
        Args:
            img_features (torch.Tensor): Batch of image features (batch_size, img_feature_dim).
                                         Can be zero tensor if masked/missing.
            nih_features (torch.Tensor): Batch of NIH features (batch_size, nih_feature_dim).
                                         Can be zero tensor if masked/missing.
            sensor_features (torch.Tensor): Batch of sensor features (batch_size, sensor_feature_dim).
                                            Can be zero tensor if masked/missing.
            img_mask (torch.Tensor): Boolean tensor (batch_size,), True if image is masked/missing for a sample.
            nih_mask (torch.Tensor): Boolean tensor (batch_size,), True if NIH is masked/missing for a sample.
            sensor_mask (torch.Tensor): Boolean tensor (batch_size,), True if sensor is masked/missing for a sample.
        Returns:
            torch.Tensor: Output logits (batch_size, num_classes).
        """
        batch_size = img_features.size(0)
        device = img_features.device

        img_emb = self.img_projection(img_features)          # (batch_size, embed_dim)
        nih_emb = self.nih_projection(nih_features)          # (batch_size, embed_dim)
        sensor_emb = self.sensor_projection(sensor_features)  # (batch_size, embed_dim)

       
        modal_embeddings = torch.stack([img_emb, nih_emb, sensor_emb], dim=0)

        key_padding_mask = torch.stack([
            img_mask.to(device),
            nih_mask.to(device),
            sensor_mask.to(device)
        ], dim=1)


        attn_output, attn_weights = self.multihead_attn(
            modal_embeddings, modal_embeddings, modal_embeddings,
            key_padding_mask=key_padding_mask
        )

        attended_features = self.norm1(attn_output) # (3, batch_size, embed_dim)

        # Shape: (3, batch_size, 1) for broadcasting
        expanded_key_padding_mask = (~key_padding_mask).permute(1, 0).unsqueeze(-1).to(device)
        
        masked_attn_contributions = attended_features * expanded_key_padding_mask
        sum_attn_contributions = torch.sum(masked_attn_contributions, dim=0)  # (batch_size, embed_dim)
        
        num_unmasked_modalities = expanded_key_padding_mask.sum(dim=0)  # (batch_size, 1)
        num_unmasked_modalities = torch.clamp(num_unmasked_modalities, min=1.0)
        
        aggregated_features = sum_attn_contributions / num_unmasked_modalities  # (batch_size, embed_dim)

        x = self.fc1(aggregated_features)
        x = self.relu(x)
        x = self.dropout_layer(x) 
        x = self.fc2(x)

        return x