"""
FPD Mix AE Position Embedding Models
통합 모델 파일: Hybrid, Dual Positional, RvC 모델 포함
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from model_defs.autoencoder_16x16 import ResidualEncoder16x16


# ========================================
# 1. Hybrid Model (LatentEmbeddingModel)
# ========================================
class LatentEmbeddingModel(nn.Module):
    """
    Hybrid Model: Hand-crafted features + Latent features with positional embedding
    Section 6 in req.md
    """
    def __init__(self, config):
        super().__init__()

        # Load pre-trained encoder
        self.encoder = ResidualEncoder16x16()
        if config.get('encoder_path'):
            self.load_encoder(config['encoder_path'])

        # Freeze encoder if specified
        if config.get('encoder_frozen', True):
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Positional Embeddings for 7x7 grid
        if config.get('positional_embedding', {}).get('enabled', True):
            self.position_embeddings = nn.Embedding(49, config['positional_embedding']['embedding_dim'])
        else:
            self.position_embeddings = None

        # Feature dimensions
        self.handcrafted_dim = config.get('handcrafted_dim', 903)
        self.latent_dim = config.get('latent_dim', 784)
        self.input_dim = config.get('input_dim', 1687)

        # Fusion Network
        hidden_dims = config.get('hidden_dims', [768, 512, 256])
        dropout_rates = config.get('dropout_rates', [0.2, 0.15, 0.1])

        layers = []
        prev_dim = self.input_dim

        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.get('use_batch_norm', True):
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer (8 coordinates: 4 points × 2)
        layers.append(nn.Linear(prev_dim, 8))

        self.fusion_network = nn.Sequential(*layers)

        # Initialize weights
        if config.get('weight_init') == 'he_normal':
            self._init_weights()

    def load_encoder(self, path):
        """Load pre-trained encoder weights"""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location='cpu')
            # Handle checkpoint format from autoencoder training
            if 'model_state_dict' in checkpoint:
                # Extract encoder weights (remove 'encoder.' prefix)
                state_dict = {
                    k.replace('encoder.', ''): v
                    for k, v in checkpoint['model_state_dict'].items()
                    if k.startswith('encoder.')
                }
            elif 'encoder_state_dict' in checkpoint:
                state_dict = checkpoint['encoder_state_dict']
            else:
                state_dict = checkpoint
            self.encoder.load_state_dict(state_dict)
            print(f"Loaded encoder from {path}")
        else:
            print(f"Encoder file not found: {path}, using random initialization")

    def _init_weights(self):
        """Initialize weights using He normal initialization"""
        for m in self.fusion_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_latent_features(self, image, device):
        """Extract latent features from 7x7 grid of 16x16 patches"""
        batch_size = image.size(0)
        latent_features = []

        # Divide into 7x7 grid
        for y in range(7):
            for x in range(7):
                # Extract 16x16 patch
                patch = image[:, :, y*16:(y+1)*16, x*16:(x+1)*16]

                # Get latent vector from encoder
                with torch.no_grad() if self.encoder.training == False else torch.enable_grad():
                    latent, _ = self.encoder(patch)  # Returns (latent, residuals), we only need latent

                # Add positional embedding if enabled
                if self.position_embeddings is not None:
                    pos_idx = y * 7 + x
                    pos_idx_tensor = torch.full((batch_size,), pos_idx, dtype=torch.long, device=device)
                    pos_embed = self.position_embeddings(pos_idx_tensor)
                    latent = latent + pos_embed

                latent_features.append(latent)

        # Concatenate all latent features
        latent_features = torch.cat(latent_features, dim=1)  # (batch, 49*16=784)
        return latent_features

    def forward(self, handcrafted_features, image):
        """
        Forward pass
        Args:
            handcrafted_features: (batch, 903) hand-crafted features
            image: (batch, 1, 112, 112) input image
        Returns:
            coords: (batch, 8) predicted coordinates
        """
        device = handcrafted_features.device

        # Extract latent features
        latent_features = self.extract_latent_features(image, device)

        # Combine features
        combined_features = torch.cat([handcrafted_features, latent_features], dim=1)

        # Pass through fusion network
        coords = self.fusion_network(combined_features)

        return coords


# ========================================
# 2. Dual Positional Model
# ========================================
class DualPositionalEmbeddingModel(nn.Module):
    """
    Dual Positional Model: Positional embeddings for both hand-crafted and latent features
    Section 7 in req.md
    """
    def __init__(self, config):
        super().__init__()

        # Load pre-trained encoder
        self.encoder = ResidualEncoder16x16()
        if config.get('encoder_path'):
            self.load_encoder(config['encoder_path'])

        # Freeze encoder
        if config.get('encoder_frozen', True):
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Dual Positional Embeddings
        pos_config = config.get('positional_embedding', {})

        # For hand-crafted features
        if pos_config.get('handcrafted', {}).get('enabled', True):
            self.handcrafted_pos_embed = nn.Embedding(49, pos_config['handcrafted']['embedding_dim'])
            self.handcrafted_proj = nn.Linear(
                pos_config['handcrafted']['features_per_patch'],
                pos_config['handcrafted']['projection_dim']
            )
        else:
            self.handcrafted_pos_embed = None
            self.handcrafted_proj = None

        # For latent features
        if pos_config.get('latent', {}).get('enabled', True):
            self.latent_pos_embed = nn.Embedding(49, pos_config['latent']['embedding_dim'])
        else:
            self.latent_pos_embed = None

        # Feature dimensions
        self.input_dim = config.get('input_dim', 1687)

        # Fusion Network (similar to Hybrid)
        hidden_dims = config.get('hidden_dims', [768, 512, 256])
        dropout_rates = config.get('dropout_rates', [0.2, 0.15, 0.1])

        layers = []
        prev_dim = self.input_dim

        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.get('use_batch_norm', True):
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 8))
        self.fusion_network = nn.Sequential(*layers)

    def load_encoder(self, path):
        """Load pre-trained encoder weights"""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location='cpu')
            # Handle checkpoint format from autoencoder training
            if 'model_state_dict' in checkpoint:
                # Extract encoder weights (remove 'encoder.' prefix)
                state_dict = {
                    k.replace('encoder.', ''): v
                    for k, v in checkpoint['model_state_dict'].items()
                    if k.startswith('encoder.')
                }
            elif 'encoder_state_dict' in checkpoint:
                state_dict = checkpoint['encoder_state_dict']
            else:
                state_dict = checkpoint
            self.encoder.load_state_dict(state_dict)

    def forward(self, handcrafted_features, image):
        """Forward pass with dual positional embeddings"""
        device = handcrafted_features.device
        batch_size = image.size(0)

        # Note: This is a simplified version
        # In full implementation, would need grid-aligned feature extraction

        # Extract latent features with positional embedding
        latent_features = []
        for y in range(7):
            for x in range(7):
                patch = image[:, :, y*16:(y+1)*16, x*16:(x+1)*16]
                latent, _ = self.encoder(patch)  # Returns (latent, residuals), we only need latent

                if self.latent_pos_embed is not None:
                    pos_idx = y * 7 + x
                    pos_embed = self.latent_pos_embed(torch.tensor(pos_idx, device=device))
                    latent = latent + pos_embed

                latent_features.append(latent)

        latent_features = torch.cat(latent_features, dim=1)

        # Combine all features
        combined = torch.cat([handcrafted_features, latent_features], dim=1)

        # Pass through fusion network
        coords = self.fusion_network(combined)

        return coords


# ========================================
# 3. RvC Model (Regression via Classification)
# ========================================
class RvCModel(nn.Module):
    """
    Regression via Classification Model
    Section 12 in req.md
    """
    def __init__(self, config):
        super().__init__()

        # Bin configuration
        self.num_x_bins = config.get('num_x_bins', 168)
        self.num_y_bins = config.get('num_y_bins', 112)
        self.x_range = config.get('x_range', [-112, 224])
        self.y_range = config.get('y_range', [0, 224])

        # Create bins
        self.x_bins = torch.linspace(self.x_range[0], self.x_range[1], self.num_x_bins + 1)
        self.y_bins = torch.linspace(self.y_range[0], self.y_range[1], self.num_y_bins + 1)

        # Architecture
        arch_config = config.get('architecture', {})
        input_dim = arch_config.get('input_dim', 903)

        # Feature extractor network
        feature_config = arch_config.get('feature_network', {})
        hidden_dims = feature_config.get('hidden_dims', [768, 512, 256])
        dropout_rates = feature_config.get('dropout_rates', [0.2, 0.15, 0.1])

        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if feature_config.get('use_batch_norm', True):
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Classification heads
        classifier_config = arch_config.get('classifier_head', {})
        hidden_dim = classifier_config.get('hidden_dim', 128)
        dropout = classifier_config.get('dropout', 0.1)

        # X coordinate classifiers (4 points)
        self.x_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_x_bins)
            ) for _ in range(4)
        ])

        # Y coordinate classifiers (4 points)
        self.y_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_y_bins)
            ) for _ in range(4)
        ])

        # Uncertainty head (optional)
        self.predict_uncertainty = config.get('predict_uncertainty', False)
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Linear(prev_dim, 8)

    def forward(self, features):
        """
        Forward pass
        Args:
            features: (batch, input_dim) input features
        Returns:
            dict with x_logits and y_logits
        """
        # Extract features
        features = self.feature_extractor(features)

        # Classification for each point
        x_logits = [clf(features) for clf in self.x_classifiers]
        y_logits = [clf(features) for clf in self.y_classifiers]

        output = {
            'x_logits': x_logits,  # List of 4 tensors, each (batch, num_x_bins)
            'y_logits': y_logits   # List of 4 tensors, each (batch, num_y_bins)
        }

        # Add uncertainty if enabled
        if self.predict_uncertainty:
            output['uncertainty'] = self.uncertainty_head(features)

        return output

    def decode_predictions(self, output, method='weighted_average', temperature=1.0):
        """
        Decode classification outputs to continuous coordinates
        Args:
            output: Model output dict
            method: 'argmax' or 'weighted_average'
            temperature: Softmax temperature
        Returns:
            coords: (batch, 8) continuous coordinates
        """
        device = output['x_logits'][0].device
        batch_size = output['x_logits'][0].size(0)

        # Move bins to device
        x_bins = self.x_bins.to(device)
        y_bins = self.y_bins.to(device)

        # Bin centers
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2

        coords = []

        for i in range(4):
            if method == 'argmax':
                # Get most likely bin
                x_idx = torch.argmax(output['x_logits'][i], dim=-1)
                y_idx = torch.argmax(output['y_logits'][i], dim=-1)
                x_coord = x_centers[x_idx]
                y_coord = y_centers[y_idx]
            else:  # weighted_average
                # Softmax weighted average
                x_probs = F.softmax(output['x_logits'][i] / temperature, dim=-1)
                y_probs = F.softmax(output['y_logits'][i] / temperature, dim=-1)
                x_coord = torch.sum(x_probs * x_centers, dim=-1)
                y_coord = torch.sum(y_probs * y_centers, dim=-1)

            coords.append(x_coord)
            coords.append(y_coord)

        return torch.stack(coords, dim=1)


# ========================================
# 4. RvC Loss Function
# ========================================
class RvCLoss(nn.Module):
    """
    Loss function for RvC model
    """
    def __init__(self, config):
        super().__init__()

        self.use_soft_labels = config.get('use_soft_labels', True)
        self.soft_label_sigma = config.get('soft_label_sigma', 2.0)
        self.ordinal_weight = config.get('ordinal_weight', 0.1)

        # Bin configuration
        self.num_x_bins = config.get('num_x_bins', 168)
        self.num_y_bins = config.get('num_y_bins', 112)
        self.x_range = config.get('x_range', [-112, 224])
        self.y_range = config.get('y_range', [0, 224])

        # Create bins
        self.x_bins = torch.linspace(self.x_range[0], self.x_range[1], self.num_x_bins + 1)
        self.y_bins = torch.linspace(self.y_range[0], self.y_range[1], self.num_y_bins + 1)

    def create_soft_labels(self, value, bins, sigma=None):
        """Create soft labels with Gaussian distribution"""
        if sigma is None:
            sigma = self.soft_label_sigma

        bin_centers = (bins[:-1] + bins[1:]) / 2
        distances = torch.abs(bin_centers - value.unsqueeze(-1))
        soft_labels = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
        soft_labels = soft_labels / soft_labels.sum(dim=-1, keepdim=True)
        return soft_labels

    def forward(self, predictions, targets):
        """
        Calculate loss
        Args:
            predictions: Model output dict with x_logits and y_logits
            targets: (batch, 8) target coordinates
        Returns:
            total_loss: Scalar loss value
        """
        device = targets.device
        batch_size = targets.size(0)

        # Move bins to device
        x_bins = self.x_bins.to(device)
        y_bins = self.y_bins.to(device)

        total_loss = 0

        for i in range(4):
            # Extract target coordinates for this point
            target_x = targets[:, i*2]
            target_y = targets[:, i*2 + 1]

            if self.use_soft_labels:
                # Create soft labels
                x_soft_labels = self.create_soft_labels(target_x, x_bins)
                y_soft_labels = self.create_soft_labels(target_y, y_bins)

                # KL divergence loss
                x_loss = F.kl_div(
                    F.log_softmax(predictions['x_logits'][i], dim=-1),
                    x_soft_labels,
                    reduction='batchmean'
                )
                y_loss = F.kl_div(
                    F.log_softmax(predictions['y_logits'][i], dim=-1),
                    y_soft_labels,
                    reduction='batchmean'
                )
            else:
                # Hard labels - find closest bin
                x_bins_expanded = x_bins.unsqueeze(0).expand(batch_size, -1)
                y_bins_expanded = y_bins.unsqueeze(0).expand(batch_size, -1)

                x_diffs = torch.abs(x_bins_expanded - target_x.unsqueeze(-1))
                y_diffs = torch.abs(y_bins_expanded - target_y.unsqueeze(-1))

                x_classes = torch.argmin(x_diffs[:, :-1] + x_diffs[:, 1:], dim=-1)
                y_classes = torch.argmin(y_diffs[:, :-1] + y_diffs[:, 1:], dim=-1)

                # Cross-entropy loss
                x_loss = F.cross_entropy(predictions['x_logits'][i], x_classes)
                y_loss = F.cross_entropy(predictions['y_logits'][i], y_classes)

            total_loss += x_loss + y_loss

            # Add ordinal penalty if specified
            if self.ordinal_weight > 0:
                pred_x_class = torch.argmax(predictions['x_logits'][i], dim=-1)
                pred_y_class = torch.argmax(predictions['y_logits'][i], dim=-1)

                if not self.use_soft_labels:
                    x_distance = torch.abs(pred_x_class.float() - x_classes.float()).mean()
                    y_distance = torch.abs(pred_y_class.float() - y_classes.float()).mean()
                    total_loss += self.ordinal_weight * (x_distance + y_distance)

        return total_loss


# ========================================
# 5. Model Factory
# ========================================
def create_model(model_type, config):
    """
    Create model based on type
    Args:
        model_type: 'hybrid', 'dual_positional', or 'rvc'
        config: Model configuration dict
    Returns:
        model: Created model instance
    """
    if model_type == 'hybrid':
        return LatentEmbeddingModel(config)
    elif model_type == 'dual_positional':
        return DualPositionalEmbeddingModel(config)
    elif model_type == 'rvc':
        return RvCModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Test Hybrid Model
    hybrid_config = {
        'encoder_path': None,
        'encoder_frozen': True,
        'positional_embedding': {'enabled': True, 'embedding_dim': 16},
        'handcrafted_dim': 903,
        'latent_dim': 784,
        'input_dim': 1687,
        'hidden_dims': [768, 512, 256],
        'dropout_rates': [0.2, 0.15, 0.1],
        'use_batch_norm': True,
        'weight_init': 'he_normal'
    }

    hybrid_model = create_model('hybrid', hybrid_config)
    print(f"Hybrid model created: {sum(p.numel() for p in hybrid_model.parameters())} parameters")

    # Test RvC Model
    rvc_config = {
        'num_x_bins': 168,
        'num_y_bins': 112,
        'x_range': [-112, 224],
        'y_range': [0, 224],
        'predict_uncertainty': True,
        'architecture': {
            'input_dim': 903,
            'feature_network': {
                'hidden_dims': [768, 512, 256],
                'dropout_rates': [0.2, 0.15, 0.1],
                'use_batch_norm': True
            },
            'classifier_head': {
                'hidden_dim': 128,
                'dropout': 0.1
            }
        }
    }

    rvc_model = create_model('rvc', rvc_config)
    print(f"RvC model created: {sum(p.numel() for p in rvc_model.parameters())} parameters")

    print("All models created successfully!")