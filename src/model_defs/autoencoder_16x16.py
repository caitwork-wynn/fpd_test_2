#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16x16 이미지를 위한 Residual Autoencoder 모델 정의
분리된 Encoder와 Decoder로 구성되어 나중에 latent vector만 사용 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEncoder16x16(nn.Module):
    """
    Residual Style Encoder for 16x16 images
    - Separate encoder with residual connections
    - Returns latent vector and residual features for skip connections
    """
    def __init__(self, input_channels=1, latent_dim=16):
        super(ResidualEncoder16x16, self).__init__()
        self.latent_dim = latent_dim

        # Encoder blocks with residual connections
        self.enc1 = self._make_encoder_block(input_channels, 32, 3, 1, 1)  # 16x16
        self.enc2 = self._make_encoder_block(32, 64, 3, 2, 1)              # 8x8
        self.enc3 = self._make_encoder_block(64, 128, 3, 2, 1)             # 4x4
        self.enc4 = self._make_encoder_block(128, 256, 3, 2, 1)            # 2x2

        # Bottleneck to latent space
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling: 256x2x2 → 256x1x1
            nn.Flatten(),             # 256x1x1 → 256
            nn.Linear(256, latent_dim, bias=False),
            nn.ReLU(inplace=True)
        )

        # Residual connection adapters (for dimension matching)
        self.res_adapt1 = nn.Conv2d(32, 32, 1, bias=False)   # For skip connection 1
        self.res_adapt2 = nn.Conv2d(64, 64, 1, bias=False)   # For skip connection 2
        self.res_adapt3 = nn.Conv2d(128, 128, 1, bias=False) # For skip connection 3

    def _make_encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Create encoder block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through encoder
        Returns:
            latent: Compressed representation (batch_size, latent_dim)
            residuals: List of residual features for skip connections
        """
        # Encoder forward with residual feature collection
        res1 = self.enc1(x)              # 32 channels, 16x16
        res2 = self.enc2(res1)           # 64 channels, 8x8
        res3 = self.enc3(res2)           # 128 channels, 4x4
        encoded = self.enc4(res3)        # 256 channels, 2x2

        # Convert to latent representation
        latent = self.bottleneck(encoded)  # latent_dim

        # Prepare residual features for skip connections
        residuals = {
            'res1': self.res_adapt1(res1),  # Adapted 32x16x16
            'res2': self.res_adapt2(res2),  # Adapted 64x8x8
            'res3': self.res_adapt3(res3)   # Adapted 128x4x4
        }

        return latent, residuals


class ResidualDecoder16x16(nn.Module):
    """
    Residual Style Decoder for 16x16 images
    - Separate decoder that uses residual features from encoder
    - Reconstructs image using skip connections
    """
    def __init__(self, output_channels=1, latent_dim=16):
        super(ResidualDecoder16x16, self).__init__()
        self.latent_dim = latent_dim

        # Expand latent to feature map
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 1, 1))  # Reshape to 256x1x1
        )

        # Decoder blocks
        self.dec4 = self._make_decoder_block(256, 128, 4, 2, 1, 0)         # 1x1 → 2x2
        self.dec3 = self._make_decoder_block(128, 64, 4, 2, 1, 0)          # 2x2 → 4x4
        self.dec2 = self._make_decoder_block(64, 32, 4, 2, 1, 0)           # 4x4 → 8x8
        self.dec1 = nn.Conv2d(32, output_channels, 3, padding=1)           # 8x8 → 16x16

        # Output activation
        self.sigmoid = nn.Sigmoid()

    def _make_decoder_block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        """Create decoder block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                              padding, output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, latent, residuals):
        """
        Forward pass through decoder
        Args:
            latent: Compressed representation (batch_size, latent_dim)
            residuals: Dictionary of residual features from encoder
        Returns:
            reconstructed: Reconstructed image (batch_size, output_channels, 16, 16)
        """
        # Expand latent to feature map
        x = self.latent_expand(latent)  # latent_dim → 256x1x1

        # Upsample to 2x2 for decoder start
        x = F.interpolate(x, size=(2, 2), mode='bilinear', align_corners=False)

        # Decoder with residual connections
        x = self.dec4(x)                           # 128 channels, 2x2
        x = x + residuals['res3']                  # Residual connection 3

        x = self.dec3(x)                           # 64 channels, 4x4
        x = x + residuals['res2']                  # Residual connection 2

        x = self.dec2(x)                           # 32 channels, 8x8
        x = x + residuals['res1']                  # Residual connection 1

        # Final output - need ConvTranspose2d for 8x8 → 16x16
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        x = self.sigmoid(self.dec1(x))             # output_channels, 16x16

        return x


class SeparatedResidualAutoencoder16x16(nn.Module):
    """
    Complete Autoencoder with separated Encoder and Decoder
    - Combines ResidualEncoder16x16 and ResidualDecoder16x16
    - Maintains residual skip connections between them
    """
    def __init__(self, input_channels=1, output_channels=1, latent_dim=16):
        super(SeparatedResidualAutoencoder16x16, self).__init__()

        # Separated encoder and decoder
        self.encoder = ResidualEncoder16x16(input_channels, latent_dim)
        self.decoder = ResidualDecoder16x16(output_channels, latent_dim)

    def encode(self, x):
        """Encode input to latent representation"""
        latent, residuals = self.encoder(x)
        return latent, residuals

    def decode(self, latent, residuals):
        """Decode latent representation to output"""
        return self.decoder(latent, residuals)

    def forward(self, x):
        """Complete autoencoder forward pass"""
        latent, residuals = self.encode(x)
        reconstructed = self.decode(latent, residuals)
        return reconstructed

    def get_latent_representation(self, x):
        """Get only the latent representation (for analysis)"""
        latent, _ = self.encode(x)
        return latent