#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16x16 이미지를 위한 최소 크기 Residual Autoencoder 모델 정의
분리된 Encoder와 Decoder로 구성되어 나중에 latent vector만 사용 가능
최소 파라미터 수를 위해 채널 수와 레이어 깊이를 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalEncoder16x16(nn.Module):
    """
    Minimal Residual Style Encoder for 16x16 images
    - 최소 크기 encoder with residual connections
    - Returns latent vector and residual features for skip connections
    - 채널 수: 16→32→64 (7x7 모델과 유사)
    """
    def __init__(self, input_channels=1, latent_dim=32):
        super(MinimalEncoder16x16, self).__init__()
        self.latent_dim = latent_dim

        # Encoder blocks with residual connections (최소 채널 수)
        self.enc1 = self._make_encoder_block(input_channels, 16, 3, 1, 1)  # 16x16
        self.enc2 = self._make_encoder_block(16, 32, 3, 2, 1)              # 8x8
        self.enc3 = self._make_encoder_block(32, 64, 3, 2, 1)              # 4x4

        # Bottleneck to latent space (Global Pooling 방식 - FC보다 12배 작음)
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling: 64x4x4 → 64x1x1 (0 params)
            nn.Flatten(),             # 64x1x1 → 64
            nn.Linear(64, latent_dim, bias=False),  # 64 × 32 = 2,048 params
            nn.ReLU(inplace=True)
        )

        # Residual connection adapters (for dimension matching)
        self.res_adapt1 = nn.Conv2d(16, 16, 1, bias=False)  # For skip connection 1 (256 params)
        self.res_adapt2 = nn.Conv2d(32, 32, 1, bias=False)  # For skip connection 2 (1,024 params)

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
            residuals: Dictionary of residual features for skip connections
        """
        # Encoder forward with residual feature collection
        res1 = self.enc1(x)              # 16 channels, 16x16
        res2 = self.enc2(res1)           # 32 channels, 8x8
        encoded = self.enc3(res2)        # 64 channels, 4x4

        # Convert to latent representation (Global Pooling)
        latent = self.bottleneck(encoded)  # latent_dim

        # Prepare residual features for skip connections
        residuals = {
            'res1': self.res_adapt1(res1),  # Adapted 16x16x16
            'res2': self.res_adapt2(res2)   # Adapted 32x8x8
        }

        return latent, residuals


class MinimalDecoder16x16(nn.Module):
    """
    Minimal Residual Style Decoder for 16x16 images
    - 최소 크기 decoder that uses residual features from encoder
    - Reconstructs image using skip connections
    """
    def __init__(self, output_channels=1, latent_dim=32):
        super(MinimalDecoder16x16, self).__init__()
        self.latent_dim = latent_dim

        # Expand latent to feature map
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, 64, bias=False),  # 32 × 64 = 2,048 params
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 1, 1))  # Reshape to 64x1x1
        )

        # Decoder blocks (최소 채널 수)
        self.dec3 = self._make_decoder_block(64, 32, 4, 2, 1, 0)           # 1x1 → 4x4
        self.dec2 = self._make_decoder_block(32, 16, 4, 2, 1, 0)           # 4x4 → 8x8
        self.dec1 = nn.Conv2d(16, output_channels, 3, padding=1)           # 8x8 → 16x16

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
        x = self.latent_expand(latent)  # latent_dim → 64x1x1

        # Upsample to 4x4 for decoder start
        x = F.interpolate(x, size=(4, 4), mode='bilinear', align_corners=False)

        # Decoder with residual connections
        x = self.dec3(x)                           # 32 channels, 4x4
        x = x + residuals['res2']                  # Residual connection 2

        x = self.dec2(x)                           # 16 channels, 8x8
        x = x + residuals['res1']                  # Residual connection 1

        # Final output - upsample to 16x16
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        x = self.sigmoid(self.dec1(x))             # output_channels, 16x16

        return x


class MinimalAutoencoder16x16(nn.Module):
    """
    최소 크기 Complete Autoencoder with separated Encoder and Decoder
    - Combines MinimalEncoder16x16 and MinimalDecoder16x16
    - Maintains residual skip connections between them
    - 최소 파라미터 수: ~5,376 params (bottleneck + adapters)
    """
    def __init__(self, input_channels=1, output_channels=1, latent_dim=32):
        super(MinimalAutoencoder16x16, self).__init__()

        # Separated encoder and decoder
        self.encoder = MinimalEncoder16x16(input_channels, latent_dim)
        self.decoder = MinimalDecoder16x16(output_channels, latent_dim)

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
