# -*- coding: utf-8 -*-
"""
MPM MobileNetV2 Hybrid 모델
- 96×96 RGB 입력
- MobileNetV2(0.5) Backbone (Pretrained ImageNet)
- CBAM Attention (Channel + Spatial)
- Lightweight Self-Attention
- Classification-based Coordinate Prediction
- 210.learning_new.py 호환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import warnings

# util 폴더를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))
from util.data_augmentation import apply_crop_augmentation, apply_flip_augmentation
from util.image_resize import resize_image_with_coordinates

# matplotlib 경고 억제
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning)

# torchvision import
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. MobileNetV2 backbone will not be available.")


# ============================================
# 모델 설정 상수
# ============================================
TARGET_POINTS = ['center', 'floor', 'front', 'side']
USE_FPD_ARCHITECTURE = True
IMAGE_SIZE = [96, 96]
GRID_SIZE = 7
USE_AUTOENCODER = False  # MobileNetV2는 이미지 직접 사용
ENCODER_PATH = None
ENCODER_LATENT_DIM = 0
SAVE_FILE_NAME = f'mpm_100_mobilenet_{IMAGE_SIZE[0]}'

# 모델 설정
COORD_BINS = 32  # 원본 64 대신 32 bins (경량화)
GAUSSIAN_SIGMA = 1.5  # Gaussian soft label sigma
EMBED_DIM = 128  # Feature embedding dimension
NUM_HEADS = 2  # Self-Attention heads (경량화)
FFN_EXPANSION = 2  # FFN expansion factor (경량화)


def get_model_config():
    """모델 설정 반환 (210.learning_new.py 호환)"""
    return {
        'target_points': TARGET_POINTS,
        'use_fpd_architecture': USE_FPD_ARCHITECTURE,
        'save_file_name': SAVE_FILE_NAME,
        'features': {
            'image_size': IMAGE_SIZE,
            'grid_size': GRID_SIZE,
            'use_autoencoder': USE_AUTOENCODER,
            'encoder_path': ENCODER_PATH,
            'encoder_latent_dim': ENCODER_LATENT_DIM
        }
    }


def get_input_dim():
    """모델의 기대 입력 차원 반환 (ONNX 변환용)

    Returns:
        tuple: (channels, height, width) - (3, 96, 96)
    """
    return (3, IMAGE_SIZE[0], IMAGE_SIZE[1])


# ============================================
# MobileNetV2 Backbone
# ============================================

class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 Feature Extractor (Pretrained)"""

    def __init__(self, pretrained: bool = True, width_mult: float = 0.5, freeze_backbone: bool = False):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for MobileNetV2Backbone")

        # MobileNetV2 로드 (width_mult=0.5로 경량화)
        try:
            # torchvision >= 0.13 (weights 파라미터 사용)
            if pretrained:
                from torchvision.models import MobileNet_V2_Weights
                self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                self.mobilenet = models.mobilenet_v2(weights=None)
        except:
            # torchvision < 0.13 (pretrained 파라미터 사용)
            self.mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # width_mult 적용 (수동으로 채널 조정)
        # 주의: torchvision의 mobilenet_v2는 width_mult를 직접 지원하지 않으므로
        # pretrained weights를 그대로 사용 (1.0 width_mult)
        # 성능 향상을 위해 pretrained weights를 활용

        # features만 사용 (classifier 제거)
        self.features = self.mobilenet.features

        # 출력 채널 수 (MobileNetV2의 마지막 conv 채널)
        self.out_channels = 1280  # width_mult=1.0 기준

        # Freeze 설정
        if freeze_backbone:
            self.freeze()

    def freeze(self):
        """Backbone freeze (Transfer Learning Phase 1)"""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Backbone unfreeze (Transfer Learning Phase 2)"""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 96, 96]
        Returns:
            features: [B, 1280, 3, 3]
        """
        return self.features(x)


# ============================================
# CBAM Attention Modules
# ============================================

class ChannelAttention(nn.Module):
    """Channel Attention Module (SE-style, ONNX compatible)"""

    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # MaxPool을 제거하고 AvgPool만 사용 (ONNX 호환성)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Global average pooling only (ONNX compatible)
        avg_pool = self.avg_pool(x).view(B, C)

        # FC layers
        avg_out = self.fc(avg_pool)

        # Channel attention
        attention = self.sigmoid(avg_out).view(B, C, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention Module (ONNX compatible)"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # Single channel input (avg pool only for ONNX)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention: [B, C, H, W]
        """
        # Channel-wise average pooling only (ONNX compatible)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Spatial attention
        attention = self.sigmoid(self.conv(avg_pool))  # [B, 1, H, W]
        return x * attention


class CBAMModule(nn.Module):
    """CBAM: Channel + Spatial Attention"""

    def __init__(self, in_channels: int, reduction_ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================
# Feature Compression
# ============================================

class FeatureCompression(nn.Module):
    """Global Pooling + FC for feature compression"""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [B, out_dim]
        """
        B = x.size(0)
        x = self.gap(x).view(B, -1)  # [B, C]
        x = self.fc(x)  # [B, out_dim]
        return x


# ============================================
# Lightweight Self-Attention
# ============================================

class LightweightSelfAttention(nn.Module):
    """Lightweight Self-Attention (경량화 버전)"""

    def __init__(self, embed_dim: int = 128, num_heads: int = 2,
                 ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_expansion, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, embed_dim]
        Returns:
            features: [B, 1, embed_dim]
        """
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# ============================================
# Coordinate Classification Head
# ============================================

class CoordinateClassificationHead(nn.Module):
    """분류 기반 좌표 예측 헤드"""

    def __init__(self, input_dim: int, num_bins: int,
                 coord_min: float, coord_max: float, sigma: float = 1.5):
        super().__init__()
        self.num_bins = num_bins
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.sigma = sigma

        # Bin centers 등록
        bin_centers = torch.linspace(coord_min, coord_max, num_bins)
        self.register_buffer('bin_centers', bin_centers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_bins)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B, num_bins]
            coords: [B]
        """
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        coords = torch.sum(probs * self.bin_centers.unsqueeze(0), dim=-1)
        return logits, coords

    def create_soft_label(self, targets: torch.Tensor) -> torch.Tensor:
        """Gaussian soft label 생성"""
        B = targets.size(0)
        targets = targets.unsqueeze(1)  # [B, 1]

        # Gaussian distribution
        distances = (self.bin_centers.unsqueeze(0) - targets) ** 2
        soft_labels = torch.exp(-distances / (2 * self.sigma ** 2))

        # Normalize
        soft_labels = soft_labels / (soft_labels.sum(dim=-1, keepdim=True) + 1e-8)
        return soft_labels


# ============================================
# Main Model
# ============================================

class MPMMobileNetV2HybridModel(nn.Module):
    """MobileNetV2 Hybrid 모델 - RGB 이미지 → 좌표 예측"""
    MODEL_NAME = "mpm_mobilenet_v2_hybrid"

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # 좌표 설정
        self.x_range = (-112, 224)
        self.y_range = (0, 224)
        self.num_coordinates = 8  # 4 points × 2
        self.point_names = ['center', 'floor', 'front', 'side']

        # 하이퍼파라미터
        self.coord_bins = config.get('coord_bins', COORD_BINS)
        self.gaussian_sigma = config.get('gaussian_sigma', GAUSSIAN_SIGMA)
        self.embed_dim = config.get('embed_dim', EMBED_DIM)
        self.num_heads = config.get('num_heads', NUM_HEADS)
        self.ffn_expansion = config.get('ffn_expansion', FFN_EXPANSION)
        self.loss_alpha = config.get('loss_alpha', 0.7)  # KL vs MSE weight

        # Stage 1: MobileNetV2 Backbone
        freeze_backbone = config.get('freeze_backbone', False)
        self.backbone = MobileNetV2Backbone(
            pretrained=True,
            width_mult=0.5,
            freeze_backbone=freeze_backbone
        )
        backbone_channels = self.backbone.out_channels  # 1280

        # Stage 2: CBAM Attention
        self.cbam = CBAMModule(
            in_channels=backbone_channels,
            reduction_ratio=8
        )

        # Stage 3: Feature Compression
        self.compression = FeatureCompression(
            in_channels=backbone_channels,
            out_dim=self.embed_dim
        )

        # Stage 4: Self-Attention
        self.self_attention = LightweightSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_expansion=self.ffn_expansion
        )

        # Stage 5: Coordinate Heads (8개)
        self.coordinate_heads = nn.ModuleList()
        for i in range(self.num_coordinates):
            if i % 2 == 0:  # x
                head = CoordinateClassificationHead(
                    self.embed_dim, self.coord_bins,
                    self.x_range[0], self.x_range[1],
                    self.gaussian_sigma
                )
            else:  # y
                head = CoordinateClassificationHead(
                    self.embed_dim, self.coord_bins,
                    self.y_range[0], self.y_range[1],
                    self.gaussian_sigma
                )
            self.coordinate_heads.append(head)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, 96, 96] RGB images

        Returns:
            dict with 'coordinates', 'pixel_coordinates', 'logits'
        """
        B = images.size(0)

        # Stage 1: Backbone
        features = self.backbone(images)  # [B, 1280, 3, 3]

        # Stage 2: CBAM Attention
        features = self.cbam(features)  # [B, 1280, 3, 3]

        # Stage 3: Compression
        compressed = self.compression(features)  # [B, 128]

        # Stage 4: Self-Attention
        compressed = compressed.unsqueeze(1)  # [B, 1, 128]
        enhanced = self.self_attention(compressed)  # [B, 1, 128]
        enhanced = enhanced.squeeze(1)  # [B, 128]

        # Stage 5: Coordinate Prediction
        all_logits = []
        all_coords = []

        for head in self.coordinate_heads:
            logits, coords = head(enhanced)
            all_logits.append(logits)
            all_coords.append(coords)

        pixel_coordinates = torch.stack(all_coords, dim=1)  # [B, 8]

        # 정규화
        norm_coordinates = torch.zeros_like(pixel_coordinates)
        for i in range(self.num_coordinates):
            if i % 2 == 0:  # x
                norm_coordinates[:, i] = (pixel_coordinates[:, i] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
            else:  # y
                norm_coordinates[:, i] = (pixel_coordinates[:, i] - self.y_range[0]) / (self.y_range[1] - self.y_range[0])

        return {
            'logits': all_logits,
            'coordinates': norm_coordinates,
            'pixel_coordinates': pixel_coordinates
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor, sigma: float = None) -> torch.Tensor:
        """손실 계산 (KL + MSE Hybrid)"""
        if sigma is None:
            sigma = self.gaussian_sigma

        logits_list = outputs['logits']
        pred_coords = outputs['pixel_coordinates']

        total_kl_loss = 0

        # KL Divergence Loss
        for i in range(self.num_coordinates):
            if i % 2 == 0:  # x
                target_pixel = targets[:, i] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            else:  # y
                target_pixel = targets[:, i] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

            soft_label = self.coordinate_heads[i].create_soft_label(target_pixel)
            log_probs = F.log_softmax(logits_list[i], dim=-1)
            kl_loss = -torch.sum(soft_label * log_probs, dim=-1).mean()
            total_kl_loss += kl_loss

        total_kl_loss = total_kl_loss / self.num_coordinates

        # MSE Loss (픽셀 좌표 직접 비교)
        target_coords_pixel = torch.zeros_like(targets)
        for i in range(self.num_coordinates):
            if i % 2 == 0:
                target_coords_pixel[:, i] = targets[:, i] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            else:
                target_coords_pixel[:, i] = targets[:, i] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        mse_loss = F.mse_loss(pred_coords, target_coords_pixel)

        # Hybrid Loss
        total_loss = self.loss_alpha * total_kl_loss + (1 - self.loss_alpha) * mse_loss

        return total_loss

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """예측 (평가 모드)"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ============================================
# ONNX Wrapper
# ============================================

class MPMMobileNetV2HybridModelONNX(nn.Module):
    """ONNX 변환용 래퍼 - 좌표만 반환"""

    def __init__(self, model: MPMMobileNetV2HybridModel):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ONNX 변환용 - 좌표만 반환"""
        output = self.model(images)
        return output['coordinates']  # [B, 8]


# ============================================
# Dataset (이미지 기반)
# ============================================

class DataSet(Dataset):
    """이미지 기반 데이터셋 (MobileNetV2용)"""

    def __init__(self,
                 source_folder: str,
                 labels_file: str,
                 detector: 'PointDetector',
                 mode: str = 'train',
                 config: Dict = None,
                 augment: bool = True,
                 feature_mean: Optional[np.ndarray] = None,
                 feature_std: Optional[np.ndarray] = None,
                 extract_features: bool = False,
                 target_points: List[str] = None):
        self.source_folder = source_folder
        self.detector = detector
        self.mode = mode
        self.config = config or {}
        self.augment = augment and (mode == 'train')
        self.target_points = target_points or ['center', 'floor', 'front', 'side']
        self.num_coordinates = len(self.target_points) * 2

        # 설정
        self.test_id_suffix = str(config['data_split']['test_id_suffix'])
        self.val_ratio = config['data_split']['validation_ratio']
        self.augment_count = config['training']['augmentation']['augment_count'] if self.augment else 0
        self.image_size = tuple(config['learning_model']['architecture']['features']['image_size'])
        self.max_train_images = config.get('data', {}).get('max_train_images', 0)

        # 크롭 증강
        self.crop_config = config['training']['augmentation'].get('crop', {})
        self.crop_enabled = self.crop_config.get('enabled', False) and self.augment

        # 좌표 범위
        self.coord_min_x = -112
        self.coord_max_x = 224
        self.coord_min_y = 0
        self.coord_max_y = 224

        # ImageNet normalization (MobileNetV2 pretrained)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # 데이터 로드
        self.data = []
        self.load_data(labels_file)

        print(f"{mode.upper()} 데이터셋: {len(self.data)}개 샘플 (증강 전)")

    def load_data(self, labels_file: str):
        """레이블 파일 로드"""
        labels_path = os.path.join(self.source_folder, labels_file)

        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        all_train_data = []
        test_data = []

        for idx, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 11:
                continue

            data_id = parts[0]
            filename = parts[2]

            sample = {
                'id': data_id,
                'filename': filename,
                'center_x': int(parts[3]),
                'center_y': int(parts[4]),
                'floor_x': int(parts[5]) if parts[5] != '0' else int(parts[3]),
                'floor_y': int(parts[6]) if parts[6] != '0' else int(parts[4]),
                'front_x': int(parts[7]) if parts[7] != '0' else int(parts[3]),
                'front_y': int(parts[8]) if parts[8] != '0' else int(parts[4]),
                'side_x': int(parts[9]) if parts[9] != '0' else int(parts[3]),
                'side_y': int(parts[10]) if parts[10] != '0' else int(parts[4])
            }

            if data_id.endswith(self.test_id_suffix):
                test_data.append(sample)
            else:
                all_train_data.append(sample)

        print(f"전체 데이터: Train {len(all_train_data)}개, Test {len(test_data)}개")

        if self.max_train_images > 0 and self.mode != 'test':
            all_train_data = all_train_data[:self.max_train_images]
            print(f"max_train_images 설정으로 train 데이터를 {len(all_train_data)}개로 제한")

        if self.mode == 'test':
            self.data = test_data
        else:
            random.seed(self.config['data_split']['random_seed'])
            random.shuffle(all_train_data)
            val_size = int(len(all_train_data) * self.val_ratio)

            if self.mode == 'val':
                self.data = all_train_data[:val_size]
            else:
                self.data = all_train_data[val_size:]

    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        norm_x = (x - self.coord_min_x) / (self.coord_max_x - self.coord_min_x)
        norm_y = (y - self.coord_min_y) / (self.coord_max_y - self.coord_min_y)
        return np.clip(norm_x, 0, 1), np.clip(norm_y, 0, 1)

    def denormalize_coordinates(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        x = norm_x * (self.coord_max_x - self.coord_min_x) + self.coord_min_x
        y = norm_y * (self.coord_max_y - self.coord_min_y) + self.coord_min_y
        return x, y

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리 (BGR → RGB → Normalize)"""
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # [0, 255] → [0, 1]
        image_tensor = torch.FloatTensor(image_rgb).permute(2, 0, 1) / 255.0

        # ImageNet normalization
        image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def __len__(self):
        base_len = len(self.data)
        if self.mode == 'train' and self.augment:
            # 원본 + crop×N + flip + flip+crop×N = 2 + 2×augment_count
            return base_len * (2 + 2 * self.augment_count)
        return base_len

    def __getitem__(self, idx):
        # 원본 인덱스와 증강 타입 계산
        if self.mode == 'train' and self.augment:
            total_variations = 2 + 2 * self.augment_count
            base_idx = idx // total_variations
            variation_idx = idx % total_variations

            # 증강 타입 결정
            if variation_idx == 0:
                apply_flip = False
                apply_crop = False
            elif variation_idx <= self.augment_count:
                apply_flip = False
                apply_crop = True
            elif variation_idx == self.augment_count + 1:
                apply_flip = True
                apply_crop = False
            else:
                apply_flip = True
                apply_crop = True
        else:
            base_idx = idx
            apply_flip = False
            apply_crop = False

        sample = self.data[base_idx]
        image_path = os.path.join(self.source_folder, sample['filename'])
        orig_image = cv2.imread(image_path)

        if orig_image is None:
            print(f"Warning: 이미지 로드 실패 - {image_path}")
            orig_image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        coords_orig = {
            'center_x': sample['center_x'],
            'center_y': sample['center_y'],
            'floor_x': sample['floor_x'],
            'floor_y': sample['floor_y'],
            'front_x': sample['front_x'],
            'front_y': sample['front_y'],
            'side_x': sample['side_x'],
            'side_y': sample['side_y']
        }

        # 1단계: Flip 적용
        if apply_flip:
            orig_image, coords_orig, _ = apply_flip_augmentation(orig_image, coords_orig)

        # 2단계: Crop 또는 Resize 적용
        if apply_crop and self.crop_enabled:
            image_resized, coords_resized = apply_crop_augmentation(
                orig_image, coords_orig, self.image_size, self.crop_config
            )
        else:
            # 리사이즈만
            labels = [
                [coords_orig['center_x'], coords_orig['center_y']],
                [coords_orig['floor_x'], coords_orig['floor_y']],
                [coords_orig['front_x'], coords_orig['front_y']],
                [coords_orig['side_x'], coords_orig['side_y']]
            ]

            image_resized, adjusted_labels = resize_image_with_coordinates(
                self.image_size[0], orig_image, labels
            )

            coords_resized = {}
            for i, key in enumerate(['center', 'floor', 'front', 'side']):
                coords_resized[f'{key}_x'] = adjusted_labels[i][0]
                coords_resized[f'{key}_y'] = adjusted_labels[i][1]

        # 이미지 전처리
        image_tensor = self.preprocess_image(image_resized)

        # 좌표 정규화
        targets = []
        for key in self.target_points:
            x = coords_resized[f'{key}_x']
            y = coords_resized[f'{key}_y']
            norm_x, norm_y = self.normalize_coordinates(x, y)
            targets.extend([norm_x, norm_y])

        targets = np.array(targets, dtype=np.float32)

        return {
            'data': image_tensor,
            'targets': torch.FloatTensor(targets),
            'filename': sample['filename'],
            'id': sample['id'],
            'point_names': self.target_points
        }


# ============================================
# PointDetector Wrapper
# ============================================

class PointDetector:
    """210.learning_new.py 호환 래퍼"""

    def __init__(self, config: Dict, device):
        self.config = config
        self.device = device

        self.target_points = config.get('target_points', ['center', 'floor', 'front', 'side'])
        self.num_output_coords = len(self.target_points) * 2

        # 모델 생성
        model_config = {
            'coord_bins': COORD_BINS,
            'gaussian_sigma': GAUSSIAN_SIGMA,
            'embed_dim': EMBED_DIM,
            'num_heads': NUM_HEADS,
            'ffn_expansion': FFN_EXPANSION,
            'loss_alpha': 0.7,
            'freeze_backbone': False  # 초기에는 freeze 안함
        }

        self.model = MPMMobileNetV2HybridModel(model_config)
        self.model.to(device)

        # 정규화 파라미터 (사용 안함)
        self.feature_mean = None
        self.feature_std = None

    def __call__(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward"""
        outputs = self.model(data)
        outputs['point_names'] = self.target_points
        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        return self.model.compute_loss(outputs, targets, sigma)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        이미지로부터 좌표 예측

        Args:
            image: [H, W, 3] numpy array (BGR)

        Returns:
            coordinates: [8] numpy array (정규화된 좌표)
        """
        # 리사이즈
        image_size = self.config.get('features', {}).get('image_size', IMAGE_SIZE)
        image_resized = cv2.resize(image, tuple(image_size), interpolation=cv2.INTER_LINEAR)

        # 전처리
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.FloatTensor(image_rgb).permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 예측
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)

        coords = outputs['coordinates'].cpu().numpy().flatten()
        return coords


# Alias
PointDetectorDataSet = DataSet


if __name__ == "__main__":
    print("MPM MobileNetV2 Hybrid 모델 정의 완료")
    print(f"입력: RGB 96×96 이미지")
    print(f"아키텍처: MobileNetV2(0.5) + CBAM + Self-Attention")
    print(f"출력: 8개 좌표 (Classification-based)")
