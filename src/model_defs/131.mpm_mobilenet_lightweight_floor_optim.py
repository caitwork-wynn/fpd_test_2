# -*- coding: utf-8 -*-
"""
MPM MobileNetV2 Lightweight Optimized 모델 (Self-Attention 제거 버전)
- 96×96 Grayscale 입력 (1채널)
- Custom MobileNetV2(0.5) Backbone (torchvision 독립)
- CBAM Attention (Channel + Spatial)
- Self-Attention 제거 (단순 FC 처리)
- Classification-based Coordinate Prediction (32 bins)
- 210.learning_new.py 호환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
from util.data_augmentation import apply_crop_augmentation
from util.image_resize import resize_image_with_coordinates

# matplotlib 경고 억제
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================
# 모델 설정 상수
# ============================================
TARGET_POINTS = ['floor']
USE_FPD_ARCHITECTURE = True
IMAGE_SIZE = [96, 96]
GRID_SIZE = 7
USE_AUTOENCODER = False
ENCODER_PATH = None
ENCODER_LATENT_DIM = 0
SAVE_FILE_NAME = f'mpm_lightweight_floor_optim_{IMAGE_SIZE[0]}'

# 모델 설정 (경량화)
COORD_BINS = 32  # 64 → 32 bins (경량화)
GAUSSIAN_SIGMA = 1.5
EMBED_DIM = 128


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
        tuple: (channels, height, width) - (1, 96, 96) Grayscale
    """
    return (1, IMAGE_SIZE[0], IMAGE_SIZE[1])


# ============================================
# Custom MobileNetV2 Backbone (Grayscale)
# ============================================

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block (MobileNetV2 핵심 블록)"""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, expansion: int = 6):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = int(in_channels * expansion)

        layers = []

        # Expansion (1×1 Conv)
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise (3×3 Conv)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Projection (1×1 Conv, Linear activation)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class CustomMobileNetV2Backbone(nn.Module):
    """Custom MobileNetV2 Backbone (Grayscale 1채널, Width Mult 0.5)"""

    def __init__(self, width_mult: float = 0.5):
        super().__init__()

        def _make_divisible(v, divisor=8):
            """채널 수를 8의 배수로 조정"""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # 설정 (width_mult 적용)
        # [t, c, n, s] = [expansion, channels, repeat, stride]
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   # Block 1
            [6, 24, 2, 2],   # Block 2
            [6, 32, 3, 2],   # Block 3
            [6, 64, 4, 2],   # Block 4
            [6, 96, 3, 1],   # Block 5
            [6, 160, 3, 2],  # Block 6
            [6, 320, 1, 1],  # Block 7
        ]

        # Width multiplier 적용
        input_channel = _make_divisible(32 * width_mult)

        # 첫 Conv Layer (1채널 Grayscale 입력)
        self.features = [
            nn.Sequential(
                nn.Conv2d(1, input_channel, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True)
            )
        ]

        # Inverted Residual Blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidualBlock(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        # 마지막 Conv Layer
        last_channel = _make_divisible(1280 * width_mult)
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, 1, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True)
            )
        )

        self.features = nn.Sequential(*self.features)
        self.out_channels = last_channel

        # 가중치 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화 (Kaiming)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 96, 96] Grayscale
        Returns:
            features: [B, 640, 3, 3]
        """
        return self.features(x)


# ============================================
# CBAM Attention Modules
# ============================================

class ChannelAttention(nn.Module):
    """Channel Attention Module (ONNX compatible)"""

    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

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
        B, C, _, _ = x.size()

        # Global average pooling
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
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention: [B, C, H, W]
        """
        # Channel-wise average pooling (ONNX opset 11 compatible)
        # torch.mean with dim parameter causes issues in opset 11
        # Use sum + division instead
        avg_pool = torch.sum(x, dim=1, keepdim=True) / x.size(1)  # [B, 1, H, W]

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
# Coordinate Classification Head
# ============================================

class CoordinateClassificationHead(nn.Module):
    """분류 기반 좌표 예측 헤드 (32 bins)"""

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

        # Classifier (경량화)
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

class MPMMobileNetLightweightModel(nn.Module):
    """MobileNetV2 Lightweight Optimized 모델 - Grayscale → 좌표 예측"""
    MODEL_NAME = "mpm_mobilenet_lightweight_optim"

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # 좌표 설정
        self.x_range = (-112, 224)
        self.y_range = (0, 224)
        self.num_coordinates = 2  # 1 point × 2 (floor만)
        self.point_names = ['floor']

        # 하이퍼파라미터
        self.coord_bins = config.get('coord_bins', COORD_BINS)
        self.gaussian_sigma = config.get('gaussian_sigma', GAUSSIAN_SIGMA)
        self.embed_dim = config.get('embed_dim', EMBED_DIM)
        self.loss_alpha = config.get('loss_alpha', 0.7)  # KL vs MSE weight

        # Stage 1: Custom MobileNetV2 Backbone (Grayscale)
        self.backbone = CustomMobileNetV2Backbone(width_mult=0.5)
        backbone_channels = self.backbone.out_channels  # 640

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

        # Stage 4: Coordinate Heads (2개 - floor만)
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
            images: [B, 1, 96, 96] Grayscale images

        Returns:
            dict with 'coordinates', 'pixel_coordinates', 'logits'
        """
        B = images.size(0)

        # Stage 1: Backbone
        features = self.backbone(images)  # [B, 640, 3, 3]

        # Stage 2: CBAM Attention
        features = self.cbam(features)  # [B, 640, 3, 3]

        # Stage 3: Compression (Self-Attention 제거됨)
        enhanced = self.compression(features)  # [B, 128]

        # Stage 4: Coordinate Prediction
        all_logits = []
        all_coords = []

        for head in self.coordinate_heads:
            logits, coords = head(enhanced)
            all_logits.append(logits)
            all_coords.append(coords)

        pixel_coordinates = torch.stack(all_coords, dim=1)  # [B, 2] - floor만

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

class MPMMobileNetLightweightModelONNX(nn.Module):
    """ONNX 변환용 래퍼 - 좌표만 반환"""

    def __init__(self, model: MPMMobileNetLightweightModel):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ONNX 변환용 - 좌표만 반환"""
        output = self.model(images)
        return output['coordinates']  # [B, 2] - floor만


# ============================================
# Dataset (Grayscale 이미지 기반)
# ============================================

class DataSet(Dataset):
    """Grayscale 이미지 기반 데이터셋"""

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

        # image_size 안전하게 가져오기 (200.learning.py 호환)
        try:
            self.image_size = tuple(config['learning_model']['architecture']['features']['image_size'])
        except (KeyError, TypeError):
            # 200.learning.py에서는 architecture 섹션이 없으므로 모듈 상수 사용
            self.image_size = tuple(IMAGE_SIZE)  # (96, 96)

        self.max_train_images = config.get('data', {}).get('max_train_images', 0)

        # 크롭 증강
        self.crop_config = config['training']['augmentation'].get('crop', {})
        self.crop_enabled = self.crop_config.get('enabled', False) and self.augment

        # 좌표 범위
        self.coord_min_x = -112
        self.coord_max_x = 224
        self.coord_min_y = 0
        self.coord_max_y = 224

        # Grayscale normalization (통계 기반)
        # 일반적인 Grayscale 이미지 통계값 사용
        self.mean = torch.tensor([0.449]).view(1, 1, 1)
        self.std = torch.tensor([0.226]).view(1, 1, 1)

        # 데이터 로드
        self.data = []
        self.load_data(labels_file)

        print(f"{mode.upper()} 데이터셋: {len(self.data)}개 샘플 (증강 전)")

        # 데이터 구조 출력
        self.print_dataset_info()

    def load_data(self, labels_file: str):
        """레이블 파일 로드"""
        labels_path = os.path.join(self.source_folder, labels_file)

        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        all_train_data = []
        test_data = []
        skipped_count = 0

        for idx, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 5:
                continue

            data_id = parts[0]
            filename = parts[2]

            # 이미지 파일 존재 여부 확인
            image_path = os.path.join(self.source_folder, filename)
            if not os.path.exists(image_path):
                if skipped_count < 10:  # 처음 10개만 경고 출력
                    print(f"  경고: 이미지 파일 없음 - {filename}")
                skipped_count += 1
                continue

            # 5개 컬럼: ID,class,파일명,floor_x,floor_y
            sample = {
                'id': data_id,
                'filename': filename,
                'floor_x': float(parts[3]),
                'floor_y': float(parts[4])
            }

            if data_id.endswith(self.test_id_suffix):
                test_data.append(sample)
            else:
                all_train_data.append(sample)

        if skipped_count > 0:
            print(f"  이미지 없어서 스킵된 레코드: {skipped_count}개")
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

    def print_dataset_info(self):
        """데이터셋 구조 정보 출력"""
        print("\n" + "="*50)
        print(f"Dataset 구조 정보 [{self.mode.upper()}]")
        print("="*50)
        print(f"전체 샘플 수: {len(self.data)}개")
        print(f"이미지 크기: {self.image_size}")

        # 사용 가능한 포인트 감지
        if len(self.data) > 0:
            sample = self.data[0]
            available_points = []
            for point in ['center', 'floor', 'front', 'side']:
                if f'{point}_x' in sample and f'{point}_y' in sample:
                    # 0이 아닌 실제 좌표가 있는지 확인
                    if sample[f'{point}_x'] != 0 or sample[f'{point}_y'] != 0:
                        available_points.append(point)

            print(f"사용 가능한 포인트: {available_points}")
            print(f"좌표 개수: {len(available_points) * 2}개 ({len(available_points)} points × 2)")

        print(f"좌표 범위: x=[{self.coord_min_x}, {self.coord_max_x}], y=[{self.coord_min_y}, {self.coord_max_y}]")

        # 증강 설정
        if self.mode == 'train' and self.augment:
            print(f"증강 설정: 활성화됨 (count={self.augment_count})")
            if self.crop_enabled:
                print(f"  - 크롭 증강: 활성화")
            total_samples = len(self.data) * (1 + self.augment_count)
            print(f"증강 후 총 샘플 수: {total_samples}개")
        else:
            print(f"증강 설정: 비활성화")

        print("="*50 + "\n")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리 (BGR → Grayscale → Normalize)"""
        # BGR → Grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # [0, 255] → [0, 1]
        image_tensor = torch.FloatTensor(image_gray).unsqueeze(0) / 255.0

        # Normalization
        image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def __len__(self):
        base_len = len(self.data)
        if self.mode == 'train' and self.augment:
            return base_len * (1 + self.augment_count)
        return base_len

    def __getitem__(self, idx):
        # 원본 인덱스와 증강 여부 계산
        if self.mode == 'train' and self.augment:
            base_idx = idx // (1 + self.augment_count)
            is_augmented = (idx % (1 + self.augment_count)) > 0
        else:
            base_idx = idx
            is_augmented = False

        sample = self.data[base_idx]
        image_path = os.path.join(self.source_folder, sample['filename'])
        orig_image = cv2.imread(image_path)

        if orig_image is None:
            print(f"Warning: 이미지 로드 실패 - {image_path}")
            orig_image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        # floor 좌표만 사용
        coords_orig = {
            'floor_x': sample['floor_x'],
            'floor_y': sample['floor_y']
        }

        # 증강 적용
        if is_augmented and self.crop_enabled:
            image_resized, coords_resized = apply_crop_augmentation(
                orig_image, coords_orig, self.image_size, self.crop_config
            )
        else:
            # 리사이즈만 (floor만)
            labels = [
                [coords_orig['floor_x'], coords_orig['floor_y']]
            ]

            image_resized, adjusted_labels = resize_image_with_coordinates(
                self.image_size[0], orig_image, labels
            )

            coords_resized = {
                'floor_x': adjusted_labels[0][0],
                'floor_y': adjusted_labels[0][1]
            }

        # 이미지 전처리 (Grayscale)
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
            'loss_alpha': 0.7
        }

        self.model = MPMMobileNetLightweightModel(model_config)
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

        # 전처리 (Grayscale)
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        image_tensor = torch.FloatTensor(image_gray).unsqueeze(0) / 255.0

        # Normalization
        mean = torch.tensor([0.449]).view(1, 1, 1)
        std = torch.tensor([0.226]).view(1, 1, 1)
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
    print("MPM MobileNetV2 Lightweight Optimized 모델 정의 완료")
    print(f"입력: Grayscale 96×96 이미지 (1채널)")
    print(f"아키텍처: Custom MobileNetV2(0.5) + CBAM (Self-Attention 제거)")
    print(f"출력: 8개 좌표 (Classification-based, 32 bins)")
    print(f"최적화: Self-Attention 제거로 파라미터 감소")
