# -*- coding: utf-8 -*-
"""
MPM MoveNet 아키텍처 구현
- Google MoveNet의 핵심 구조 적용 (특징 사전 추출 방식)
- MobileNetV2 + FPN Backbone으로 특징 추출
- 4개 예측 헤드: Center Heatmap, Keypoint Heatmap, Regression Field, Offset Field
- 특징 사전 추출 방식 (27648-dim FPN features)
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
from util.data_augmentation import apply_crop_augmentation
from util.image_resize import resize_image_with_coordinates

# matplotlib 경고 억제
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================
# 모델 설정 상수
# ============================================
TARGET_POINTS = ['center', 'floor', 'front', 'side']
USE_FPD_ARCHITECTURE = False  # MoveNet은 회귀 기반
IMAGE_SIZE = [96, 96]  # 이미지 크기
GRID_SIZE = 24  # Output stride 4: 96 / 4 = 24
HEATMAP_SIGMA = 2.0
SAVE_FILE_NAME = f'mpm_100_movenet_{IMAGE_SIZE[0]}'
USE_AUTOENCODER = False  # MobileNetV2+FPN 특징 사용


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
        }
    }


def get_input_dim():
    """모델의 기대 입력 차원 반환 (ONNX 변환용)

    Returns:
        int: 입력 특징 벡터 차원 (48 * 24 * 24 = 27648)
    """
    return 48 * GRID_SIZE * GRID_SIZE  # 27648


# ============================================
# MobileNetV2 Building Blocks
# ============================================

class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block"""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 Backbone for Feature Extraction"""

    def __init__(self, width_mult=1.0):
        super().__init__()

        # Building first layer
        input_channel = int(32 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )

        # Building inverted residual blocks
        # t: expansion factor, c: output channels, n: repeat times, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   # stage 1
            [6, 24, 2, 2],   # stage 2 - stride 4
            [6, 32, 3, 2],   # stage 3 - stride 8
            [6, 64, 4, 2],   # stage 4 - stride 16
            [6, 96, 3, 1],   # stage 5 - stride 16
            [6, 160, 3, 2],  # stage 6 - stride 32
            [6, 320, 1, 1],  # stage 7 - stride 32
        ]

        # Build stages
        self.stage1 = self._make_stage(input_channel, inverted_residual_setting[0])
        self.stage2 = self._make_stage(16, inverted_residual_setting[1])  # stride 4
        self.stage3 = self._make_stage(24, inverted_residual_setting[2])  # stride 8
        self.stage4 = self._make_stage(32, inverted_residual_setting[3])  # stride 16
        self.stage5 = self._make_stage(64, inverted_residual_setting[4])  # stride 16
        self.stage6 = self._make_stage(96, inverted_residual_setting[5])  # stride 32
        self.stage7 = self._make_stage(160, inverted_residual_setting[6]) # stride 32

    def _make_stage(self, input_channel, setting):
        """Build a stage with inverted residual blocks"""
        t, c, n, s = setting
        layers = []
        output_channel = c

        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(InvertedResidual(input_channel, output_channel, stride, t))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns multi-scale features for FPN
        """
        x = self.features(x)  # stride 2

        c1 = self.stage1(x)   # stride 2, 16 channels
        c2 = self.stage2(c1)  # stride 4, 24 channels
        c3 = self.stage3(c2)  # stride 8, 32 channels
        c4 = self.stage4(c3)  # stride 16, 64 channels
        c5 = self.stage5(c4)  # stride 16, 96 channels

        return {
            'c2': c2,  # stride 4
            'c3': c3,  # stride 8
            'c4': c4,  # stride 16
            'c5': c5,  # stride 16
        }


# ============================================
# Feature Pyramid Network (FPN)
# ============================================

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""

    def __init__(self, in_channels_list: List[int], out_channels: int = 48):
        super().__init__()

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_c2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[3], out_channels, 1)

        # Top-down pathway (3x3 conv for fusion)
        self.fpn_c5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fpn_c4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fpn_c3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fpn_c2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Top-down feature fusion

        Args:
            features: Dict with c2, c3, c4, c5

        Returns:
            High-resolution feature map (stride 4)
        """
        c2, c3, c4, c5 = features['c2'], features['c3'], features['c4'], features['c5']

        # Lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3)
        p2 = self.lateral_c2(c2)

        # Top-down fusion
        p5 = self.fpn_c5(p5)
        p4 = self.fpn_c4(p4 + F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False))
        p3 = self.fpn_c3(p3 + F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False))
        p2 = self.fpn_c2(p2 + F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False))

        return p2  # Output stride 4


# ============================================
# MoveNet Prediction Heads
# ============================================

class CenterHeatmapHead(nn.Module):
    """Person/Object Center Heatmap Prediction Head"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),  # 1 channel for center
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            center_heatmap: [B, 1, H, W]
        """
        return self.head(x)


class KeypointHeatmapHead(nn.Module):
    """Per-Keypoint Heatmap Prediction Head"""

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_keypoints, 1),  # N channels for N keypoints
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            keypoint_heatmaps: [B, N, H, W]
        """
        return self.head(x)


class RegressionFieldHead(nn.Module):
    """Keypoint Regression Field from Center"""

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = num_keypoints
        # Each keypoint needs 2 values (dx, dy from center)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_keypoints * 2, 1),  # 2N channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            regression_field: [B, 2N, H, W]
        """
        return self.head(x)


class OffsetFieldHead(nn.Module):
    """2D Per-Keypoint Offset Field for Sub-pixel Accuracy"""

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = num_keypoints
        # Each keypoint needs 2 values (offset_x, offset_y)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_keypoints * 2, 1),  # 2N channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            offset_field: [B, 2N, H, W]
        """
        return self.head(x)


# ============================================
# MoveNet Main Model
# ============================================

class MoveNetModel(nn.Module):
    """MoveNet inspired architecture - 사전 추출된 FPN 특징을 입력으로 받음"""
    MODEL_NAME = "mpm_movenet"

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # 좌표 설정
        self.x_range = (-112, 224)
        self.y_range = (0, 224)
        self.num_keypoints = 4  # center, floor, front, side
        self.num_coordinates = 8  # 4 points × 2
        self.point_names = ['center', 'floor', 'front', 'side']
        self.image_size = IMAGE_SIZE[0]
        self.grid_size = GRID_SIZE

        # 입력: 사전 추출된 FPN 특징 [48, 24, 24] = 27648
        self.feature_dim = 48 * GRID_SIZE * GRID_SIZE

        # Prediction Heads
        self.center_head = CenterHeatmapHead(in_channels=48)
        self.keypoint_head = KeypointHeatmapHead(in_channels=48, num_keypoints=self.num_keypoints)
        self.regression_head = RegressionFieldHead(in_channels=48, num_keypoints=self.num_keypoints)
        self.offset_head = OffsetFieldHead(in_channels=48, num_keypoints=self.num_keypoints)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, 27648] 사전 추출된 FPN 특징 (flattened)

        Returns:
            Dict containing:
                - center_heatmap: [B, 1, H/4, W/4]
                - keypoint_heatmaps: [B, N, H/4, W/4]
                - regression_field: [B, 2N, H/4, W/4]
                - offset_field: [B, 2N, H/4, W/4]
                - coordinates: [B, 8] normalized coordinates
                - pixel_coordinates: [B, 8] pixel coordinates
        """
        B = features.size(0)

        # Reshape flattened features to [B, 48, 24, 24]
        fpn_features = features.view(B, 48, self.grid_size, self.grid_size)

        # Predictions
        center_heatmap = self.center_head(fpn_features)  # [B, 1, H/4, W/4]
        keypoint_heatmaps = self.keypoint_head(fpn_features)  # [B, N, H/4, W/4]
        regression_field = self.regression_head(fpn_features)  # [B, 2N, H/4, W/4]
        offset_field = self.offset_head(fpn_features)  # [B, 2N, H/4, W/4]

        # Extract keypoint coordinates from heatmaps
        pixel_coordinates = self._extract_coordinates_from_heatmaps(
            keypoint_heatmaps, offset_field, center_heatmap
        )

        # Normalize coordinates
        norm_coordinates = torch.zeros_like(pixel_coordinates)
        for i in range(self.num_coordinates):
            if i % 2 == 0:  # x
                norm_coordinates[:, i] = (pixel_coordinates[:, i] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
            else:  # y
                norm_coordinates[:, i] = (pixel_coordinates[:, i] - self.y_range[0]) / (self.y_range[1] - self.y_range[0])

        return {
            'center_heatmap': center_heatmap,
            'keypoint_heatmaps': keypoint_heatmaps,
            'regression_field': regression_field,
            'offset_field': offset_field,
            'coordinates': norm_coordinates,
            'pixel_coordinates': pixel_coordinates,
        }

    def _extract_coordinates_from_heatmaps(self, keypoint_heatmaps: torch.Tensor,
                                           offset_field: torch.Tensor,
                                           center_heatmap: torch.Tensor) -> torch.Tensor:
        """
        Extract keypoint coordinates from heatmaps

        Args:
            keypoint_heatmaps: [B, N, H, W]
            offset_field: [B, 2N, H, W]
            center_heatmap: [B, 1, H, W]

        Returns:
            coordinates: [B, 2N] pixel coordinates
        """
        B, N, H, W = keypoint_heatmaps.shape

        # Find max locations in heatmaps
        heatmaps_flat = keypoint_heatmaps.view(B, N, -1)  # [B, N, H*W]
        max_vals, max_indices = torch.max(heatmaps_flat, dim=2)  # [B, N]

        # Convert flat indices to 2D coordinates
        max_y = (max_indices // W).float()  # [B, N]
        max_x = (max_indices % W).float()   # [B, N]

        # Extract offsets at max locations
        coords = []
        for b in range(B):
            for n in range(N):
                y_idx = int(max_y[b, n].item())
                x_idx = int(max_x[b, n].item())

                # Get offset
                offset_x = offset_field[b, n*2, y_idx, x_idx]
                offset_y = offset_field[b, n*2+1, y_idx, x_idx]

                # Apply offset and scale to original image size
                scale = self.image_size / self.grid_size
                final_x = (max_x[b, n] + offset_x) * scale
                final_y = (max_y[b, n] + offset_y) * scale

                # Adjust to coordinate range
                final_x = final_x + self.x_range[0]
                final_y = final_y + self.y_range[0]

                coords.extend([final_x, final_y])

        return torch.stack(coords).view(B, -1)

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor,
                    target_heatmaps: Optional[Dict[str, torch.Tensor]] = None,
                    sigma: float = 1.0) -> torch.Tensor:
        """
        Compute MoveNet-style loss

        Args:
            outputs: Model outputs
            targets: [B, 2N] normalized target coordinates
            target_heatmaps: Optional pre-computed heatmaps

        Returns:
            total_loss: Combined loss
        """
        # Coordinate loss (fallback)
        pred_coords = outputs['pixel_coordinates']

        target_coords_pixel = torch.zeros_like(targets)
        for i in range(self.num_coordinates):
            if i % 2 == 0:
                target_coords_pixel[:, i] = targets[:, i] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            else:
                target_coords_pixel[:, i] = targets[:, i] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        coord_loss = F.mse_loss(pred_coords, target_coords_pixel)

        # Heatmap losses (if target heatmaps provided)
        heatmap_loss = 0
        if target_heatmaps is not None:
            # Center heatmap loss
            if 'center_heatmap' in target_heatmaps:
                center_loss = F.mse_loss(outputs['center_heatmap'], target_heatmaps['center_heatmap'])
                heatmap_loss += center_loss

            # Keypoint heatmap loss
            if 'keypoint_heatmaps' in target_heatmaps:
                kp_loss = F.mse_loss(outputs['keypoint_heatmaps'], target_heatmaps['keypoint_heatmaps'])
                heatmap_loss += kp_loss * 2.0  # Weight more

        total_loss = coord_loss + heatmap_loss
        return total_loss

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ============================================
# Feature Extractor (MobileNetV2 특징 추출)
# ============================================

class MoveNetFeatureExtractor:
    """MobileNetV2 백본으로 특징 사전 추출"""

    def __init__(self, image_size: Tuple[int, int] = (96, 96), device: str = 'cpu'):
        self.image_size = image_size
        self.device = device

        # MobileNetV2 백본 생성 (FPN까지)
        self.backbone = MobileNetV2Backbone(width_mult=1.0)
        self.fpn = FPN(in_channels_list=[24, 32, 64, 96], out_channels=48)

        # 평가 모드로 설정 (특징 추출 시)
        self.backbone.eval()
        self.fpn.eval()
        self.backbone.to(device)
        self.fpn.to(device)

        # 특징 추출 시 gradient 계산 안함
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.fpn.parameters():
            param.requires_grad = False

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 MobileNetV2 + FPN 특징 추출

        Args:
            image: [H, W, 3] numpy array (BGR)

        Returns:
            features: [48*24*24] flatten된 특징 벡터
        """
        # BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Transpose to [C, H, W]
        image = np.transpose(image, (2, 0, 1))

        # Tensor 변환
        image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)

        # 특징 추출 (gradient 없이)
        with torch.no_grad():
            backbone_features = self.backbone(image_tensor)
            fpn_features = self.fpn(backbone_features)  # [1, 48, 24, 24]

        # Flatten하여 반환
        features = fpn_features.cpu().numpy().flatten()  # [48*24*24 = 27648]

        return features


# ============================================
# Dataset
# ============================================

def generate_gaussian_heatmap(center_x: float, center_y: float,
                              grid_size: int, sigma: float = 2.0) -> np.ndarray:
    """
    Generate Gaussian heatmap for a keypoint

    Args:
        center_x, center_y: Keypoint location in grid coordinates
        grid_size: Heatmap size
        sigma: Gaussian sigma

    Returns:
        heatmap: [grid_size, grid_size]
    """
    x = np.arange(0, grid_size, 1, np.float32)
    y = np.arange(0, grid_size, 1, np.float32)
    y = y[:, np.newaxis]

    # Gaussian heatmap
    heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    return heatmap


class DataSet(Dataset):
    """MoveNet 데이터셋 - MobileNetV2+FPN 특징 사전 추출"""

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

        # Feature extractor
        self.feature_extractor = detector.feature_extractor

        # 설정
        self.test_id_suffix = str(config['data_split']['test_id_suffix'])
        self.val_ratio = config['data_split']['validation_ratio']
        self.augment_count = config['training']['augmentation']['augment_count'] if self.augment else 0
        self.image_size = IMAGE_SIZE[0]
        self.grid_size = GRID_SIZE
        self.max_train_images = config.get('data', {}).get('max_train_images', 0)

        # 크롭 증강
        self.crop_config = config['training']['augmentation'].get('crop', {})
        self.crop_enabled = self.crop_config.get('enabled', False) and self.augment

        # 좌표 범위
        self.coord_min_x = -112
        self.coord_max_x = 224
        self.coord_min_y = 0
        self.coord_max_y = 224

        # 데이터 로드
        self.data = []
        self.load_data(labels_file)

        # 특징 및 타겟 저장 (210.learning_new.py 호환)
        self.features = []
        self.targets = []
        self.metadata = []

        self.precompute_features()

        print(f"{mode.upper()} 데이터셋: {len(self.features)}개 샘플")

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

    def precompute_features(self):
        """모든 데이터의 특징 미리 추출"""
        print(f"{self.mode.upper()} 데이터셋 MobileNetV2+FPN 특징 추출 중...")

        for sample in tqdm(self.data, desc="특징 추출"):
            image_path = os.path.join(self.source_folder, sample['filename'])
            orig_image = cv2.imread(image_path)

            if orig_image is None:
                print(f"Warning: 이미지 로드 실패 - {image_path}")
                orig_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

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

            # 리사이즈
            labels = [
                [coords_orig['center_x'], coords_orig['center_y']],
                [coords_orig['floor_x'], coords_orig['floor_y']],
                [coords_orig['front_x'], coords_orig['front_y']],
                [coords_orig['side_x'], coords_orig['side_y']]
            ]

            image_resized, adjusted_labels = resize_image_with_coordinates(
                self.image_size, orig_image, labels
            )

            coords_resized = {}
            for i, key in enumerate(['center', 'floor', 'front', 'side']):
                coords_resized[f'{key}_x'] = adjusted_labels[i][0]
                coords_resized[f'{key}_y'] = adjusted_labels[i][1]

            # 특징 추출 (MobileNetV2 + FPN)
            features = self.feature_extractor.extract_features(image_resized)

            # 좌표 정규화
            targets = []
            for key in self.target_points:
                x = coords_resized[f'{key}_x']
                y = coords_resized[f'{key}_y']
                norm_x, norm_y = self.normalize_coordinates(x, y)
                targets.extend([norm_x, norm_y])

            targets = np.array(targets, dtype=np.float32)

            self.features.append(torch.FloatTensor(features))
            self.targets.append(torch.FloatTensor(targets))
            self.metadata.append({
                'filename': sample['filename'],
                'id': sample['id'],
                'augmented': False
            })

            # 증강
            if self.mode == 'train' and self.augment and self.crop_enabled:
                for aug_idx in range(self.augment_count):
                    cropped_img, coords_cropped = apply_crop_augmentation(
                        orig_image, coords_orig, (self.image_size, self.image_size), self.crop_config
                    )

                    features_aug = self.feature_extractor.extract_features(cropped_img)

                    targets_aug = []
                    for key in self.target_points:
                        x = coords_cropped[f'{key}_x']
                        y = coords_cropped[f'{key}_y']
                        norm_x, norm_y = self.normalize_coordinates(x, y)
                        targets_aug.extend([norm_x, norm_y])

                    targets_aug = np.array(targets_aug, dtype=np.float32)

                    self.features.append(torch.FloatTensor(features_aug))
                    self.targets.append(torch.FloatTensor(targets_aug))
                    self.metadata.append({
                        'filename': sample['filename'],
                        'id': sample['id'],
                        'augmented': True
                    })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'data': self.features[idx].clone(),
            'targets': self.targets[idx].clone(),
            'filename': self.metadata[idx]['filename'],
            'id': self.metadata[idx]['id'],
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

        # Feature extractor (이미지 전처리만)
        self.feature_extractor = MoveNetFeatureExtractor(image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]))

        # 모델 생성
        model_config = {}
        self.model = MoveNetModel(model_config)
        self.model.to(device)

        # 정규화 파라미터
        self.feature_mean = None
        self.feature_std = None

    def __call__(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward"""
        outputs = self.model(data)
        # point_names 추가 (210.learning_new.py 호환)
        outputs['point_names'] = self.target_points
        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        return self.model.compute_loss(outputs, targets, sigma=sigma)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        이미지로부터 좌표 예측

        Args:
            image: [H, W, 3] numpy array (BGR)

        Returns:
            coordinates: [8] numpy array (정규화된 좌표)
        """
        # 특징 추출
        features = self.feature_extractor.extract_features(image)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # 예측
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)

        # 좌표 반환
        coords = outputs['coordinates'].cpu().numpy().flatten()
        return coords


# Alias
PointDetectorDataSet = DataSet


if __name__ == "__main__":
    print("MPM MoveNet 모델 정의 완료")
    print(f"입력: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB 이미지 -> 27648-dim FPN 특징")
    print(f"특징 추출: MobileNetV2 + FPN (사전 추출)")
    print(f"모델 입력: [B, 27648] 플랫 특징 벡터")
    print(f"아키텍처: 4개 MoveNet 예측 헤드 (Center/Keypoint Heatmap, Regression, Offset)")
    print(f"출력: 4개 좌표점 (center, floor, front, side)")
