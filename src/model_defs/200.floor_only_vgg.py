# -*- coding: utf-8 -*-
"""
VGG 기반 Floor 좌표 예측 모델 (경량화 버전)
- 96×96 Grayscale 입력 (1채널)
- Custom VGG Backbone (경량화: 32→64→128→256→256)
- Grid Classification 방식 (32 bins)
- Soft Label (Gaussian σ=1.5) + Soft Argmax
- Hybrid Loss (KL + MSE, α=0.7)
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
from util.data_augmentation import apply_crop_augmentation, apply_flip_augmentation
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
SAVE_FILE_NAME = f'floor_only_vgg_light_{IMAGE_SIZE[0]}'

# Grid Classification 설정
COORD_BINS = 32  # 32 bins
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
# VGG Backbone (경량화 버전)
# ============================================

class VGGBlock(nn.Module):
    """VGG 기본 블록 (Conv × 2 + MaxPool)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            out: [B, out_channels, H//2, W//2]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool(x)
        return x


class VGGBackbone(nn.Module):
    """경량화된 VGG Backbone (Grayscale 1채널)

    Architecture:
        Block 1: 1 → 32 → [B, 32, 48, 48]
        Block 2: 32 → 64 → [B, 64, 24, 24]
        Block 3: 64 → 128 → [B, 128, 12, 12]
        Block 4: 128 → 256 → [B, 256, 6, 6]
        Block 5: 256 → 256 → [B, 256, 3, 3]
    """

    def __init__(self):
        super().__init__()

        # VGG Blocks (경량화: 채널 수 절반)
        self.block1 = VGGBlock(1, 32)      # 96×96 → 48×48
        self.block2 = VGGBlock(32, 64)     # 48×48 → 24×24
        self.block3 = VGGBlock(64, 128)    # 24×24 → 12×12
        self.block4 = VGGBlock(128, 256)   # 12×12 → 6×6
        self.block5 = VGGBlock(256, 256)   # 6×6 → 3×3

        self.out_channels = 256

        # 가중치 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화 (Kaiming)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
            features: [B, 256, 3, 3]
        """
        x = self.block1(x)  # [B, 32, 48, 48]
        x = self.block2(x)  # [B, 64, 24, 24]
        x = self.block3(x)  # [B, 128, 12, 12]
        x = self.block4(x)  # [B, 256, 6, 6]
        x = self.block5(x)  # [B, 256, 3, 3]
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
            nn.Linear(in_channels, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
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

        # Classifier (2-layer)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_bins)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B, num_bins]
            coords: [B]
            confidence: [B] - max probability (0~1, 높을수록 확실)
            entropy: [B] - entropy (낮을수록 확실)
        """
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        coords = torch.sum(probs * self.bin_centers.unsqueeze(0), dim=-1)

        # Confidence: max probability
        confidence = torch.max(probs, dim=-1)[0]

        # Entropy: uncertainty measure
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        return logits, coords, confidence, entropy

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

class FloorOnlyVGGModel(nn.Module):
    """VGG 기반 Floor 좌표 예측 모델 - Grayscale → 좌표 예측"""
    MODEL_NAME = "floor_only_vgg_light"

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

        # Stage 1: VGG Backbone (경량화)
        self.backbone = VGGBackbone()
        backbone_channels = self.backbone.out_channels  # 256

        # Stage 2: Feature Compression
        self.compression = FeatureCompression(
            in_channels=backbone_channels,
            out_dim=self.embed_dim
        )

        # Stage 3: Coordinate Heads (2개 - floor만)
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
            dict with 'coordinates', 'pixel_coordinates', 'logits', 'confidence', 'entropy'
        """
        B = images.size(0)

        # Stage 1: Backbone
        features = self.backbone(images)  # [B, 256, 3, 3]

        # Stage 2: Compression
        enhanced = self.compression(features)  # [B, 128]

        # Stage 3: Coordinate Prediction
        all_logits = []
        all_coords = []
        all_confidences = []
        all_entropies = []

        for head in self.coordinate_heads:
            logits, coords, confidence, entropy = head(enhanced)
            all_logits.append(logits)
            all_coords.append(coords)
            all_confidences.append(confidence)
            all_entropies.append(entropy)

        pixel_coordinates = torch.stack(all_coords, dim=1)  # [B, 2] - floor만
        confidences = torch.stack(all_confidences, dim=1)  # [B, 2]
        entropies = torch.stack(all_entropies, dim=1)  # [B, 2]

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
            'pixel_coordinates': pixel_coordinates,
            'confidence': confidences,
            'entropy': entropies
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

class FloorOnlyVGGModelONNX(nn.Module):
    """ONNX 변환용 래퍼 - 좌표, confidence, entropy 반환"""

    def __init__(self, model: FloorOnlyVGGModel, include_confidence: bool = True, include_entropy: bool = True):
        super().__init__()
        self.model = model
        self.include_confidence = include_confidence
        self.include_entropy = include_entropy

    def forward(self, images: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """ONNX 변환용 - 좌표 (및 confidence, entropy) 반환

        Args:
            images: [B, 1, 96, 96]

        Returns:
            if include_confidence=False and include_entropy=False:
                coordinates: [B, 2] - floor만
            if include_confidence=True and include_entropy=False:
                (coordinates, confidence): ([B, 2], [B, 2])
            if include_confidence=True and include_entropy=True:
                (coordinates, confidence, entropy): ([B, 2], [B, 2], [B, 2])
        """
        output = self.model(images)

        if self.include_confidence and self.include_entropy:
            return output['coordinates'], output['confidence'], output['entropy']
        elif self.include_confidence:
            return output['coordinates'], output['confidence']
        else:
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
        self.target_points = target_points or ['floor']
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
            print(f"  - 좌우 반전(Flip) 증강: 활성화 (항상 적용)")
            print(f"  - 증강 패턴: 원본(1) + crop×{self.augment_count} + flip(1) + flip+crop×{self.augment_count}")
            total_samples = len(self.data) * (2 + 2 * self.augment_count)
            print(f"증강 후 총 샘플 수: {total_samples}개 (원본 {len(self.data)}개 × {2 + 2 * self.augment_count})")
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
            # 0: 원본
            # 1 ~ augment_count: 원본 + crop
            # augment_count+1: flip
            # augment_count+2 ~ end: flip + crop

            if variation_idx == 0:
                # 원본
                apply_flip = False
                apply_crop = False
            elif variation_idx <= self.augment_count:
                # 원본 + crop
                apply_flip = False
                apply_crop = True
            elif variation_idx == self.augment_count + 1:
                # flip만
                apply_flip = True
                apply_crop = False
            else:
                # flip + crop
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

        # floor 좌표만 사용
        coords_orig = {
            'floor_x': sample['floor_x'],
            'floor_y': sample['floor_y']
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

        self.target_points = config.get('target_points', ['floor'])
        self.num_output_coords = len(self.target_points) * 2

        # 모델 생성
        model_config = {
            'coord_bins': COORD_BINS,
            'gaussian_sigma': GAUSSIAN_SIGMA,
            'embed_dim': EMBED_DIM,
            'loss_alpha': 0.7
        }

        self.model = FloorOnlyVGGModel(model_config)
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
            coordinates: [2] numpy array (정규화된 좌표)
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
    print("VGG 기반 Floor 좌표 예측 모델 정의 완료")
    print(f"입력: Grayscale 96×96 이미지 (1채널)")
    print(f"아키텍처: 경량화된 VGG (32→64→128→256→256)")
    print(f"출력: 2개 좌표 (Classification-based, 32 bins)")
    print(f"파라미터 수: 약 1.9M (경량화)")
