# -*- coding: utf-8 -*-
"""
MPM Cross-Self Attention VPI 모델 (재설계 버전)
- 특징 추출과 AI 모델 완전 분리
- 16x16 Grayscale 인코더 + 16x16 Canny 인코더 + VPI ORB 상위 5개
- 순수 PyTorch 모델로 ONNX 변환 가능
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

# VPI import 시도
VPI_AVAILABLE = False
try:
    import vpi
    VPI_AVAILABLE = True
except Exception as e:
    print(f"Warning: VPI import failed ({e}). Using OpenCV ORB as fallback.")
    VPI_AVAILABLE = False


# ============================================
# 모델 설정 상수
# ============================================
TARGET_POINTS = ['center', 'floor', 'front', 'side']
USE_FPD_ARCHITECTURE = True
IMAGE_SIZE = [64, 64]
GRID_SIZE = 7
USE_AUTOENCODER = True
ENCODER_PATH = '../model/autoencoder_16x16_best.pth'
ENCODER_LATENT_DIM = 32
SAVE_FILE_NAME = f'mpm_cross_self_attention_vpi_{IMAGE_SIZE[0]}'

# VPI/ORB 설정
VPI_MAX_FEATURES = 5
VPI_MAX_PYR_LEVELS = 3
VPI_INTENSITY_THRESHOLD = 145
COORD_BINS = 64
GAUSSIAN_SIGMA = 2.0


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


# ============================================
# VPI/OpenCV ORB 검출기
# ============================================

class ORBDetector:
    """VPI 우선, 실패 시 OpenCV로 fallback하는 ORB 검출기"""

    def __init__(self, max_features=5, max_pyr_levels=3, intensity_threshold=145):
        self.max_features = max_features
        self.max_pyr_levels = max_pyr_levels
        self.intensity_threshold = intensity_threshold
        self.use_vpi = False
        self.vpi_backend = None

        # VPI 초기화 시도
        if VPI_AVAILABLE:
            try:
                for backend in [vpi.Backend.PVA, vpi.Backend.CUDA, vpi.Backend.CPU]:
                    try:
                        self.vpi_backend = backend
                        self.use_vpi = True
                        print(f"VPI ORB initialized with backend: {backend}")
                        break
                    except:
                        continue
            except Exception as e:
                print(f"VPI initialization failed: {e}")
                self.use_vpi = False

        # OpenCV fallback
        if not self.use_vpi:
            self.orb = cv2.ORB_create(
                nfeatures=max_features * 10,
                scaleFactor=1.2,
                nlevels=max_pyr_levels
            )
            print("Using OpenCV ORB as fallback")

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """ORB 특징점 검출 및 설명자 계산"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.use_vpi:
            return self._detect_vpi(gray)
        else:
            return self._detect_opencv(gray)

    def _detect_vpi(self, gray: np.ndarray) -> Tuple[List, np.ndarray]:
        """VPI ORB 검출"""
        try:
            with vpi.Backend.CUDA:
                vpi_img = vpi.asimage(gray, vpi.Format.U8)
                keypoints, descriptors = vpi_img.orb(
                    max_features_per_level=self.max_features,
                    max_pyramid_levels=self.max_pyr_levels
                )
                kpts_np = keypoints.cpu()
                desc_np = descriptors.cpu()

                if len(kpts_np) > self.max_features:
                    responses = kpts_np[:, 3]
                    top_indices = np.argsort(responses)[-self.max_features:][::-1]
                    kpts_np = kpts_np[top_indices]
                    desc_np = desc_np[top_indices]

                cv_keypoints = []
                for kpt in kpts_np:
                    x, y, size, response = kpt[0], kpt[1], kpt[2], kpt[3]
                    cv_keypoints.append(
                        cv2.KeyPoint(float(x), float(y), float(size), -1, float(response))
                    )
                return cv_keypoints, desc_np
        except Exception as e:
            print(f"VPI detection failed, falling back to OpenCV: {e}")
            self.use_vpi = False
            return self._detect_opencv(gray)

    def _detect_opencv(self, gray: np.ndarray) -> Tuple[List, np.ndarray]:
        """OpenCV ORB 검출"""
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if keypoints is None or len(keypoints) == 0:
            return [], np.zeros((0, 32), dtype=np.uint8)

        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        keypoints = keypoints[:self.max_features]
        _, descriptors = self.orb.compute(gray, keypoints)

        if descriptors is None:
            descriptors = np.zeros((len(keypoints), 32), dtype=np.uint8)

        return keypoints, descriptors


# ============================================
# 특징 추출기 (분리된 클래스)
# ============================================

def apply_canny_edge_detection(image: np.ndarray, low_threshold: int = 50,
                                high_threshold: int = 150) -> np.ndarray:
    """
    Canny edge detection 적용

    Args:
        image: [H, W] or [H, W, 3] numpy array
        low_threshold: Canny 하한 임계값
        high_threshold: Canny 상한 임계값

    Returns:
        edges: [H, W] binary edge map (0 or 255)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    return edges


class MPMFeatureExtractor:
    """16x16 Grayscale 인코더 + 16x16 Canny 인코더 + VPI ORB 특징 추출기"""

    def __init__(self, encoder_path: str, image_size: Tuple[int, int] = (64, 64),
                 latent_dim: int = 32, device: str = 'cpu',
                 canny_low: int = 50, canny_high: int = 150):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.device = device
        self.canny_low = canny_low
        self.canny_high = canny_high

        # ORB 검출기
        self.orb_detector = ORBDetector(
            max_features=VPI_MAX_FEATURES,
            max_pyr_levels=VPI_MAX_PYR_LEVELS,
            intensity_threshold=VPI_INTENSITY_THRESHOLD
        )

        # 16x16 Grayscale 인코더 로드
        self.encoder_gray = self._load_encoder(encoder_path, name="Grayscale")

        # 16x16 Canny 인코더 로드 (같은 구조, 별도 파라미터)
        self.encoder_canny = self._load_encoder(encoder_path, name="Canny")

    def _load_encoder(self, encoder_path: str, name: str = "Encoder"):
        """16x16 인코더 로드"""
        if not Path(encoder_path).is_absolute():
            encoder_path = Path(__file__).parent.parent / encoder_path

        try:
            from model_defs.autoencoder_16x16_min import MinimalEncoder16x16
        except:
            spec = __import__('importlib').util.spec_from_file_location(
                "autoencoder_16x16_min",
                str(Path(__file__).parent / "autoencoder_16x16_min.py")
            )
            module = __import__('importlib').util.module_from_spec(spec)
            spec.loader.exec_module(module)
            MinimalEncoder16x16 = module.MinimalEncoder16x16

        encoder = MinimalEncoder16x16(input_channels=1, latent_dim=self.latent_dim)

        if Path(encoder_path).exists():
            checkpoint = torch.load(str(encoder_path), map_location='cpu', weights_only=False)
            if 'encoder_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
            elif 'model_state_dict' in checkpoint:
                encoder_dict = {k.replace('encoder.', ''): v
                               for k, v in checkpoint['model_state_dict'].items()
                               if k.startswith('encoder.')}
                encoder.load_state_dict(encoder_dict, strict=False)
            print(f"Loaded pretrained {name} encoder from {encoder_path}")
        else:
            print(f"Warning: {name} encoder checkpoint not found at {encoder_path}")

        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        encoder.to(self.device)

        return encoder

    def _preprocess_for_encoder(self, image: np.ndarray, use_canny: bool = False) -> torch.Tensor:
        """이미지를 16x16 Grayscale 또는 Canny로 전처리"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Canny edge detection 적용
        if use_canny:
            gray = apply_canny_edge_detection(gray, self.canny_low, self.canny_high)

        # 4픽셀 간격 다운샘플링 (64→16)
        downsampled = gray[::4, ::4]

        if downsampled.shape != (16, 16):
            downsampled = cv2.resize(downsampled, (16, 16), interpolation=cv2.INTER_LINEAR)

        img_tensor = torch.FloatTensor(downsampled / 255.0).unsqueeze(0).unsqueeze(0)
        return img_tensor

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 특징 추출

        Args:
            image: [H, W, 3] numpy array (BGR)

        Returns:
            features: [234] numpy array
                - latent_gray: [32]
                - latent_canny: [32]
                - ORB: [(x, y, desc[32]) × 5] = [170]
        """
        # 1. 16x16 Grayscale latent 추출
        img_gray_16x16 = self._preprocess_for_encoder(image, use_canny=False).to(self.device)
        with torch.no_grad():
            latent_gray, _ = self.encoder_gray(img_gray_16x16)
        latent_gray_np = latent_gray.cpu().numpy().flatten()  # [32]

        # 2. 16x16 Canny latent 추출
        img_canny_16x16 = self._preprocess_for_encoder(image, use_canny=True).to(self.device)
        with torch.no_grad():
            latent_canny, _ = self.encoder_canny(img_canny_16x16)
        latent_canny_np = latent_canny.cpu().numpy().flatten()  # [32]

        # 3. ORB 특징 추출
        keypoints, descriptors = self.orb_detector.detect_and_compute(image)

        # 4. ORB 특징 벡터화
        orb_features = []
        num_kpts = len(keypoints)

        if num_kpts < VPI_MAX_FEATURES:
            # Padding
            for i in range(num_kpts):
                x_norm = keypoints[i].pt[0] / self.image_size[0]
                y_norm = keypoints[i].pt[1] / self.image_size[1]
                desc = descriptors[i].astype(np.float32) / 255.0
                orb_features.extend([x_norm, y_norm])
                orb_features.extend(desc.tolist())

            # 더미 추가
            for _ in range(VPI_MAX_FEATURES - num_kpts):
                orb_features.extend([0.5, 0.5])  # 중앙 위치
                orb_features.extend([0.0] * 32)  # 더미 descriptor
        else:
            # 상위 5개만
            for i in range(VPI_MAX_FEATURES):
                x_norm = keypoints[i].pt[0] / self.image_size[0]
                y_norm = keypoints[i].pt[1] / self.image_size[1]
                desc = descriptors[i].astype(np.float32) / 255.0
                orb_features.extend([x_norm, y_norm])
                orb_features.extend(desc.tolist())

        orb_features = np.array(orb_features, dtype=np.float32)  # [170]

        # 5. 결합
        features = np.concatenate([latent_gray_np, latent_canny_np, orb_features])  # [32 + 32 + 170 = 234]

        return features


# ============================================
# Positional Encoding
# ============================================

class PositionalEncoding2D(nn.Module):
    """2D 좌표 기반 위치 인코딩"""

    def __init__(self, d_model: int, max_position: int = 100):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() *
            (-np.log(10000.0) / (d_model // 2))
        )
        self.register_buffer('div_term', div_term)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [B, N, 2] (x, y 좌표, 0~1 정규화)
        Returns:
            [B, N, d_model]
        """
        B, N, _ = positions.shape
        x_pos = positions[:, :, 0] * 100
        y_pos = positions[:, :, 1] * 100

        pe = torch.zeros(B, N, self.d_model, device=positions.device)
        pe[:, :, 0::4] = torch.sin(x_pos.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::4] = torch.cos(x_pos.unsqueeze(-1) * self.div_term)
        pe[:, :, 2::4] = torch.sin(y_pos.unsqueeze(-1) * self.div_term)
        pe[:, :, 3::4] = torch.cos(y_pos.unsqueeze(-1) * self.div_term)

        return pe


# ============================================
# Attention Modules
# ============================================

class CrossAttentionModule(nn.Module):
    """Cross Attention: Global ← Local"""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=global_feat,
            key=local_feat,
            value=local_feat
        )
        return self.norm(global_feat + self.dropout(attn_out))


class SelfAttentionModule(nn.Module):
    """Self Attention"""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(features, features, features)
        features = self.norm1(features + self.dropout(attn_out))
        ffn_out = self.ffn(features)
        features = self.norm2(features + self.dropout(ffn_out))
        return features


# ============================================
# Coordinate Classification Head
# ============================================

class CoordinateClassificationHead(nn.Module):
    """분류 기반 좌표 예측 헤드"""

    def __init__(self, input_dim: int, num_bins: int,
                 coord_min: float, coord_max: float, sigma: float = 2.0):
        super().__init__()
        self.num_bins = num_bins
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.sigma = sigma

        bin_centers = torch.linspace(coord_min, coord_max, num_bins)
        self.register_buffer('bin_centers', bin_centers)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_bins)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        coords = torch.sum(probs * self.bin_centers.unsqueeze(0), dim=-1)
        return logits, coords

    def create_soft_label(self, targets: torch.Tensor) -> torch.Tensor:
        B = targets.size(0)
        targets = targets.unsqueeze(1)
        distances = (self.bin_centers.unsqueeze(0) - targets) ** 2
        soft_labels = torch.exp(-distances / (2 * self.sigma ** 2))
        soft_labels = soft_labels / (soft_labels.sum(dim=-1, keepdim=True) + 1e-8)
        return soft_labels


# ============================================
# Main Model (순수 PyTorch - ONNX 변환 가능)
# ============================================

class MPMAttentionModel(nn.Module):
    """순수 PyTorch 모델 - 추출된 특징으로 좌표 예측"""
    MODEL_NAME = "mpm_cross_self_attention_vpi"

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # 입력 차원: latent_gray(32) + latent_canny(32) + ORB(170) = 234
        self.input_dim = 64 + 170  # gray_latent + canny_latent + ORB features
        self.latent_dim = 32  # 각 인코더의 latent 차원
        self.total_latent_dim = 64  # gray(32) + canny(32)
        self.orb_feature_dim = 34  # x, y, desc[32] per keypoint
        self.num_orb_kpts = 5

        # 좌표 설정
        self.x_range = (-112, 224)
        self.y_range = (0, 224)
        self.num_coordinates = 8  # 4 points × 2
        self.point_names = ['center', 'floor', 'front', 'side']

        # 하이퍼파라미터
        self.coord_bins = config.get('coord_bins', COORD_BINS)
        self.gaussian_sigma = config.get('gaussian_sigma', GAUSSIAN_SIGMA)
        self.embed_dim = config.get('embed_dim', 256)
        self.loss_alpha = config.get('loss_alpha', 0.7)

        # Feature projection
        self.latent_proj = nn.Linear(self.total_latent_dim, self.embed_dim)  # gray(32) + canny(32) = 64
        self.orb_proj = nn.Linear(32, self.embed_dim)  # descriptor만

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(self.embed_dim)

        # Attention
        self.cross_attention = CrossAttentionModule(self.embed_dim, num_heads=4)
        self.self_attention = SelfAttentionModule(self.embed_dim, num_heads=4)

        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Coordinate heads (8개)
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

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, 234] 추출된 특징

        Returns:
            dict with 'coordinates', 'pixel_coordinates', 'logits', 'point_names'
        """
        B = features.size(0)

        # 1. 특징 분리
        latent = features[:, :self.total_latent_dim]  # [B, 64] (gray 32 + canny 32)
        orb_features = features[:, self.total_latent_dim:].reshape(B, self.num_orb_kpts, self.orb_feature_dim)  # [B, 5, 34]

        # ORB x, y, descriptor 분리
        orb_positions = orb_features[:, :, :2]  # [B, 5, 2]
        orb_descriptors = orb_features[:, :, 2:]  # [B, 5, 32]

        # 2. Feature projection
        latent_proj = self.latent_proj(latent).unsqueeze(1)  # [B, 1, 256]
        orb_proj = self.orb_proj(orb_descriptors)  # [B, 5, 256]

        # 3. Positional encoding
        orb_pe = self.pos_encoding(orb_positions)  # [B, 5, 256]
        orb_enhanced = orb_proj + orb_pe

        # 4. Cross Attention
        global_enhanced = self.cross_attention(latent_proj, orb_enhanced)  # [B, 1, 256]

        # 5. Self Attention
        all_features = torch.cat([global_enhanced, orb_enhanced], dim=1)  # [B, 6, 256]
        all_enhanced = self.self_attention(all_features)  # [B, 6, 256]

        # 6. Global Pooling
        pooled = torch.mean(all_enhanced, dim=1)  # [B, 256]
        pooled = self.global_pool(pooled)

        # 7. Coordinate Classification
        all_logits = []
        all_coords = []

        for head in self.coordinate_heads:
            logits, coords = head(pooled)
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
        """손실 계산"""
        if sigma is None:
            sigma = self.gaussian_sigma

        logits_list = outputs['logits']
        pred_coords = outputs['pixel_coordinates']

        total_kl_loss = 0

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

        # MSE loss
        target_coords_pixel = torch.zeros_like(targets)
        for i in range(self.num_coordinates):
            if i % 2 == 0:
                target_coords_pixel[:, i] = targets[:, i] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            else:
                target_coords_pixel[:, i] = targets[:, i] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        mse_loss = F.mse_loss(pred_coords, target_coords_pixel)

        total_loss = self.loss_alpha * total_kl_loss + (1 - self.loss_alpha) * mse_loss

        return total_loss

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ============================================
# ONNX Wrapper (좌표만 반환)
# ============================================

class MPMAttentionModelONNX(nn.Module):
    """ONNX 변환용 래퍼 - 좌표만 반환"""

    def __init__(self, model: MPMAttentionModel):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """ONNX 변환용 - 좌표만 반환"""
        output = self.model(features)
        return output['coordinates']  # [B, 8]


# ============================================
# Dataset (특징 기반)
# ============================================

class DataSet(Dataset):
    """특징 기반 데이터셋"""

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

        # 특징 추출기
        self.feature_extractor = detector.feature_extractor

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

        # 데이터 로드
        self.data = []
        self.load_data(labels_file)

        # 특징 및 타겟 저장
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
        print(f"{self.mode.upper()} 데이터셋 특징 추출 중...")

        for sample in tqdm(self.data, desc="특징 추출"):
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

            # 리사이즈
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

            # 특징 추출
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
            if self.mode == 'train' and self.augment:
                for aug_idx in range(self.augment_count):
                    cropped_img, coords_cropped = apply_crop_augmentation(
                        orig_image, coords_orig, self.image_size, self.crop_config
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

        # 특징 추출기 생성
        encoder_path = config.get('features', {}).get('encoder_path', ENCODER_PATH)
        image_size = tuple(config.get('features', {}).get('image_size', IMAGE_SIZE))
        latent_dim = config.get('features', {}).get('encoder_latent_dim', ENCODER_LATENT_DIM)

        self.feature_extractor = MPMFeatureExtractor(
            encoder_path=encoder_path,
            image_size=image_size,
            latent_dim=latent_dim,
            device=device
        )

        # 모델 생성
        model_config = {
            'coord_bins': COORD_BINS,
            'gaussian_sigma': GAUSSIAN_SIGMA,
            'embed_dim': 256,
            'loss_alpha': 0.7
        }

        self.model = MPMAttentionModel(model_config)
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
        return self.model.compute_loss(outputs, targets, sigma)

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
    print("MPM Cross-Self Attention VPI 모델 (재설계 버전) 정의 완료")
    print(f"VPI 사용 가능: {VPI_AVAILABLE}")
    print(f"특징 차원: 234 (gray_latent 32 + canny_latent 32 + ORB 170)")
