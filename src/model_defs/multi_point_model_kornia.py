# -*- coding: utf-8 -*-
"""
Kornia 기반 다중 포인트 검출 모델 (ONNX 호환 버전)
- 색상 히스토그램, HSV 통계, 그레이디언트 통계 (단순화)
- LBP 대체 (라플라시안 기반 텍스처 통계)
- 가버 필터 응답 (고정 크기)
- SOSNet 특징 추출
- ONNX 변환 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2

# Kornia imports
import kornia
import kornia.filters as kfilters
import kornia.color as kcolor
import kornia.feature as kfeature
import kornia.enhance as kenhance
import kornia.geometry.transform as ktransform


class KorniaFeatureExtractor(nn.Module):
    """Kornia 기반 특징 추출기"""

    def __init__(self, config: Dict):
        super(KorniaFeatureExtractor, self).__init__()

        self.image_size = tuple(config.get('image_size', [112, 112]))
        self.grid_size = config.get('grid_size', 7)

        # 특징 추출 설정
        self.use_color_hist = config.get('use_color_hist', True)
        self.use_hsv = config.get('use_hsv', True)
        self.use_gradient = config.get('use_gradient', True)
        self.use_texture = config.get('use_texture', True)  # LBP 대체
        self.use_gabor = config.get('use_gabor', True)
        self.use_sosnet = config.get('use_sosnet', True)

        # 색상 히스토그램 bins
        self.hist_bins = config.get('hist_bins', 8)

        # Gabor 필터 설정 (고정 크기로 ONNX 호환성 개선)
        if self.use_gabor:
            self.gabor_kernel_size = 21  # 고정 크기
            self.create_gabor_filters_fixed()

        # SOSNet 디스크립터
        self.use_sosnet = self.use_sosnet and config.get('sosnet_enabled', True)
        if self.use_sosnet:
            try:
                self.sosnet = kfeature.SOSNet(pretrained=True)
                self.sosnet.eval()
                # ONNX 변환시 eval 모드 고정
                for param in self.sosnet.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"SOSNet 초기화 실패, 비활성화: {e}")
                self.use_sosnet = False

        # 특징 차원 계산
        self.feature_dim = self.calculate_feature_dim()

    def _create_gabor_kernel(self, size, sigma, theta, lambd, gamma):
        """Gabor 커널을 직접 생성"""
        x = torch.arange(-size // 2, size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-size // 2, size // 2 + 1, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        # Gabor formula
        exp_component = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
        cos_component = torch.cos(2 * np.pi * x_theta / lambd)

        kernel = exp_component * cos_component
        return kernel / kernel.sum()

    def create_gabor_filters_fixed(self):
        """ONNX 호환 고정 크기 가버 필터 뱅크 생성"""
        self.gabor_kernels = nn.ModuleList()

        # ONNX 호환을 위해 2개 필터만 사용
        wavelengths = [10.0]  # 단일 wavelength
        orientations = [0.0, 90.0]  # 2개 방향만

        for wavelength in wavelengths:
            for orientation in orientations:
                theta = orientation * np.pi / 180
                lambd = wavelength
                sigma = wavelength * 0.5
                gamma = 0.5

                # Gabor 커널 직접 생성
                kernel = self._create_gabor_kernel(
                    self.gabor_kernel_size,
                    sigma, theta, lambd, gamma
                )

                # nn.Conv2d로 래핑하여 ONNX 호환성 향상
                conv = nn.Conv2d(1, 1, self.gabor_kernel_size,
                                padding=self.gabor_kernel_size//2, bias=False)
                conv.weight.data = kernel.unsqueeze(0).unsqueeze(0)
                conv.weight.requires_grad = False

                self.gabor_kernels.append(conv)

    def calculate_feature_dim(self):
        """특징 벡터 차원 계산"""
        dim = 0

        # 기본 정보 (3)
        dim += 3  # width, height, aspect_ratio

        # 색상 히스토그램 (3채널 × bins × grid^2)
        if self.use_color_hist:
            dim += 3 * self.hist_bins * (self.grid_size ** 2)

        # HSV 통계 (6 global + 3×grid^2)
        if self.use_hsv:
            dim += 6  # global mean, std
            dim += 3 * (self.grid_size ** 2)  # grid mean

        # 그레이디언트 통계 (4×grid^2) - 단순화된 버전
        if self.use_gradient:
            dim += 4 * (self.grid_size ** 2)  # mean, std, energy, entropy

        # 텍스처 통계 (LBP 대체, 4×grid^2)
        if self.use_texture:
            dim += 4 * (self.grid_size ** 2)  # Laplacian 기반 통계

        # 가버 필터 응답 (ONNX 최적화를 위해 2개 필터만 사용)
        if self.use_gabor:
            dim += 2 * (self.grid_size ** 2)  # 2개 필터 (0도, 90도)

        # SOSNet (128차원 × grid^2)
        if self.use_sosnet:
            dim += 128 * (self.grid_size ** 2)  # 7x7 그리드에서 추출

        return dim

    def extract_color_histogram(self, image: torch.Tensor) -> torch.Tensor:
        """ONNX 호환 색상 히스토그램 추출"""
        B, C, H, W = image.shape
        features = []

        # 그리드별 히스토그램 (정적 연산)
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y1 = gy * H // self.grid_size
                y2 = (gy + 1) * H // self.grid_size
                x1 = gx * W // self.grid_size
                x2 = (gx + 1) * W // self.grid_size

                grid_region = image[:, :, y1:y2, x1:x2]

                # 각 채널별 간단한 통계 (histc 대신)
                for c in range(C):
                    channel = grid_region[:, c, :, :]
                    # Quantization을 통한 히스토그램 근사
                    for bin_idx in range(self.hist_bins):
                        bin_min = bin_idx / self.hist_bins
                        bin_max = (bin_idx + 1) / self.hist_bins
                        bin_mask = (channel >= bin_min) & (channel < bin_max)
                        # 배치 차원 유지
                        bin_count = bin_mask.float().mean(dim=[1, 2])  # [B]
                        features.append(bin_count)

        # features는 이제 [B] 텐서들의 리스트
        # 모든 특징을 1차원으로 평탄화
        flat_features = []
        for f in features:
            if f.dim() == 1:
                f = f.unsqueeze(1)  # [B] -> [B, 1]
            flat_features.append(f)
        return torch.cat(flat_features, dim=1)  # [B, num_features]

    def extract_hsv_features(self, image: torch.Tensor) -> torch.Tensor:
        """HSV 색공간 특징 추출"""
        # BGR to HSV 변환
        hsv = kcolor.rgb_to_hsv(image)

        features = []

        # Global HSV 통계
        for c in range(3):
            channel = hsv[:, c, :, :]
            features.append(channel.mean(dim=[1, 2]))  # [B]
            features.append(channel.std(dim=[1, 2]))   # [B]

        # Grid HSV mean
        B, C, H, W = hsv.shape
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y1 = gy * H // self.grid_size
                y2 = (gy + 1) * H // self.grid_size
                x1 = gx * W // self.grid_size
                x2 = (gx + 1) * W // self.grid_size

                grid_region = hsv[:, :, y1:y2, x1:x2]
                grid_mean = grid_region.mean(dim=[2, 3])  # [B, C]
                for i in range(3):
                    features.append(grid_mean[:, i])  # [B]

        # 모든 특징을 1차원으로 평탄화
        flat_features = []
        for f in features:
            if f.dim() == 1:
                f = f.unsqueeze(1)  # [B] -> [B, 1]
            flat_features.append(f)
        return torch.cat(flat_features, dim=1)  # [B, num_features]

    def extract_gradient_features(self, image: torch.Tensor) -> torch.Tensor:
        """ONNX 호환 그레이디언트 통계 추출 (단순화)"""
        # Grayscale 변환
        gray = kcolor.rgb_to_grayscale(image)

        # Sobel 그레이디언트 - spatial_gradient는 [B, 2, H, W] 형태 반환
        grads = kfilters.spatial_gradient(gray, mode='sobel')  # [B, 2, H, W]
        grad_x = grads[:, 0:1, :, :]  # [B, 1, H, W]
        grad_y = grads[:, 1:2, :, :]  # [B, 1, H, W]

        # Gradient magnitude와 angle
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # [B, 1, H, W]
        angle = torch.atan2(grad_y, grad_x)  # [B, 1, H, W]

        features = []
        # magnitude의 shape 사용 (grad_x와 동일)
        B = magnitude.shape[0]
        H = magnitude.shape[2]
        W = magnitude.shape[3]

        # Grid별 그레이디언트 통계 (확장)
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y1 = gy * H // self.grid_size
                y2 = (gy + 1) * H // self.grid_size
                x1 = gx * W // self.grid_size
                x2 = (gx + 1) * W // self.grid_size

                grid_mag = magnitude[:, :, y1:y2, x1:x2]
                grid_angle = angle[:, :, y1:y2, x1:x2]

                # 빈 그리드 체크
                if grid_mag.numel() == 0 or grid_mag.shape[2] == 0 or grid_mag.shape[3] == 0:
                    # 빈 그리드인 경우 0으로 채움
                    features.append(torch.zeros(B, device=image.device))
                    features.append(torch.zeros(B, device=image.device))
                    features.append(torch.zeros(B, device=image.device))
                    features.append(torch.zeros(B, device=image.device))
                else:
                    # 4개 통계: mean, std, energy, dominant_direction
                    features.append(grid_mag.mean(dim=[2, 3]).squeeze(1))  # [B]
                    features.append(grid_mag.std(dim=[2, 3]).squeeze(1))   # [B]
                    features.append((grid_mag**2).mean(dim=[2, 3]).squeeze(1))  # energy [B]
                    features.append(torch.cos(grid_angle).mean(dim=[2, 3]).squeeze(1))  # dominant direction [B]

        # 모든 특징을 1차원으로 평탄화
        flat_features = []
        for f in features:
            if f.dim() == 1:
                f = f.unsqueeze(1)  # [B] -> [B, 1]
            flat_features.append(f)
        return torch.cat(flat_features, dim=1)  # [B, num_features]

    def extract_texture_features(self, image: torch.Tensor) -> torch.Tensor:
        """ONNX 호환 텍스처 통계 (LBP 대체 - 라플라시안 기반)"""
        # Grayscale 변환
        gray = kcolor.rgb_to_grayscale(image)

        # Laplacian 필터로 텍스처 패턴 추출
        laplacian = kfilters.laplacian(gray, kernel_size=3)

        features = []
        # laplacian의 shape 사용
        B = laplacian.shape[0]
        H = laplacian.shape[2]
        W = laplacian.shape[3]

        # Grid별 텍스처 통계 (4개 특징으로 단순화)
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y1 = gy * H // self.grid_size
                y2 = (gy + 1) * H // self.grid_size
                x1 = gx * W // self.grid_size
                x2 = (gx + 1) * W // self.grid_size

                grid_region = laplacian[:, :, y1:y2, x1:x2]

                # 빈 그리드 체크
                if grid_region.numel() == 0 or grid_region.shape[2] == 0 or grid_region.shape[3] == 0:
                    # 빈 그리드인 경우 0으로 채움
                    features.append(torch.zeros(B, device=image.device))
                    features.append(torch.zeros(B, device=image.device))
                    features.append(torch.zeros(B, device=image.device))
                    features.append(torch.zeros(B, device=image.device))
                else:
                    # 4개의 핵심 텍스처 통계
                    features.append(grid_region.mean(dim=[2, 3]).squeeze(1))  # 평균 텍스처 강도 [B]
                    features.append(grid_region.std(dim=[2, 3]).squeeze(1))   # 텍스처 변화량 [B]
                    features.append((grid_region**2).mean(dim=[2, 3]).squeeze(1))  # 텍스처 에너지 [B]
                    features.append(grid_region.abs().mean(dim=[2, 3]).squeeze(1))  # 텍스처 복잡도 [B]

        # 모든 특징을 1차원으로 평탄화
        flat_features = []
        for f in features:
            if f.dim() == 1:
                f = f.unsqueeze(1)  # [B] -> [B, 1]
            flat_features.append(f)
        return torch.cat(flat_features, dim=1)  # [B, num_features]

    def extract_gabor_features(self, image: torch.Tensor) -> torch.Tensor:
        """ONNX 호환 가버 필터 응답 추출 (고정 크기)"""
        # Grayscale 변환
        gray = kcolor.rgb_to_grayscale(image)

        features = []
        B, C, H, W = gray.shape

        # 사전 정의된 가버 필터 적용
        for conv in self.gabor_kernels:
            filtered = conv(gray)
            filtered_abs = filtered.abs()

            # Grid별 응답 평균
            for gy in range(self.grid_size):
                for gx in range(self.grid_size):
                    y1 = gy * H // self.grid_size
                    y2 = (gy + 1) * H // self.grid_size
                    x1 = gx * W // self.grid_size
                    x2 = (gx + 1) * W // self.grid_size

                    grid_response = filtered_abs[:, :, y1:y2, x1:x2]
                    # 빈 그리드 체크
                    if grid_response.numel() == 0 or grid_response.shape[2] == 0 or grid_response.shape[3] == 0:
                        features.append(torch.zeros(B, device=image.device))
                    else:
                        features.append(grid_response.mean(dim=[2, 3]).squeeze(1))  # [B]

        # 모든 특징을 1차원으로 평탄화
        flat_features = []
        for f in features:
            if f.dim() == 1:
                f = f.unsqueeze(1)  # [B] -> [B, 1]
            flat_features.append(f)
        return torch.cat(flat_features, dim=1)  # [B, num_features]

    def extract_sosnet_features(self, image: torch.Tensor) -> torch.Tensor:
        """SOSNet 디스크립터 추출 - 7x7 그리드 기반"""
        # Grayscale 변환
        gray = kcolor.rgb_to_grayscale(image)

        B, C, H, W = gray.shape

        # 7x7 그리드에서 16x16 패치 추출
        patch_size = 16
        patches = []

        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                # 각 그리드 중심 위치
                cy = (gy * H // self.grid_size) + (H // (self.grid_size * 2))
                cx = (gx * W // self.grid_size) + (W // (self.grid_size * 2))

                # 16x16 패치 추출 - ONNX 호환 방식
                y1 = torch.clamp(torch.tensor(cy - patch_size//2), min=0).item()
                y2 = torch.clamp(torch.tensor(cy + patch_size//2), max=H).item()
                x1 = torch.clamp(torch.tensor(cx - patch_size//2), min=0).item()
                x2 = torch.clamp(torch.tensor(cx + patch_size//2), max=W).item()

                patch = gray[:, :, y1:y2, x1:x2]

                # 32x32로 리사이즈 (SOSNet 입력 크기) - 항상 수행 (ONNX 호환)
                patch = F.interpolate(patch, size=(32, 32), mode='bilinear', align_corners=False)

                patches.append(patch)

        # 배치로 합치기
        patches = torch.cat(patches, dim=0)  # [B*49, 1, 32, 32]

        # SOSNet 특징 추출
        with torch.no_grad():
            descriptors = self.sosnet(patches)  # [B*49, 128]

        # [B*49, 128] -> [B, 49*128=6272]
        B_total = descriptors.shape[0]
        B = B_total // 49
        descriptors = descriptors.view(B, 49*128)  # 명시적 크기로 view

        return descriptors

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """전체 특징 추출"""
        features = []

        # 입력 정규화 (0-255 -> 0-1) - ONNX 호환 방식
        max_val = image.max()
        scale = torch.where(max_val > 1, torch.tensor(255.0, device=image.device), torch.tensor(1.0, device=image.device))
        image = image / scale

        B, C, H, W = image.shape

        # 1. 기본 정보 - 각각을 [B, 1] 형태로
        features.append(torch.full((B, 1), W / self.image_size[0], dtype=torch.float32, device=image.device))
        features.append(torch.full((B, 1), H / self.image_size[1], dtype=torch.float32, device=image.device))
        features.append(torch.full((B, 1), W / (H + 1e-8), dtype=torch.float32, device=image.device))

        # 2. 색상 히스토그램
        if self.use_color_hist:
            color_hist = self.extract_color_histogram(image)  # [B, N]
            features.append(color_hist)

        # 3. HSV 특징
        if self.use_hsv:
            hsv_features = self.extract_hsv_features(image)  # [B, N]
            features.append(hsv_features)

        # 4. 그레이디언트 특징
        if self.use_gradient:
            grad_features = self.extract_gradient_features(image)  # [B, N]
            features.append(grad_features)

        # 5. 텍스처 특징 (LBP 대체)
        if self.use_texture:
            texture_features = self.extract_texture_features(image)  # [B, N]
            features.append(texture_features)

        # 6. 가버 필터 특징
        if self.use_gabor:
            gabor_features = self.extract_gabor_features(image)  # [B, N]
            features.append(gabor_features)

        # 7. SOSNet 특징
        if self.use_sosnet:
            sosnet_features = self.extract_sosnet_features(image)
            features.append(sosnet_features)

        # 특징 결합 - 모든 특징은 이미 [B, N] 형태
        combined_features = torch.cat(features, dim=1)

        return combined_features


class PositionalEncoding(nn.Module):
    """7x7 그리드에 대한 위치 인코딩"""

    def __init__(self, d_model: int = 64, grid_size: int = 7):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size

        # 학습 가능한 위치 임베딩 (49개 위치)
        self.position_embedding = nn.Parameter(
            torch.randn(grid_size * grid_size, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 49, features] 패치 특징
        Returns:
            [batch_size, 49, features + d_model] 위치 인코딩이 추가된 특징
        """
        batch_size = x.size(0)
        # 위치 임베딩을 배치에 맞게 확장
        pos_emb = self.position_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        # 원본 특징과 위치 임베딩 연결
        return torch.cat([x, pos_emb], dim=-1)


class KorniaPatchEncoder(nn.Module):
    """Kornia 패치 특징을 인코딩하는 모듈"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, pos_dim: int = 64):
        super().__init__()

        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(pos_dim, grid_size=7)

        # 패치 특징 변환
        self.patch_transform = nn.Sequential(
            nn.Linear(input_dim + pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features: [batch_size, 49, input_dim]
        Returns:
            [batch_size, 49, hidden_dim]
        """
        # Add positional encoding
        patches_with_pos = self.positional_encoding(patch_features)  # [B, 49, input_dim+64]

        # Transform
        B, N, D = patches_with_pos.shape
        patches_flat = patches_with_pos.reshape(B * N, D)
        encoded = self.patch_transform(patches_flat)  # [B*49, hidden_dim]
        encoded = encoded.reshape(B, N, -1)  # [B, 49, hidden_dim]

        return encoded


class CrossAttentionFusion(nn.Module):
    """Cross-Attention 기반 특징 융합"""

    def __init__(self, patch_dim: int, global_dim: int,
                 fusion_dim: int = 512, num_heads: int = 4):
        super().__init__()

        # 1. Global feature를 patch 차원으로 투영
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            nn.ReLU()
        )

        # 2. Cross-attention (patches attend to global)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 3. Attention pooling layer
        self.attention_pool = nn.Sequential(
            nn.Linear(patch_dim, patch_dim),
            nn.ReLU()
        )

        # 4. Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(patch_dim * 2, fusion_dim),  # patch + global 결합
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

    def forward(self, global_features: torch.Tensor,
                patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            global_features: [batch_size, global_dim]
            patch_features: [batch_size, 49, 128]
        Returns:
            fused: [batch_size, fusion_dim]
            attn_weights: [batch_size, num_heads, 49, 1] attention weights
        """
        B = global_features.size(0)

        # 1. Global feature 투영
        global_proj = self.global_proj(global_features)  # [B, 128]
        global_expanded = global_proj.unsqueeze(1)  # [B, 1, 128]

        # 2. Cross-attention (patches query, global key/value)
        attn_output, attn_weights = self.cross_attention(
            query=patch_features,  # [B, 49, 128]
            key=global_expanded,   # [B, 1, 128]
            value=global_expanded,  # [B, 1, 128]
            need_weights=True
        )

        # 3. Residual connection
        enhanced_patches = patch_features + attn_output  # [B, 49, 128]

        # 4. Attention-based pooling (mean pooling)
        patch_pooled = self.attention_pool(enhanced_patches)  # [B, 49, 128]
        patch_pooled = torch.mean(patch_pooled, dim=1)  # [B, 128]

        # 5. Global과 Patch 특징 결합
        combined = torch.cat([patch_pooled, global_proj], dim=-1)  # [B, 256]

        # 6. Final fusion
        fused = self.fusion_mlp(combined)  # [B, 512]

        return fused, attn_weights


class CoordinateHead(nn.Module):
    """단일 좌표를 예측하는 분류 헤드"""

    def __init__(self, input_dim: int, num_classes: int,
                 min_val: float, max_val: float):
        super().__init__()
        self.num_classes = num_classes
        self.min_val = min_val
        self.max_val = max_val

        # 클래스 중심 좌표 (미리 계산)
        self.register_buffer('class_centers',
                           torch.linspace(min_val, max_val, num_classes))

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            logits: [batch_size, num_classes]
            coords: [batch_size] 연속 좌표값
        """
        logits = self.classifier(x)  # [B, num_classes]

        # Softmax → 가중평균으로 연속 좌표 계산
        probs = F.softmax(logits, dim=-1)  # [B, num_classes]
        coords = torch.sum(probs * self.class_centers.unsqueeze(0), dim=-1)  # [B]

        return logits, coords

    def create_soft_label(self, target: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        연속 좌표값을 soft label 분포로 변환

        Args:
            target: [batch_size] 실제 좌표값
            sigma: Gaussian 분포의 표준편차
        Returns:
            [batch_size, num_classes] soft label 분포
        """
        batch_size = target.size(0)
        target = target.unsqueeze(1)  # [B, 1]

        # 각 클래스와의 거리 계산
        distances = (self.class_centers.unsqueeze(0) - target) ** 2  # [B, num_classes]

        # Gaussian 분포
        soft_labels = torch.exp(-distances / (2 * sigma ** 2))

        # 정규화
        soft_labels = soft_labels / soft_labels.sum(dim=-1, keepdim=True)

        return soft_labels


class FPDKorniaModel(nn.Module):
    """Kornia 특징 + FPD 아키텍처 통합 모델"""
    MODEL_NAME = "fpd_kornia"

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Kornia 특징 추출기
        self.feature_extractor = KorniaFeatureExtractor(config.get('features', {}))
        self.total_feature_dim = self.feature_extractor.feature_dim

        # 특징 차원 계산
        # 전체 특징을 49개 패치로 나누고 나머지는 전역 특징으로
        self.patch_feature_dim = self.total_feature_dim // 49  # 패치당 특징
        self.global_feature_dim = self.total_feature_dim - (self.patch_feature_dim * 49)  # 나머지는 전역

        # 최소 차원 보장
        if self.global_feature_dim < 10:
            self.global_feature_dim = 64
            self.global_proj = nn.Linear(self.total_feature_dim, self.global_feature_dim)
        else:
            self.global_proj = None

        # 네트워크 차원
        patch_hidden = 128
        fusion_dim = 512
        intermediate_dim = 128

        # 좌표 범위 및 클래스 수
        self.image_size = tuple(config.get('features', {}).get('image_size', [112, 112]))

        # config에서 coordinate_range 읽기 (없으면 기본값 사용)
        # config 구조가 중첩되어 있을 수 있으므로 여러 위치 확인
        coord_range = None

        # 먼저 최상위 config에서 찾기
        if 'coordinate_range' in config:
            coord_range = config['coordinate_range']
        # training.augmentation에서 찾기 (실제 config 구조)
        elif 'training' in config and 'augmentation' in config['training']:
            coord_range = config['training']['augmentation'].get('coordinate_range', {})

        if coord_range:
            # config에 정의된 비율 사용
            x_min_ratio = coord_range.get('x_min_ratio', -1.0)
            x_max_ratio = coord_range.get('x_max_ratio', 2.0)
            y_min_ratio = coord_range.get('y_min_ratio', 0.0)
            y_max_ratio = coord_range.get('y_max_ratio', 2.0)

            self.x_range = (
                int(x_min_ratio * self.image_size[0]),  # x_min
                int(x_max_ratio * self.image_size[0])   # x_max
            )
            self.y_range = (
                int(y_min_ratio * self.image_size[1]),  # y_min
                int(y_max_ratio * self.image_size[1])   # y_max
            )
        else:
            # 기본값 사용 (이전과 동일)
            self.x_range = (-self.image_size[0], self.image_size[0] * 2)  # -width ~ width*2
            self.y_range = (0, self.image_size[1] * 2)  # 0 ~ height*2
        self.x_classes = self.x_range[1] - self.x_range[0] + 1  # 337
        self.y_classes = self.y_range[1] - self.y_range[0] + 1  # 225
        self.num_coordinates = 8  # 4쌍의 (x, y)

        # 모듈 생성
        self.patch_encoder = KorniaPatchEncoder(
            self.patch_feature_dim, patch_hidden
        )

        self.cross_attention_fusion = CrossAttentionFusion(
            patch_dim=patch_hidden,
            global_dim=self.global_feature_dim,
            fusion_dim=fusion_dim,
            num_heads=4
        )

        # 중간 FC 네트워크
        self.intermediate_fc = nn.Sequential(
            nn.Linear(fusion_dim, intermediate_dim * 2),  # 512 → 256
            nn.LayerNorm(intermediate_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(intermediate_dim * 2, intermediate_dim),  # 256 → 128
            nn.LayerNorm(intermediate_dim),
            nn.ReLU()
        )

        # 8개 좌표 헤드 (x1, y1, x2, y2, x3, y3, x4, y4)
        self.coordinate_heads = nn.ModuleList()
        for i in range(self.num_coordinates):
            if i % 2 == 0:  # x 좌표
                head = CoordinateHead(intermediate_dim, self.x_classes, *self.x_range)
            else:  # y 좌표
                head = CoordinateHead(intermediate_dim, self.y_classes, *self.y_range)
            self.coordinate_heads.append(head)

        # 가중치 초기화
        self._initialize_weights(config.get('weight_init', 'he_normal'))

    def _initialize_weights(self, init_type: str):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'he_normal':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def reshape_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Kornia 특징을 전역/패치 특징으로 재구성

        Args:
            features: [batch_size, total_dim] Kornia 특징
        Returns:
            global_features: [batch_size, global_dim]
            patch_features: [batch_size, 49, patch_dim]
        """
        B = features.size(0)

        # 패치 특징 추출 (앞부분)
        patch_features = features[:, :self.patch_feature_dim * 49]
        patch_features = patch_features.view(B, 49, self.patch_feature_dim)

        # 전역 특징 추출 (뒷부분)
        if self.global_proj is not None:
            # 전체 특징을 투영하여 전역 특징 생성
            global_features = self.global_proj(features)
        else:
            global_features = features[:, self.patch_feature_dim * 49:]

        return global_features, patch_features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        # 특징 추출
        if len(x.shape) == 4:  # 이미지 입력인 경우
            features = self.feature_extractor(x)
        else:
            features = x

        # 특징 재구성
        global_features, patch_features = self.reshape_features(features)

        # 패치 인코딩
        encoded_patches = self.patch_encoder(patch_features)

        # Cross-attention 융합
        fused, attn_weights = self.cross_attention_fusion(global_features, encoded_patches)

        # 중간 FC 레이어
        intermediate = self.intermediate_fc(fused)  # [B, 128]

        # 좌표 예측
        all_logits = []
        all_coords = []

        for i, head in enumerate(self.coordinate_heads):
            logits, coords = head(intermediate)
            all_logits.append(logits)
            all_coords.append(coords)

        # Stack results
        coordinates = torch.stack(all_coords, dim=1)  # [B, 8]

        # Reshape to 2D points
        coordinates_2d = coordinates.view(-1, 4, 2)  # [B, 4, 2]

        return {
            'logits': all_logits,  # List of [B, num_classes]
            'coordinates': coordinates,  # [B, 8]
            'coordinates_2d': coordinates_2d  # [B, 4, 2]
        }

    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                     targets: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        손실 계산

        Args:
            predictions: forward()의 출력
            targets: [batch_size, 8] 실제 좌표값 (정규화된)
            sigma: soft label의 Gaussian 표준편차
        Returns:
            총 손실값
        """
        logits_list = predictions['logits']  # List of [B, num_classes]

        total_loss = 0

        for i in range(self.num_coordinates):
            # 해당 좌표의 헤드
            head = self.coordinate_heads[i]

            # 타겟을 실제 픽셀 좌표로 변환
            if i % 2 == 0:  # x 좌표
                # 정규화된 좌표를 픽셀 좌표로 변환
                pixel_coords = targets[:, i] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            else:  # y 좌표
                pixel_coords = targets[:, i] * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

            # Soft label 생성
            soft_labels = head.create_soft_label(pixel_coords, sigma)

            # Cross-entropy loss
            logits = logits_list[i]
            loss = -torch.sum(soft_labels * F.log_softmax(logits, dim=-1), dim=-1)
            loss = loss.mean()

            total_loss += loss

        return total_loss / self.num_coordinates

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """추론 모드에서 예측"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output['coordinates']  # [B, 8]


# ConfigurableMLPModel 클래스 제거됨 - FPDKorniaModel만 사용


class MultiPointDetectorKornia:
    """Kornia 기반 다중 포인트 검출기"""

    def __init__(self, model_config: Dict, device: str = 'cpu'):
        """
        Args:
            model_config: 모델 설정
            device: 'cpu' 또는 'cuda'
        """
        self.model_config = model_config
        self.device = device

        # 모델 생성 (FPD 아키텍처만 사용)
        # architecture config에 training 정보 추가 (coordinate_range 포함)
        arch_config = model_config['architecture'].copy()
        if 'training' in model_config:
            arch_config['training'] = model_config['training']
        self.model = FPDKorniaModel(arch_config)
        self.model.to(device)

        # 특징 추출 설정
        self.image_size = tuple(model_config['features']['image_size'])

        # 정규화 파라미터 (학습 후 설정됨)
        self.feature_mean = None
        self.feature_std = None

    def extract_features_numpy(self, image: np.ndarray) -> np.ndarray:
        """NumPy 이미지에서 특징 추출"""
        # NumPy to Tensor 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 특징 추출
        self.model.eval()
        with torch.no_grad():
            features = self.model.feature_extractor(image_tensor)

        return features.cpu().numpy()[0]

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """특징 정규화"""
        if self.feature_mean is not None and self.feature_std is not None:
            return (features - self.feature_mean) / (self.feature_std + 1e-8)
        return features

    def denormalize_features(self, features: np.ndarray) -> np.ndarray:
        """특징 역정규화"""
        if self.feature_mean is not None and self.feature_std is not None:
            return features * (self.feature_std + 1e-8) + self.feature_mean
        return features

    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """정규화 파라미터 설정"""
        self.feature_mean = mean
        self.feature_std = std

    def predict(self, image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """이미지에서 4개 포인트 예측"""
        orig_h, orig_w = image.shape[:2]

        # 이미지를 모델 입력 크기로 리사이즈
        image_resized = cv2.resize(image, self.image_size)

        # NumPy to Tensor 변환
        if len(image_resized.shape) == 2:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)

        # BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 예측
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(image_tensor)
            predictions = predictions.cpu().numpy()[0]

        # 정규화된 좌표를 픽셀 좌표로 변환
        x_min = -self.image_size[0]
        x_max = self.image_size[0] * 2
        y_min = 0
        y_max = self.image_size[1] * 2

        coords_112 = {}
        for i, key in enumerate(['center', 'floor', 'front', 'side']):
            norm_x = predictions[i * 2]
            norm_y = predictions[i * 2 + 1]

            # 역정규화
            x_112 = norm_x * (x_max - x_min) + x_min
            y_112 = norm_y * (y_max - y_min) + y_min

            coords_112[key] = (x_112, y_112)

        # 원본 이미지 크기로 변환
        scale_x = orig_w / self.image_size[0]
        scale_y = orig_h / self.image_size[1]

        coords_orig = {}
        for key, (x_112, y_112) in coords_112.items():
            x_orig = int(x_112 * scale_x)
            y_orig = int(y_112 * scale_y)
            coords_orig[key] = (x_orig, y_orig)

        return coords_orig

    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state=None, additional_info=None):
        """모델 체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if additional_info is not None:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)
        print(f"체크포인트 저장: {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer=False):
        """모델 체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_mean = checkpoint.get('feature_mean')
        self.feature_std = checkpoint.get('feature_std')

        print(f"체크포인트 로드: {filepath} (Epoch: {checkpoint.get('epoch', 'Unknown')})")

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            return checkpoint['optimizer_state_dict']

        return None