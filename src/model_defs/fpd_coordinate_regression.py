#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FPD Coordinate Regression Model
FPD 특징을 입력으로 8개 좌표(4쌍의 x,y)를 예측하는 회귀 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


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


class PatchEncoder(nn.Module):
    """패치 특징을 인코딩하는 모듈"""
    
    def __init__(self, opencv_dim: int = 21, latent_dim: int = 16, 
                 hidden_dim: int = 128, pos_dim: int = 64):
        super().__init__()
        self.patch_dim = opencv_dim + latent_dim  # 37
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(pos_dim, grid_size=7)
        
        # 패치 특징 변환
        self.patch_transform = nn.Sequential(
            nn.Linear(self.patch_dim + pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, opencv_features: torch.Tensor, latent_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opencv_features: [batch_size, 49, 21]
            latent_features: [batch_size, 49, 16]
        Returns:
            [batch_size, 49, hidden_dim]
        """
        # Stack features
        patches = torch.cat([opencv_features, latent_features], dim=-1)  # [B, 49, 37]
        
        # Add positional encoding
        patches_with_pos = self.positional_encoding(patches)  # [B, 49, 37+64]
        
        # Transform - reshape for LayerNorm
        B, N, D = patches_with_pos.shape
        patches_flat = patches_with_pos.reshape(B * N, D)
        encoded = self.patch_transform(patches_flat)  # [B*49, hidden_dim]
        encoded = encoded.reshape(B, N, -1)  # [B, 49, hidden_dim]
        
        return encoded


class FusionNetwork(nn.Module):
    """전역 특징과 패치 특징을 융합하는 네트워크"""

    def __init__(self, global_dim: int = 21, patch_hidden: int = 128,
                 fusion_dim: int = 512):
        super().__init__()

        # 패치 특징 집계 (49 패치 → 단일 벡터)
        self.patch_aggregation = nn.Sequential(
            nn.Linear(49 * patch_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 전역 특징 변환
        self.global_transform = nn.Sequential(
            nn.Linear(global_dim, fusion_dim // 4),
            nn.LayerNorm(fusion_dim // 4),
            nn.ReLU()
        )

        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim + fusion_dim // 4, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

    def forward(self, global_features: torch.Tensor,
                patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_features: [batch_size, 21]
            patch_features: [batch_size, 49, hidden_dim]
        Returns:
            [batch_size, fusion_dim]
        """
        batch_size = global_features.size(0)

        # Flatten patch features
        patch_flat = patch_features.view(batch_size, -1)  # [B, 49*hidden]
        patch_agg = self.patch_aggregation(patch_flat)  # [B, fusion_dim]

        # Transform global features
        global_transformed = self.global_transform(global_features)  # [B, fusion_dim/4]

        # Concatenate and fuse
        combined = torch.cat([patch_agg, global_transformed], dim=-1)
        fused = self.fusion(combined)  # [B, fusion_dim]

        return fused


class CrossAttentionFusion(nn.Module):
    """Cross-Attention 기반 특징 융합"""

    def __init__(self, patch_dim: int = 128, global_dim: int = 21,
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
            global_features: [batch_size, 21]
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


class FPDCoordinateRegression(nn.Module):
    """FPD 특징 기반 좌표 회귀 모델"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # 기본 설정
        self.global_dim = 21
        self.opencv_dim = 21
        self.latent_dim = 16
        self.num_coordinates = 8  # 4쌍의 (x, y)
        
        # 네트워크 차원
        patch_hidden = 128
        fusion_dim = 512
        intermediate_dim = 128  # 중간층 차원

        # 좌표 범위 및 클래스 수
        self.x_range = (-112, 228)
        self.y_range = (0, 228)
        self.x_classes = 341  # -112 ~ 228
        self.y_classes = 229  # 0 ~ 228

        # 모듈 생성
        self.patch_encoder = PatchEncoder(
            self.opencv_dim, self.latent_dim, patch_hidden
        )

        # Cross-Attention 융합 사용 (FusionNetwork 대신)
        self.use_cross_attention = True  # Cross-attention 사용 여부 플래그
        if self.use_cross_attention:
            self.cross_attention_fusion = CrossAttentionFusion(
                patch_dim=patch_hidden,
                global_dim=self.global_dim,
                fusion_dim=fusion_dim,
                num_heads=4
            )
        else:
            self.fusion_network = FusionNetwork(
                self.global_dim, patch_hidden, fusion_dim
            )

        # 중간 FC 네트워크 추가 (512 → 256 → 128)
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
            
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: FPDFeatureExtractor의 출력
                - 'global_features': [batch_size, 21]
                - 'patch_opencv_features': [batch_size, 49, 21]
                - 'patch_latent_features': [batch_size, 49, 16]
        Returns:
            - 'logits': [batch_size, 8, max_classes] 각 좌표의 logits
            - 'coordinates': [batch_size, 8] 예측된 연속 좌표
            - 'coordinates_2d': [batch_size, 4, 2] 4개 점의 (x,y) 좌표
        """
        # 특징 추출
        global_feat = features['global_features']
        opencv_feat = features['patch_opencv_features']
        latent_feat = features['patch_latent_features']
        
        # 패치 인코딩
        encoded_patches = self.patch_encoder(opencv_feat, latent_feat)

        # 특징 융합 (Cross-Attention 또는 기존 방식)
        if self.use_cross_attention:
            fused, attn_weights = self.cross_attention_fusion(global_feat, encoded_patches)
            # attention weights는 나중에 시각화 등에 활용 가능
        else:
            fused = self.fusion_network(global_feat, encoded_patches)

        # 중간 FC 레이어 통과
        intermediate = self.intermediate_fc(fused)  # [B, 128]

        # 좌표 예측
        all_logits = []
        all_coords = []

        for i, head in enumerate(self.coordinate_heads):
            logits, coords = head(intermediate)  # intermediate 사용
            # 패딩을 위해 최대 클래스 수에 맞춤
            max_classes = max(self.x_classes, self.y_classes)
            if logits.size(-1) < max_classes:
                padding = torch.zeros(
                    logits.size(0), 
                    max_classes - logits.size(-1),
                    device=logits.device
                )
                logits = torch.cat([logits, padding], dim=-1)
            all_logits.append(logits)
            all_coords.append(coords)
        
        # Stack results
        logits = torch.stack(all_logits, dim=1)  # [B, 8, max_classes]
        coordinates = torch.stack(all_coords, dim=1)  # [B, 8]
        
        # Reshape to 2D points
        coordinates_2d = coordinates.view(-1, 4, 2)  # [B, 4, 2]
        
        return {
            'logits': logits,
            'coordinates': coordinates,
            'coordinates_2d': coordinates_2d
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                     targets: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        손실 계산
        
        Args:
            predictions: forward()의 출력
            targets: [batch_size, 8] 실제 좌표값
            sigma: soft label의 Gaussian 표준편차
        Returns:
            총 손실값
        """
        logits = predictions['logits']  # [B, 8, max_classes]
        batch_size = logits.size(0)
        
        total_loss = 0
        
        for i in range(self.num_coordinates):
            # 해당 좌표의 헤드
            head = self.coordinate_heads[i]
            
            # Soft label 생성
            soft_labels = head.create_soft_label(targets[:, i], sigma)
            
            # 실제 클래스 수만큼 자르기
            coord_logits = logits[:, i, :head.num_classes]
            
            # Cross-entropy loss
            loss = -torch.sum(soft_labels * F.log_softmax(coord_logits, dim=-1), dim=-1)
            loss = loss.mean()
            
            total_loss += loss
        
        return total_loss / self.num_coordinates
    
    def predict(self, features: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        추론 모드: 좌표 예측
        
        Args:
            features: FPDFeatureExtractor의 출력
        Returns:
            [batch_size, 4, 2] 4개 점의 (x,y) 좌표
        """
        with torch.no_grad():
            output = self.forward(features)
            coords_2d = output['coordinates_2d'].cpu().numpy()
        return coords_2d