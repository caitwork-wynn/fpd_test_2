#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FPD Feature Extractor
16x16 패치 기반 OpenCV 특징과 Autoencoder latent 특징 추출
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import sys

# 상위 디렉토리의 모듈 import를 위한 path 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_defs.autoencoder_16x16 import ResidualEncoder16x16


class FPDFeatureExtractor:
    """
    FPD 특징 추출기
    112x112 이미지를 7x7 그리드의 16x16 패치로 분할하여
    각 패치에서 OpenCV 기반 특징과 Autoencoder latent 벡터를 추출
    """
    
    def __init__(self, config_path: str = 'config_fpd.yml'):
        """
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 파일 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 기본 설정
        self.image_size = tuple(self.config['feature_extraction']['image_size'])
        self.patch_size = self.config['feature_extraction']['patch_size']
        self.grid_size = self.config['feature_extraction']['grid_size']
        self.use_grayscale = self.config['data']['use_grayscale']
        
        # 디바이스 설정
        self.device = self._setup_device()
        
        # Autoencoder encoder 로드
        self.encoder = None
        if self.config['feature_extraction']['latent_features']['enabled']:
            self.encoder = self._load_encoder()
        
        # OpenCV 특징 추출기 설정
        self.orb = None
        if self.config['feature_extraction']['opencv_features']['orb_features']:
            orb_config = self.config['feature_extraction']['opencv_features']['orb']
            self.orb = cv2.ORB_create(
                nfeatures=orb_config['nfeatures'],
                scaleFactor=orb_config['scale_factor'],
                nlevels=orb_config['nlevels']
            )
        
        # 엣지 검출 파라미터
        edge_config = self.config['feature_extraction']['opencv_features']['edge']
        self.canny_low = edge_config['canny_low']
        self.canny_high = edge_config['canny_high']
        
        
        print(f"FPDFeatureExtractor 초기화 완료")
        print(f"- 이미지 크기: {self.image_size}")
        print(f"- 그리드 크기: {self.grid_size}x{self.grid_size}")
        print(f"- 패치 크기: {self.patch_size}x{self.patch_size}")
        print(f"- OpenCV 특징: {self.config['feature_extraction']['opencv_features']['enabled']}")
        print(f"- Latent 특징: {self.config['feature_extraction']['latent_features']['enabled']}")
        if self.encoder:
            print(f"- Encoder 로드: {self.config['model']['autoencoder_path']}")
    
    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['cuda_device']}")
            print(f"CUDA 디바이스 사용: {device}")
        else:
            device = torch.device('cpu')
            print("CPU 디바이스 사용")
        return device
    
    def _load_encoder(self) -> ResidualEncoder16x16:
        """Autoencoder encoder 모델 로드"""
        encoder_path = self.config['model']['autoencoder_path']
        
        # Encoder 모델 생성
        encoder = ResidualEncoder16x16(
            input_channels=self.config['model']['input_channels'],
            latent_dim=self.config['model']['latent_dim']
        )
        
        # 가중치 로드
        if os.path.exists(encoder_path):
            print(f"Encoder 가중치 로드 중: {encoder_path}")
            checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)
            
            # encoder_state_dict 키가 있으면 그것을 사용, 아니면 전체를 state_dict로 간주
            if isinstance(checkpoint, dict) and 'encoder_state_dict' in checkpoint:
                state_dict = checkpoint['encoder_state_dict']
            else:
                state_dict = checkpoint
            
            encoder.load_state_dict(state_dict)
            print("Encoder 가중치 로드 완료")
        else:
            print(f"경고: Encoder 파일을 찾을 수 없습니다: {encoder_path}")
            print("랜덤 초기화된 encoder를 사용합니다.")
        
        encoder.to(self.device)
        encoder.eval()  # 평가 모드로 설정
        
        return encoder
    
    def extract_global_features(self, image: np.ndarray, gray: np.ndarray, orig_size: tuple = None) -> np.ndarray:
        """
        전체 이미지 레벨 특징 추출 (21차원)
        multi_point_model_pytorch.py와 동일한 전역 특징
        
        Args:
            image: 입력 이미지 (112x112, BGR)
            gray: 그레이스케일 이미지
            orig_size: 원본 이미지 크기 (width, height) 튜플
            
        Returns:
            21차원 전역 특징 벡터
        """
        features = []
        h, w = image.shape[:2]
        
        # 1. 기본 정보 (3차원)
        features.append(w / self.image_size[0])  # 112로 정규화
        features.append(h / self.image_size[1])  # 112로 정규화
        
        # 원본 이미지 종횡비 사용 (제공된 경우)
        if orig_size is not None:
            orig_w, orig_h = orig_size
            features.append(orig_w / (orig_h + 1e-8))
        else:
            features.append(w / (h + 1e-8))
        
        # 2. 전체 이미지 색상 히스토그램 (12차원)
        for i in range(3):  # B, G, R
            hist = cv2.calcHist([image], [i], None, [4], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-8)
            features.extend(hist)
        
        # 3. 전체 HSV 통계 (6차원)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            for i in range(3):
                channel = hsv[:, :, i]
                features.append(np.mean(channel) / 255.0)
                features.append(np.std(channel) / 255.0)
        else:
            # 그레이스케일인 경우
            features.extend([np.mean(gray) / 255.0, np.std(gray) / 255.0] * 3)
        
        return np.array(features, dtype=np.float32)
    
    def get_features(self, image: np.ndarray, orig_size: tuple = None) -> Dict[str, np.ndarray]:
        """
        112x112 이미지에서 특징 추출 (패치 기반)
        
        Args:
            image: 입력 이미지 (112x112, BGR or grayscale)
            orig_size: 원본 이미지 크기 (width, height) 튜플
            
        Returns:
            dict: {
                'global_features': (21,) numpy array - 전체 이미지 특징
                'patch_opencv_features': (49, 21) numpy array - 패치별 OpenCV 특징
                'patch_latent_features': (49, 16) numpy array - 패치별 latent 특징
            }
        """
        # 이미지 전처리
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        
        # 그레이스케일 변환
        if self.use_grayscale and len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        result = {}
        
        # 전역 특징 추출 (21차원)
        if self.config['feature_extraction']['opencv_features']['enabled']:
            global_features = self.extract_global_features(image, gray, orig_size)
            result['global_features'] = global_features
        
        # 7x7 그리드로 분할
        patches = self._split_to_patches(image)
        gray_patches = self._split_to_patches(gray)
        
        # 패치별 특징 추출
        patch_opencv_features = []
        
        for i in range(self.grid_size * self.grid_size):
            patch = patches[i]
            gray_patch = gray_patches[i]
            
            # OpenCV 기반 패치 특징 추출
            if self.config['feature_extraction']['opencv_features']['enabled']:
                opencv_feat = self.extract_patch_features(patch, gray_patch)
                patch_opencv_features.append(opencv_feat)
        
        # 결과 정리
        if patch_opencv_features:
            result['patch_opencv_features'] = np.array(patch_opencv_features)  # (49, 21)
        
        # Latent 특징 (패치 기반)
        if self.config['feature_extraction']['latent_features']['enabled'] and self.encoder:
            patch_latent_features = []
            for i in range(self.grid_size * self.grid_size):
                gray_patch = gray_patches[i]
                latent_feat = self.get_latent_vector(gray_patch)
                patch_latent_features.append(latent_feat)
            
            if patch_latent_features:
                result['patch_latent_features'] = np.array(patch_latent_features)  # (49, 16)
        
        return result
    
    def _split_to_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """이미지를 7x7 그리드의 16x16 패치로 분할"""
        patches = []
        h, w = image.shape[:2]
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                y1 = y * self.patch_size
                y2 = (y + 1) * self.patch_size
                x1 = x * self.patch_size
                x2 = (x + 1) * self.patch_size
                
                patch = image[y1:y2, x1:x2]
                patches.append(patch)
        
        return patches
    
    def extract_patch_features(self, patch: np.ndarray, gray_patch: np.ndarray) -> np.ndarray:
        """
        16x16 패치에서 OpenCV 기반 특징 추출 (21차원)
        
        Args:
            patch: 16x16 컬러 패치 (BGR)
            gray_patch: 16x16 그레이스케일 패치
            
        Returns:
            21차원 특징 벡터 (HSV std 추가로 18→21차원)
        """
        features = []
        config = self.config['feature_extraction']['opencv_features']
        
        # 1. 픽셀 통계 (2차원)
        if config['pixel_stats']:
            features.append(np.mean(gray_patch) / 255.0)
            features.append(np.std(gray_patch) / 255.0)
        
        # 2. 엣지 특징 (2차원)
        if config['edge_features']:
            edges = cv2.Canny(gray_patch, self.canny_low, self.canny_high)
            features.append(np.sum(edges > 0) / edges.size)  # 엣지 밀도
            features.append(np.std(edges) / 255.0)  # 엣지 강도 변화
        
        # 3. RGB 평균 (3차원)
        if config['rgb_mean']:
            if len(patch.shape) == 3:
                for i in range(3):
                    features.append(np.mean(patch[:, :, i]) / 255.0)
            else:
                # 그레이스케일인 경우 같은 값 3번
                gray_mean = np.mean(gray_patch) / 255.0
                features.extend([gray_mean] * 3)
        
        # 4. HSV 평균과 표준편차 (6차원으로 변경)
        if config['hsv_mean']:
            if len(patch.shape) == 3:
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                for i in range(3):
                    channel = hsv[:, :, i]
                    features.append(np.mean(channel) / 255.0)
                    features.append(np.std(channel) / 255.0)
            else:
                # 그레이스케일인 경우
                gray_mean = np.mean(gray_patch) / 255.0
                gray_std = np.std(gray_patch) / 255.0
                features.extend([gray_mean, gray_std] * 3)
        
        # 5. ORB 특징 (8차원)
        if config['orb_features'] and self.orb:
            keypoints = self.orb.detect(gray_patch, None)
            if keypoints:
                features.append(len(keypoints) / 5.0)  # 키포인트 개수
                responses = [kp.response for kp in keypoints]
                features.append(np.mean(responses) / 100.0)  # 평균 response
                
                # 키포인트 위치 통계
                x_coords = [kp.pt[0] / 16.0 for kp in keypoints]
                y_coords = [kp.pt[1] / 16.0 for kp in keypoints]
                features.extend([
                    np.mean(x_coords),
                    np.mean(y_coords),
                    np.std(x_coords) if len(x_coords) > 1 else 0,
                    np.std(y_coords) if len(y_coords) > 1 else 0
                ])
                
                # 각도 통계
                angles = np.deg2rad([kp.angle for kp in keypoints])
                features.extend([
                    np.mean(np.sin(angles)),
                    np.mean(np.cos(angles))
                ])
            else:
                features.extend([0] * 8)
        
        return np.array(features, dtype=np.float32)
    
    def get_latent_vector(self, gray_patch: np.ndarray) -> np.ndarray:
        """
        16x16 그레이스케일 패치에서 autoencoder latent 벡터 추출
        
        Args:
            gray_patch: 16x16 그레이스케일 패치
            
        Returns:
            16차원 latent 벡터
        """
        if not self.encoder:
            raise ValueError("Encoder가 로드되지 않았습니다.")
        
        # Tensor로 변환
        # (H, W) -> (1, 1, H, W)
        patch_tensor = torch.FloatTensor(gray_patch).unsqueeze(0).unsqueeze(0)
        
        # 정규화 (0-255 -> 0-1)
        if self.config['data']['normalize_input']:
            patch_tensor = patch_tensor / 255.0
        
        patch_tensor = patch_tensor.to(self.device)
        
        # Latent 벡터 추출
        with torch.no_grad():
            latent, _ = self.encoder(patch_tensor)
            latent = latent.cpu().numpy().squeeze()  # (16,)
        
        return latent
    
    def get_combined_features(self, image: np.ndarray, orig_size: tuple = None) -> np.ndarray:
        """
        OpenCV 특징과 latent 특징을 결합한 전체 특징 벡터 반환
        
        Args:
            image: 입력 이미지 (112x112)
            orig_size: 원본 이미지 크기 (width, height) 튜플
            
        Returns:
            결합된 특징 벡터 (1834차원: 21 + 1029 + 784)
        """
        features = self.get_features(image, orig_size)
        
        combined = []
        
        # 전역 특징
        if 'global_features' in features:
            combined.append(features['global_features'])  # (21,)
        
        # 패치 OpenCV 특징 평탄화
        if 'patch_opencv_features' in features:
            opencv_flat = features['patch_opencv_features'].flatten()  # (49*21=1029,)
            combined.append(opencv_flat)
        
        # 패치 Latent 특징 평탄화
        if 'patch_latent_features' in features:
            latent_flat = features['patch_latent_features'].flatten()  # (49*16=784,)
            combined.append(latent_flat)
        
        if combined:
            return np.concatenate(combined)
        else:
            return np.array([])
    
    def visualize_patches(self, image: np.ndarray, save_path: Optional[str] = None):
        """
        7x7 패치 분할 시각화
        
        Args:
            image: 입력 이미지
            save_path: 저장 경로 (None이면 표시만)
        """
        import matplotlib.pyplot as plt
        
        # 이미지 리사이즈
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        
        # 패치 분할
        patches = self._split_to_patches(image)
        
        # 시각화
        fig, axes = plt.subplots(self.grid_size, self.grid_size, figsize=(10, 10))
        fig.suptitle('7x7 Grid of 16x16 Patches', fontsize=16)
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                idx = y * self.grid_size + x
                ax = axes[y, x]
                
                # BGR to RGB for matplotlib
                if len(patches[idx].shape) == 3:
                    patch_rgb = cv2.cvtColor(patches[idx], cv2.COLOR_BGR2RGB)
                else:
                    patch_rgb = patches[idx]
                
                ax.imshow(patch_rgb, cmap='gray' if len(patch_rgb.shape) == 2 else None)
                ax.set_title(f'({y},{x})', fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"패치 시각화 저장: {save_path}")
        
        plt.show()
        
        return fig


def test_feature_extractor():
    """FPDFeatureExtractor 테스트"""
    print("\n" + "="*50)
    print("FPDFeatureExtractor 테스트")
    print("="*50)
    
    # 특징 추출기 생성
    extractor = FPDFeatureExtractor('config_fpd.yml')
    
    # 테스트 이미지 생성 (112x112 랜덤 이미지)
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # 특징 추출 (원본 크기 정보 포함)
    orig_size = (224, 224)  # 예시 원본 크기
    features = extractor.get_features(test_image, orig_size)
    
    # 결과 출력
    if 'global_features' in features:
        global_feat = features['global_features']
        print(f"\n전역 특징:")
        print(f"  - Shape: {global_feat.shape}")
        print(f"  - Min: {global_feat.min():.4f}")
        print(f"  - Max: {global_feat.max():.4f}")
        print(f"  - Mean: {global_feat.mean():.4f}")
    
    if 'patch_opencv_features' in features:
        opencv_feat = features['patch_opencv_features']
        print(f"\n패치 OpenCV 특징:")
        print(f"  - Shape: {opencv_feat.shape}")
        print(f"  - Min: {opencv_feat.min():.4f}")
        print(f"  - Max: {opencv_feat.max():.4f}")
        print(f"  - Mean: {opencv_feat.mean():.4f}")
    
    if 'patch_latent_features' in features:
        latent_feat = features['patch_latent_features']
        print(f"\n패치 Latent 특징:")
        print(f"  - Shape: {latent_feat.shape}")
        print(f"  - Min: {latent_feat.min():.4f}")
        print(f"  - Max: {latent_feat.max():.4f}")
        print(f"  - Mean: {latent_feat.mean():.4f}")
    
    # 결합된 특징
    combined = extractor.get_combined_features(test_image, orig_size)
    print(f"\n결합된 특징:")
    print(f"  - Shape: {combined.shape}")
    print(f"  - Total dimensions: {combined.shape[0]}")
    print(f"  - Expected: 전역(21) + 패치OpenCV(1029) + 패치Latent(784) = 1834차원")
    
    print("\n테스트 완료!")
    print("="*50)


if __name__ == "__main__":
    test_feature_extractor()