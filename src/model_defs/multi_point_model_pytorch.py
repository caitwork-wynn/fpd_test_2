# -*- coding: utf-8 -*-
"""
PyTorch 기반 설정 가능한 다중 포인트 검출 모델
- 유연한 아키텍처 설정
- Dropout, BatchNorm 지원
- 다양한 활성화 함수 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


class ConfigurableMLPModel(nn.Module):
    """설정 가능한 MLP 모델"""
    MODEL_NAME = "mlp_configurable"
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 모델 설정 딕셔너리
                - input_dim: 입력 차원
                - hidden_dims: 은닉층 크기 리스트
                - output_dim: 출력 차원
                - dropout_rates: 각 층별 dropout 비율
                - activation: 활성화 함수
                - use_batch_norm: 배치 정규화 사용 여부
        """
        super(ConfigurableMLPModel, self).__init__()
        
        self.config = config
        self.input_dim = config.get('input_dim', 903)
        self.hidden_dims = config.get('hidden_dims', [384, 256])
        self.output_dim = config.get('output_dim', 8)
        self.dropout_rates = config.get('dropout_rates', [0.2, 0.15])
        self.activation_name = config.get('activation', 'relu')
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # 활성화 함수 설정
        self.activation = self._get_activation(self.activation_name)
        
        # 레이어 구성
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # 입력층 -> 첫 번째 은닉층
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            if i < len(self.dropout_rates):
                self.dropouts.append(nn.Dropout(self.dropout_rates[i]))
            else:
                self.dropouts.append(nn.Dropout(0.1))  # 기본값
            
            prev_dim = hidden_dim
        
        # 출력층
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
        # 가중치 초기화
        self._initialize_weights(config.get('weight_init', 'he_normal'))
        
    def _get_activation(self, name: str):
        """활성화 함수 반환"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _initialize_weights(self, init_type: str):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'he_normal':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif init_type == 'normal':
                    nn.init.normal_(module.weight, 0, 0.02)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # 은닉층들
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Batch normalization
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        # 출력층 (활성화 함수 없음)
        x = self.output_layer(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """추론 모드에서 예측"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class MultiPointDetectorPyTorch:
    """PyTorch 기반 다중 포인트 검출기"""
    
    def __init__(self, model_config: Dict, device: str = 'cpu'):
        """
        Args:
            model_config: 모델 설정
            device: 'cpu' 또는 'cuda'
        """
        self.model_config = model_config
        self.device = device
        
        # 모델 생성
        self.model = ConfigurableMLPModel(model_config['architecture'])
        self.model.to(device)
        
        # 특징 추출 설정
        self.image_size = tuple(model_config['features']['image_size'])
        self.grid_size = model_config['features']['grid_size']
        self.use_hsv = model_config['features'].get('use_hsv', True)
        self.use_orb = model_config['features'].get('use_orb', True)
        self.use_color_hist = model_config['features'].get('use_color_hist', True)
        
        # 정규화 파라미터 (학습 후 설정됨)
        self.feature_mean = None
        self.feature_std = None
        
        # ORB 디텍터
        if self.use_orb:
            self.orb = cv2.ORB_create(nfeatures=5, scaleFactor=1.2, nlevels=3)
    
    def extract_features(self, image: np.ndarray, orig_size: tuple = None) -> np.ndarray:
        """이미지에서 특징 벡터 추출 (기존 코드와 동일)
        
        Args:
            image: 입력 이미지 (112x112로 리사이즈된 상태)
            orig_size: 원본 이미지 크기 (width, height) 튜플. None이면 현재 이미지 크기 사용
        """
        features = []
        h, w = image.shape[:2]
        
        # BGR to Gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. 기본 정보 (3개)
        features.append(w / self.image_size[0])  # 112로 정규화
        features.append(h / self.image_size[1])  # 112로 정규화
        
        # 원본 이미지 종횡비 사용 (제공된 경우)
        if orig_size is not None:
            orig_w, orig_h = orig_size
            features.append(orig_w / (orig_h + 1e-8))
        else:
            features.append(w / (h + 1e-8))
        
        # 2. 그리드 밝기 특징
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y1 = gy * h // self.grid_size
                y2 = (gy + 1) * h // self.grid_size
                x1 = gx * w // self.grid_size
                x2 = (gx + 1) * w // self.grid_size
                
                grid_region = gray[y1:y2, x1:x2]
                if grid_region.size > 0:
                    features.append(np.mean(grid_region) / 255)
                    features.append(np.std(grid_region) / 255)
                else:
                    features.append(0)
                    features.append(0)
        
        # 3. 그리드 엣지 특징
        edges = cv2.Canny(gray, 50, 150)
        
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y1 = gy * h // self.grid_size
                y2 = (gy + 1) * h // self.grid_size
                x1 = gx * w // self.grid_size
                x2 = (gx + 1) * w // self.grid_size
                
                grid_edges = edges[y1:y2, x1:x2]
                if grid_edges.size > 0:
                    features.append(np.sum(grid_edges > 0) / grid_edges.size)
                    features.append(np.std(grid_edges) / 255)
                else:
                    features.append(0)
                    features.append(0)
        
        # 4. 색상 히스토그램
        for i in range(3):  # B, G, R
            hist = cv2.calcHist([image], [i], None, [4], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-8)
            features.extend(hist)
        
        # 5. HSV 통계
        if self.use_hsv:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            for i in range(3):
                channel = hsv[:, :, i]
                features.append(np.mean(channel) / 255)
                features.append(np.std(channel) / 255)
            
            # HSV 그리드 특징
            for gy in range(self.grid_size):
                for gx in range(self.grid_size):
                    y1 = gy * h // self.grid_size
                    y2 = (gy + 1) * h // self.grid_size
                    x1 = gx * w // self.grid_size
                    x2 = (gx + 1) * w // self.grid_size
                    
                    cell = hsv[y1:y2, x1:x2]
                    if cell.size > 0:
                        mean_hsv = np.mean(cell, axis=(0, 1))
                        features.extend(mean_hsv / 255)
                    else:
                        features.extend([0, 0, 0])
        
        # 6. ORB 특징
        if self.use_orb:
            for gy in range(self.grid_size):
                for gx in range(self.grid_size):
                    y1 = gy * h // self.grid_size
                    y2 = (gy + 1) * h // self.grid_size
                    x1 = gx * w // self.grid_size
                    x2 = (gx + 1) * w // self.grid_size
                    
                    grid_region = gray[y1:y2, x1:x2]
                    
                    if grid_region.size > 0:
                        keypoints, _ = self.orb.detectAndCompute(grid_region, None)
                        if keypoints:
                            # 8개 ORB 특징
                            features.append(len(keypoints) / 5.0)
                            responses = [kp.response for kp in keypoints]
                            features.append(np.mean(responses) / 100.0)
                            
                            x_coords = [kp.pt[0] for kp in keypoints]
                            y_coords = [kp.pt[1] for kp in keypoints]
                            features.append(np.mean(x_coords) / (grid_region.shape[1] + 1e-8))
                            features.append(np.mean(y_coords) / (grid_region.shape[0] + 1e-8))
                            features.append(np.std(x_coords) / (grid_region.shape[1] + 1e-8))
                            features.append(np.std(y_coords) / (grid_region.shape[0] + 1e-8))
                            
                            angles = [kp.angle for kp in keypoints]
                            angles_rad = np.deg2rad(angles)
                            features.append(np.mean(np.sin(angles_rad)))
                            features.append(np.mean(np.cos(angles_rad)))
                        else:
                            features.extend([0] * 8)
                    else:
                        features.extend([0] * 8)
        
        # 7. 색상 히스토그램 그리드
        if self.use_color_hist:
            for gy in range(self.grid_size):
                for gx in range(self.grid_size):
                    y1 = gy * h // self.grid_size
                    y2 = (gy + 1) * h // self.grid_size
                    x1 = gx * w // self.grid_size
                    x2 = (gx + 1) * w // self.grid_size
                    
                    cell = image[y1:y2, x1:x2]
                    
                    if cell.size > 0:
                        for i in range(3):  # B, G, R
                            features.append(np.mean(cell[:, :, i]) / 255)
                    else:
                        features.extend([0, 0, 0])
        
        return np.array(features)
    
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
        """이미지에서 4개 포인트 예측
        
        Args:
            image: 입력 이미지 (원본 크기)
            
        Returns:
            원본 이미지 크기 기준 픽셀 좌표
        """
        orig_h, orig_w = image.shape[:2]
        
        # 이미지를 모델 입력 크기로 리사이즈
        image_resized = cv2.resize(image, self.image_size)
        
        # 특징 추출 (원본 이미지 크기 정보 전달)
        features = self.extract_features(image_resized, orig_size=(orig_w, orig_h))
        
        # 정규화
        features = self.normalize_features(features)
        
        # Tensor 변환
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # 예측
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        # 정규화된 좌표를 112x112 픽셀 좌표로 변환
        # 정규화 범위: x는 -112~224, y는 0~224
        x_min = -self.image_size[0]  # -112
        x_max = self.image_size[0] * 2  # 224
        y_min = 0
        y_max = self.image_size[1] * 2  # 224
        
        coords_112 = {}
        for i, key in enumerate(['center', 'floor', 'front', 'side']):
            norm_x = predictions[i * 2]
            norm_y = predictions[i * 2 + 1]
            
            # 역정규화 (0~1 -> 픽셀 좌표)
            x_112 = norm_x * (x_max - x_min) + x_min
            y_112 = norm_y * (y_max - y_min) + y_min
            
            coords_112[key] = (x_112, y_112)
        
        # 112x112 좌표를 원본 이미지 크기로 변환
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