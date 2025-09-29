# -*- coding: utf-8 -*-
"""
PyTorch 기반 설정 가능한 다중 포인트 검출 모델
- 유연한 아키텍처 설정
- Dropout, BatchNorm 지원
- 다양한 활성화 함수 지원
- 200.learning.py와 호환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
from torch.utils.data import Dataset
import random
import os
from pathlib import Path
from tqdm import tqdm
import sys

# util 폴더를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))
from util.data_augmentation import apply_crop_augmentation


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
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파 - 200.learning.py와 호환되는 딕셔너리 반환"""
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

        # 딕셔너리로 반환 (200.learning.py 호환)
        return {'coordinates': x}

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """손실 계산 (200.learning.py 호환)

        Args:
            outputs: 모델 출력 {'coordinates': ...}
            targets: 타겟 좌표
            sigma: soft label sigma (사용 안 함)

        Returns:
            MSE 손실
        """
        predictions = outputs['coordinates']
        return F.mse_loss(predictions, targets)
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
            outputs = self.model(features_tensor)
            predictions = outputs['coordinates'].cpu().numpy()[0]
        
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


class DataSet(Dataset):
    """다중 포인트 검출 데이터셋 (PyTorch 버전)"""

    def __init__(self,
                 source_folder: str,
                 labels_file: str,
                 detector: 'PointDetector',
                 mode: str = 'train',  # 'train', 'val', 'test'
                 config: Dict = None,
                 augment: bool = True,
                 feature_mean: Optional[np.ndarray] = None,
                 feature_std: Optional[np.ndarray] = None,
                 extract_features: bool = False):
        """
        Args:
            source_folder: 이미지 폴더 경로
            labels_file: 레이블 파일명
            detector: 특징 추출용 검출기
            mode: 'train', 'val', 'test'
            config: 설정 딕셔너리
            augment: 데이터 증강 여부
            feature_mean: 특징 정규화용 평균
            feature_std: 특징 정규화용 표준편차
            extract_features: True면 특징을 미리 추출하여 메모리에 캐싱
        """
        self.source_folder = source_folder
        self.detector = detector
        self.mode = mode
        self.config = config or {}
        self.augment = augment and (mode == 'train')
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.extract_features = extract_features

        # 설정 읽기
        self.test_id_suffix = str(config['data_split']['test_id_suffix'])
        self.val_ratio = config['data_split']['validation_ratio']
        self.augment_count = config['training']['augmentation']['augment_count'] if self.augment else 0
        self.image_size = tuple(config['features']['image_size'])

        # max_train_images 설정 읽기
        self.max_train_images = config.get('data', {}).get('max_train_images', 0)

        # 크롭 증강 설정
        self.crop_config = config['training']['augmentation'].get('crop', {})
        self.crop_enabled = self.crop_config.get('enabled', False) and self.augment

        # 좌표 정규화 범위 설정 (PyTorch 모델 기본 설정)
        coord_range = config['training']['augmentation'].get('coordinate_range', {})
        width = self.image_size[0]
        height = self.image_size[1]
        self.coord_min_x = coord_range.get('x_min_ratio', -1.0) * width  # -width
        self.coord_max_x = coord_range.get('x_max_ratio', 2.0) * width   # width * 2
        self.coord_min_y = coord_range.get('y_min_ratio', 0.0) * height   # 0
        self.coord_max_y = coord_range.get('y_max_ratio', 2.0) * height   # height * 2

        # 데이터 로드
        self.data = []
        self.load_data(labels_file)

        # 이미지와 타겟 저장 (메모리에 미리 로드)
        self.images = []  # 이미지 저장
        self.features = []  # 추출된 특징 저장 (extract_features=True일 때)
        self.targets = []
        self.metadata = []

        if self.extract_features:
            self.precompute_features()
        else:
            self.precompute_data()

        print(f"{mode.upper()} 데이터셋: {len(self.features) if self.extract_features else len(self.images)}개 샘플")

    def load_data(self, labels_file: str):
        """레이블 파일에서 데이터 로드 및 분할"""
        labels_path = os.path.join(self.source_folder, labels_file)

        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        all_train_data = []
        test_data = []

        # 헤더 스킵하고 처리
        for idx, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 11:
                continue

            # ID와 파일명
            data_id = parts[0]  # ID 컬럼
            filename = parts[2]

            # 좌표 파싱 (실제 이미지 픽셀 좌표)
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

            # ID 끝자리로 train/test 분리
            if data_id.endswith(self.test_id_suffix):
                test_data.append(sample)
            else:
                all_train_data.append(sample)

        print(f"전체 데이터: Train {len(all_train_data)}개, Test {len(test_data)}개")

        # max_train_images 적용
        if self.max_train_images > 0 and self.mode != 'test':
            all_train_data = all_train_data[:self.max_train_images]
            print(f"max_train_images 설정으로 train 데이터를 {len(all_train_data)}개로 제한")

        # mode에 따라 데이터 할당
        if self.mode == 'test':
            self.data = test_data
        else:
            # train/val 분할
            random.seed(self.config['data_split']['random_seed'])
            random.shuffle(all_train_data)

            val_size = int(len(all_train_data) * self.val_ratio)

            if self.mode == 'val':
                self.data = all_train_data[:val_size]
            else:  # train
                self.data = all_train_data[val_size:]

    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """좌표를 min-max 범위로 정규화

        Args:
            x: x 좌표 (픽셀)
            y: y 좌표 (픽셀)

        Returns:
            정규화된 좌표 (0~1 범위)
        """
        # Min-Max 정규화
        norm_x = (x - self.coord_min_x) / (self.coord_max_x - self.coord_min_x)
        norm_y = (y - self.coord_min_y) / (self.coord_max_y - self.coord_min_y)

        # 클리핑 (0~1 범위)
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)

        return norm_x, norm_y

    def denormalize_coordinates(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """정규화된 좌표를 원본 범위로 복원

        Args:
            norm_x: 정규화된 x 좌표 (0~1)
            norm_y: 정규화된 y 좌표 (0~1)

        Returns:
            원본 좌표 (픽셀)
        """
        x = norm_x * (self.coord_max_x - self.coord_min_x) + self.coord_min_x
        y = norm_y * (self.coord_max_y - self.coord_min_y) + self.coord_min_y

        return x, y

    def precompute_features(self):
        """모든 데이터의 특징을 미리 추출하여 저장"""
        print(f"{self.mode.upper()} 데이터셋 특징 추출 중...")

        for sample in tqdm(self.data, desc="특징 추출", disable=False):
            # 이미지 로드
            image_path = os.path.join(self.source_folder, sample['filename'])
            orig_image = cv2.imread(image_path)

            if orig_image is None:
                print(f"Warning: 이미지 로드 실패 - {image_path}")
                orig_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
                orig_h, orig_w = self.image_size
            else:
                orig_h, orig_w = orig_image.shape[:2]

            # 원본 이미지의 좌표 (실제 픽셀 좌표)
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

            # 1. 원본 이미지 처리 (단순 리사이즈)
            image_112 = cv2.resize(orig_image, self.image_size)

            # 원본 이미지 크기에서 112x112로 좌표 변환
            scale_x = self.image_size[0] / orig_w
            scale_y = self.image_size[1] / orig_h

            coords_112 = {}
            for key in ['center', 'floor', 'front', 'side']:
                coords_112[f'{key}_x'] = coords_orig[f'{key}_x'] * scale_x
                coords_112[f'{key}_y'] = coords_orig[f'{key}_y'] * scale_y

            # 특징 추출
            features = self.detector.detector.extract_features(image_112)

            # 특징 정규화 (제공된 경우)
            if self.feature_mean is not None and self.feature_std is not None:
                features = (features - self.feature_mean) / (self.feature_std + 1e-8)

            # 좌표 정규화 (min-max 방식)
            targets = []
            for key in ['center', 'floor', 'front', 'side']:
                x = coords_112[f'{key}_x']
                y = coords_112[f'{key}_y']
                norm_x, norm_y = self.normalize_coordinates(x, y)
                targets.extend([norm_x, norm_y])

            targets = np.array(targets, dtype=np.float32)

            # 원본 데이터 저장
            self.features.append(torch.FloatTensor(features))
            self.targets.append(torch.FloatTensor(targets))
            self.metadata.append({
                'filename': sample['filename'],
                'id': sample['id'],
                'augmented': False,
                'aug_idx': 0
            })

            # 2. 크롭 증강 처리 (train 모드에서만)
            if self.mode == 'train' and self.augment:
                for aug_idx in range(self.augment_count):
                    # 원본 이미지에서 크롭 증강 적용
                    cropped_112, coords_cropped = apply_crop_augmentation(
                        orig_image, coords_orig, self.image_size, self.crop_config
                    )

                    # 특징 추출
                    features_aug = self.detector.detector.extract_features(cropped_112)

                    # 특징 정규화
                    if self.feature_mean is not None and self.feature_std is not None:
                        features_aug = (features_aug - self.feature_mean) / (self.feature_std + 1e-8)

                    # 좌표 정규화
                    targets_aug = []
                    for key in ['center', 'floor', 'front', 'side']:
                        x = coords_cropped[f'{key}_x']
                        y = coords_cropped[f'{key}_y']
                        norm_x, norm_y = self.normalize_coordinates(x, y)
                        targets_aug.extend([norm_x, norm_y])

                    targets_aug = np.array(targets_aug, dtype=np.float32)

                    # 증강 데이터 저장
                    self.features.append(torch.FloatTensor(features_aug))
                    self.targets.append(torch.FloatTensor(targets_aug))
                    self.metadata.append({
                        'filename': sample['filename'],
                        'id': sample['id'],
                        'augmented': True,
                        'aug_idx': aug_idx + 1
                    })

    def precompute_data(self):
        """이미지를 미리 로드하여 저장 (특징 추출 안 함)"""
        print(f"{self.mode.upper()} 데이터셋 이미지 로드 중...")

        for sample in tqdm(self.data, desc="이미지 로드", disable=False):
            # 이미지 로드
            image_path = os.path.join(self.source_folder, sample['filename'])
            orig_image = cv2.imread(image_path)

            if orig_image is None:
                print(f"Warning: 이미지 로드 실패 - {image_path}")
                orig_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
                orig_h, orig_w = self.image_size
            else:
                orig_h, orig_w = orig_image.shape[:2]

            # 원본 이미지의 좌표 (실제 픽셀 좌표)
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

            # 1. 원본 이미지 처리 (단순 리사이즈)
            image_112 = cv2.resize(orig_image, self.image_size)

            # 원본 이미지 크기에서 112x112로 좌표 변환
            scale_x = self.image_size[0] / orig_w
            scale_y = self.image_size[1] / orig_h

            coords_112 = {}
            for key in ['center', 'floor', 'front', 'side']:
                coords_112[f'{key}_x'] = coords_orig[f'{key}_x'] * scale_x
                coords_112[f'{key}_y'] = coords_orig[f'{key}_y'] * scale_y

            # 좌표 정규화 (min-max 방식)
            targets = []
            for key in ['center', 'floor', 'front', 'side']:
                x = coords_112[f'{key}_x']
                y = coords_112[f'{key}_y']
                norm_x, norm_y = self.normalize_coordinates(x, y)
                targets.extend([norm_x, norm_y])

            targets = np.array(targets, dtype=np.float32)

            # 원본 데이터 저장
            self.images.append(image_112)
            self.targets.append(torch.FloatTensor(targets))
            self.metadata.append({
                'filename': sample['filename'],
                'id': sample['id'],
                'augmented': False,
                'aug_idx': 0
            })

            # 2. 크롭 증강 처리 (train 모드에서만)
            if self.mode == 'train' and self.augment:
                for aug_idx in range(self.augment_count):
                    # 원본 이미지에서 크롭 증강 적용
                    cropped_112, coords_cropped = apply_crop_augmentation(
                        orig_image, coords_orig, self.image_size, self.crop_config
                    )

                    # 좌표 정규화
                    targets_aug = []
                    for key in ['center', 'floor', 'front', 'side']:
                        x = coords_cropped[f'{key}_x']
                        y = coords_cropped[f'{key}_y']
                        norm_x, norm_y = self.normalize_coordinates(x, y)
                        targets_aug.extend([norm_x, norm_y])

                    targets_aug = np.array(targets_aug, dtype=np.float32)

                    # 증강 데이터 저장
                    self.images.append(cropped_112)
                    self.targets.append(torch.FloatTensor(targets_aug))
                    self.metadata.append({
                        'filename': sample['filename'],
                        'id': sample['id'],
                        'augmented': True,
                        'aug_idx': aug_idx + 1
                    })

    def __len__(self):
        return len(self.features) if self.extract_features else len(self.images)

    def __getitem__(self, idx):
        """미리 계산된 데이터 반환"""
        metadata = self.metadata[idx]
        targets = self.targets[idx].clone()

        if self.extract_features:
            # 특징이 미리 추출된 경우
            features = self.features[idx].clone()

            # 데이터 증강 (노이즈 추가)
            if metadata.get('augmented', False) and self.augment:
                noise_std = self.config['training']['augmentation'].get('noise_std', 0.01)
                noise = torch.randn_like(features) * noise_std
                features = features + noise

            return {
                'data': features,
                'targets': targets,
                'filename': metadata['filename'],
                'id': metadata['id']
            }
        else:
            # 이미지에서 특징 추출
            image = self.images[idx]

            # 특징 추출
            features = self.detector.detector.extract_features(image)

            # 특징 정규화 (제공된 경우)
            if self.detector.feature_mean is not None and self.detector.feature_std is not None:
                features = (features - self.detector.feature_mean) / (self.detector.feature_std + 1e-8)

            features = torch.FloatTensor(features)

            return {
                'data': features,
                'targets': targets,
                'filename': metadata['filename'],
                'id': metadata['id']
            }


class PointDetector:
    """200.learning.py와 호환되는 PyTorch 포인트 검출기 래퍼"""

    def __init__(self, config: Dict, device):
        """초기화

        Args:
            config: 전체 설정 딕셔너리 (learning_model + features)
            device: 디바이스
        """
        self.config = config
        self.device = device

        # 먼저 임시 config로 특징 추출기만 생성
        temp_config = {
            'architecture': {
                'input_dim': 1,  # 임시값
                'hidden_dims': [1],
                'output_dim': 8
            },
            'features': config['features']
        }

        # 특징 추출을 위한 검출기 생성
        self.detector = MultiPointDetectorPyTorch(temp_config, device)

        # 더미 이미지로 실제 특징 차원 계산
        import numpy as np
        dummy_image = np.zeros((112, 112, 3), dtype=np.uint8)
        dummy_features = self.detector.extract_features(dummy_image)
        actual_input_dim = len(dummy_features)

        print(f"Detected feature dimension: {actual_input_dim}")

        # 실제 차원으로 PyTorch 모델 설정 생성
        pytorch_config = {
            'architecture': {
                'input_dim': actual_input_dim,  # 동적으로 계산된 차원
                'hidden_dims': [384, 256],  # 기존 아키텍처 유지
                'output_dim': 8,
                'dropout_rates': [0.2, 0.15],
                'activation': 'relu',
                'use_batch_norm': True
            },
            'features': config['features']
        }

        # 올바른 차원으로 모델 재생성
        self.detector.model = ConfigurableMLPModel(pytorch_config['architecture'])
        self.detector.model.to(device)
        self.model = self.detector.model

        # 정규화 파라미터
        self.feature_mean = None
        self.feature_std = None

    def __call__(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """모델 forward (200.learning.py 호환)

        Args:
            features: 특징 텐서 [B, feature_dim]

        Returns:
            {'coordinates': [B, 8]} 형태의 딕셔너리
        """
        # 모델 예측
        coordinates = self.model(features)

        return {
            'coordinates': coordinates
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """손실 계산 (200.learning.py 호환)

        Args:
            outputs: 모델 출력 {'coordinates': ...}
            targets: 타겟 좌표
            sigma: soft label sigma (사용 안 함)

        Returns:
            MSE 손실
        """
        predictions = outputs['coordinates']
        return F.mse_loss(predictions, targets)


# 200.learning.py와의 호환성을 위한 alias
PointDetectorDataSet = DataSet
# PointDetector는 이미 정의됨