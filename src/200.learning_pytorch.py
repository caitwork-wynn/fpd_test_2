# -*- coding: utf-8 -*-
"""
PyTorch 기반 7x7 Grid 다중 포인트 검출 모델 학습 스크립트
- config.yml에서 모든 설정 읽기
- ID 끝자리로 train/test 데이터 분리
"""

import os
import sys
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import csv
from typing import Dict, List, Tuple, Optional
import random
import time
import json
from tqdm import tqdm
import shutil

# 모델 import
sys.path.append(str(Path(__file__).parent))
from model_defs.multi_point_model_pytorch import MultiPointDetectorPyTorch


class DualLogger:
    """화면과 파일에 동시 출력하는 로거 클래스"""
    
    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')
    
    def write(self, message: str):
        """메시지를 화면과 파일에 동시 출력"""
        print(message, end='')
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def close(self):
        """파일 핸들 닫기"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class MultiPointDataset(Dataset):
    """다중 포인트 검출 데이터셋"""
    
    def __init__(self, 
                 source_folder: str,
                 labels_file: str,
                 detector: MultiPointDetectorPyTorch,
                 mode: str = 'train',  # 'train', 'val', 'test'
                 config: Dict = None,
                 augment: bool = True,
                 feature_mean: Optional[np.ndarray] = None,
                 feature_std: Optional[np.ndarray] = None):
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
        """
        self.source_folder = source_folder
        self.detector = detector
        self.mode = mode
        self.config = config or {}
        self.augment = augment and (mode == 'train')
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        
        # 설정 읽기
        self.test_id_suffix = str(config['data_split']['test_id_suffix'])
        self.val_ratio = config['data_split']['validation_ratio']
        self.augment_count = config['training']['augmentation']['augment_count'] if self.augment else 0
        self.noise_std = config['training']['augmentation']['noise_std']
        self.image_size = tuple(config['features']['image_size'])
        
        # 크롭 증강 설정
        self.crop_config = config['training']['augmentation'].get('crop', {})
        self.crop_enabled = self.crop_config.get('enabled', False) and self.augment
        
        # 좌표 정규화 범위 설정
        coord_range = config['training']['augmentation'].get('coordinate_range', {})
        # 이미지 크기가 width x height일 때 (현재는 width = height)
        width = self.image_size[0]
        height = self.image_size[1]
        self.coord_min_x = coord_range.get('x_min_ratio', -1.0) * width  # -width
        self.coord_max_x = coord_range.get('x_max_ratio', 2.0) * width   # width * 2
        self.coord_min_y = coord_range.get('y_min_ratio', 0.0) * height   # 0
        self.coord_max_y = coord_range.get('y_max_ratio', 2.0) * height   # height * 2
        
        # 데이터 로드
        self.data = []
        self.load_data(labels_file)
        
        # 특징과 타겟을 미리 추출하여 저장
        self.features = []
        self.targets = []
        self.metadata = []
        self.precompute_features()
        
        print(f"{mode.upper()} 데이터셋: {len(self.features)}개 샘플")
    
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
    
    def apply_crop_augmentation(self, image: np.ndarray, coords_orig: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, int]]:
        """이미지 크롭 및 좌표 조정
        
        Args:
            image: 원본 이미지 (원본 크기)
            coords_orig: 원본 이미지 픽셀 좌표 딕셔너리
            
        Returns:
            크롭된 이미지(112x112)와 조정된 좌표
        """
        h, w = image.shape[:2]
        
        # 랜덤 크롭 크기 결정
        scale = random.uniform(self.crop_config.get('min_ratio', 0.8), 
                              self.crop_config.get('max_ratio', 1.0))
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        
        # 랜덤 크롭 위치 결정
        max_shift = self.crop_config.get('max_shift', 0.15)
        max_shift_x = int(w * max_shift)
        max_shift_y = int(h * max_shift)
        
        x1 = random.randint(0, min(max_shift_x, w - crop_w))
        y1 = random.randint(0, min(max_shift_y, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # 이미지 크롭
        cropped_image = image[y1:y2, x1:x2]
        
        # 리사이즈 (112x112로)
        cropped_image = cv2.resize(cropped_image, self.image_size)
        
        # 좌표 조정 (크롭 영역 기준으로)
        adjusted_coords = {}
        scale_x = self.image_size[0] / crop_w
        scale_y = self.image_size[1] / crop_h
        
        for key in ['center', 'floor', 'front', 'side']:
            orig_x = coords_orig[f'{key}_x']
            orig_y = coords_orig[f'{key}_y']
            
            # 크롭 영역 기준으로 조정 후 112x112 스케일 적용
            new_x = (orig_x - x1) * scale_x
            new_y = (orig_y - y1) * scale_y
            
            # 조정된 좌표 저장 (이미지 밖 좌표도 허용)
            adjusted_coords[f'{key}_x'] = new_x
            adjusted_coords[f'{key}_y'] = new_y
        
        return cropped_image, adjusted_coords
    
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
                # 이미지 로드 실패 시 더미 데이터 생성
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
            features = self.detector.extract_features(image_112)
            
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
                    cropped_112, coords_cropped = self.apply_crop_augmentation(orig_image, coords_orig)
                    
                    # 특징 추출
                    features_aug = self.detector.extract_features(cropped_112)
                    
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
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """미리 계산된 특징과 타겟 반환"""
        features = self.features[idx].clone()
        targets = self.targets[idx].clone()
        metadata = self.metadata[idx]
        
        # 데이터 증강 (노이즈 추가)
        if metadata.get('augmented', False) and self.augment:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
        
        return {
            'features': features,
            'targets': targets,
            'filename': metadata['filename'],
            'id': metadata['id']
        }


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          config['training']['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """검증/테스트"""
    model.eval()
    total_loss = 0
    all_errors = {
        'center': {'x': [], 'y': [], 'dist': []},
        'floor': {'x': [], 'y': [], 'dist': []},
        'front': {'x': [], 'y': [], 'dist': []},
        'side': {'x': [], 'y': [], 'dist': []}
    }
    
    # 첫 번째 배치에서 데이터셋 객체 가져오기 (좌표 범위 정보 필요)
    dataset = dataloader.dataset
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 픽셀 단위 오차 계산
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(len(outputs_np)):
                # 정규화된 좌표를 원본 픽셀 좌표로 변환
                pred_coords = []
                true_coords = []
                
                for j in range(0, 8, 2):
                    # 예측 좌표 역정규화
                    pred_x, pred_y = dataset.denormalize_coordinates(
                        outputs_np[i][j], outputs_np[i][j+1]
                    )
                    pred_coords.extend([pred_x, pred_y])
                    
                    # 실제 좌표 역정규화
                    true_x, true_y = dataset.denormalize_coordinates(
                        targets_np[i][j], targets_np[i][j+1]
                    )
                    true_coords.extend([true_x, true_y])
                
                pred_coords = np.array(pred_coords)
                true_coords = np.array(true_coords)
                
                # 각 포인트별 X, Y 오차 계산
                point_errors = {
                    'center': {'x': pred_coords[0] - true_coords[0], 
                               'y': pred_coords[1] - true_coords[1]},
                    'floor': {'x': pred_coords[2] - true_coords[2],
                              'y': pred_coords[3] - true_coords[3]},
                    'front': {'x': pred_coords[4] - true_coords[4],
                              'y': pred_coords[5] - true_coords[5]},
                    'side': {'x': pred_coords[6] - true_coords[6],
                             'y': pred_coords[7] - true_coords[7]}
                }
                
                for key, err in point_errors.items():
                    all_errors[key]['x'].append(abs(err['x']))
                    all_errors[key]['y'].append(abs(err['y']))
                    all_errors[key]['dist'].append(np.sqrt(err['x']**2 + err['y']**2))
    
    avg_loss = total_loss / len(dataloader)
    
    # 평균 오차 계산 (거리 기준)
    avg_errors = {}
    for key in all_errors:
        if all_errors[key]['dist']:
            avg_errors[key] = np.mean(all_errors[key]['dist'])
        else:
            avg_errors[key] = 0
    
    return avg_loss, avg_errors, all_errors


def save_training_log(log_data: List[Dict], filepath: str):
    """학습 로그 CSV 저장"""
    if not log_data:
        return
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
        writer.writeheader()
        writer.writerows(log_data)
    
    print(f"학습 로그 저장: {filepath}")


def save_model_as_onnx(model, config, checkpoint_path, device):
    """PyTorch 모델을 ONNX 형식으로 변환 및 저장
    
    Args:
        model: PyTorch 모델
        config: 설정 딕셔너리
        checkpoint_path: 원본 체크포인트 경로 (.pth)
        device: 디바이스
        
    Returns:
        Path: ONNX 파일 경로 (실패 시 None)
    """
    try:
        # ONNX 파일 경로 생성 (.pth -> .onnx)
        onnx_path = Path(str(checkpoint_path).replace('.pth', '.onnx'))
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 더미 입력 생성 (batch_size=1)
        input_dim = config['architecture']['input_dim']
        dummy_input = torch.randn(1, input_dim, device=device)
        
        # ONNX로 변환
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['coordinates'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'coordinates': {0: 'batch_size'}
            }
        )
        
        return onnx_path
    except Exception as e:
        print(f"ONNX 변환 실패: {e}")
        return None


def main():
    """메인 함수"""
    print("=" * 60)
    print("PyTorch 기반 7x7 Grid 다중 포인트 검출 모델")
    print("- config.yml 기반 설정")
    print("- ID 끝자리로 train/test 분리")
    print("=" * 60)
    
    # 설정 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 경로 설정
    base_dir = Path(__file__).parent
    source_path = (base_dir / config['source']['source_folder']).resolve()
    
    # PyTorch 설정
    pytorch_config = config['pytorch_model']
    
    # 모델 파일명 설정
    save_file_name = pytorch_config['checkpointing'].get('save_file_name', 'multi_point_7x7')
    
    # 디바이스 설정
    if pytorch_config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{pytorch_config['device']['cuda_device']}")
        print(f"GPU 사용: {torch.cuda.get_device_name(device)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        print(f"현재 할당된 메모리: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    else:
        device = torch.device('cpu')
        print("CPU 사용")
    
    # 로그 디렉토리 생성
    log_dir = Path(pytorch_config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 초기화
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{save_file_name}_{timestamp}.log"
    best_log_path = log_dir / f"{save_file_name}_best.log"
    
    logger = DualLogger(str(log_path))
    best_logger = DualLogger(str(best_log_path))
    
    def log_print(message: str = ""):
        """로거를 통한 출력"""
        if logger:
            logger.write(message + "\n")
        else:
            print(message)
    
    # 체크포인트 디렉토리 설정
    base_dir = Path(__file__).parent
    checkpoint_dir = base_dir / pytorch_config['checkpointing']['save_dir']
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 검출기 및 모델 생성
    log_print("\n모델 초기화...")
    detector = MultiPointDetectorPyTorch(pytorch_config, device)
    model = detector.model
    
    # 모델 아키텍처 정보 출력
    log_print("\n=== 모델 아키텍처 ===")
    log_print(f"입력 차원: {pytorch_config['architecture']['input_dim']}")
    log_print(f"은닉층: {pytorch_config['architecture']['hidden_dims']}")
    log_print(f"출력 차원: {pytorch_config['architecture']['output_dim']}")
    log_print(f"Dropout: {pytorch_config['architecture']['dropout_rates']}")
    log_print(f"활성화 함수: {pytorch_config['architecture']['activation']}")
    log_print(f"Batch Normalization: {pytorch_config['architecture']['use_batch_norm']}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"\n총 파라미터 수: {total_params:,}")
    log_print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 데이터셋 생성 (첫 번째: 정규화 파라미터 계산용)
    print("\n데이터셋 초기 로드 (정규화 파라미터 계산용)...")
    temp_train_dataset = MultiPointDataset(
        source_folder=str(source_path),
        labels_file='labels.txt',
        detector=detector,
        mode='train',
        config=pytorch_config,
        augment=False  # 정규화 파라미터 계산 시에는 증강 없이
    )
    
    # 정규화 파라미터 계산 (train 데이터만 사용)
    print("\n특징 정규화 파라미터 계산...")
    all_features = []
    for i in range(min(1000, len(temp_train_dataset))):  # 샘플링
        sample_data = temp_train_dataset[i]
        all_features.append(sample_data['features'].numpy())
    
    all_features = np.array(all_features)
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0) + 1e-8
    
    # 정규화 파라미터를 JSON으로 저장
    norm_params = {
        'feature_mean': feature_mean.tolist(),
        'feature_std': feature_std.tolist(),
        'feature_dim': len(feature_mean),
        'description': '특징 벡터 정규화 파라미터 (903차원)'
    }
    
    norm_params_path = checkpoint_dir / 'normalization_params.json'
    with open(str(norm_params_path), 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, indent=2, ensure_ascii=False)
    print(f"\n정규화 파라미터 JSON 저장: {norm_params_path}")
    print(f"  - 특징 차원: {len(feature_mean)}")
    print(f"  - Mean 범위: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
    print(f"  - Std 범위: [{feature_std.min():.4f}, {feature_std.max():.4f}]")
    
    detector.set_normalization_params(feature_mean, feature_std)
    
    # 정규화 파라미터를 포함한 데이터셋 재생성
    print("\n정규화 파라미터를 적용한 데이터셋 재생성...")
    train_dataset = MultiPointDataset(
        source_folder=str(source_path),
        labels_file='labels.txt',
        detector=detector,
        mode='train',
        config=pytorch_config,
        augment=pytorch_config['training']['augmentation']['enabled'],
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    val_dataset = MultiPointDataset(
        source_folder=str(source_path),
        labels_file='labels.txt',
        detector=detector,
        mode='val',
        config=pytorch_config,
        augment=False,
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    test_dataset = MultiPointDataset(
        source_folder=str(source_path),
        labels_file='labels.txt',
        detector=detector,
        mode='test',
        config=pytorch_config,
        augment=False,
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=pytorch_config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=pytorch_config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=pytorch_config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 옵티마이저 설정
    optimizer_name = pytorch_config['training']['optimizer'].lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=pytorch_config['training']['learning_rate'],
            weight_decay=pytorch_config['training']['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=pytorch_config['training']['learning_rate'],
            weight_decay=pytorch_config['training']['weight_decay']
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=pytorch_config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=pytorch_config['training']['weight_decay']
        )
    
    # 기존 체크포인트 확인 및 로드
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 기존 최고 모델 체크
    best_checkpoint_path = checkpoint_dir / f"{save_file_name}_best.pth"
    if best_checkpoint_path.exists():
        log_print(f"\n기존 최고 모델 발견: {best_checkpoint_path}")
        checkpoint = torch.load(str(best_checkpoint_path), map_location=device, weights_only=False)
        
        # 모델 상태 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'feature_mean' in checkpoint:
            detector.feature_mean = checkpoint['feature_mean']
            detector.feature_std = checkpoint['feature_std']
        
        
        # 이어서 학습할 것인지 확인 (자동으로 새로운 학습 시작)
        resume_training = False  # 자동으로 새로운 학습 시작 (필요시 True로 변경)
        
        if resume_training:
            # 옵티마이저 상태 로드
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            log_print(f"에폭 {start_epoch}부터 학습 재개")
            log_print(f"이전 최고 검증 손실: {best_val_loss:.6f}")
        else:
            log_print("새로운 학습 시작 (모델 가중치만 사용)")
    
    # 스케줄러 설정
    scheduler_config = pytorch_config['training']['scheduler']
    if scheduler_config['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_config['patience'],
            factor=scheduler_config['factor'],
            min_lr=scheduler_config['min_lr']
        )
    elif scheduler_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=pytorch_config['training']['epochs'],
            eta_min=scheduler_config['min_lr']
        )
    else:
        scheduler = None
    
    # 손실 함수
    criterion = nn.MSELoss()
    
    # 학습 설정
    epochs = pytorch_config['training']['epochs']
    early_stopping_patience = pytorch_config['training']['early_stopping']['patience']
    min_delta = pytorch_config['training']['early_stopping']['min_delta']
    
    # 학습 시작
    log_print(f"\n===== 학습 시작 =====")
    log_print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"총 에폭: {epochs}")
    log_print(f"Train: {len(train_dataset)}개, Val: {len(val_dataset)}개, Test: {len(test_dataset)}개")
    log_print(f"배치 크기: {pytorch_config['training']['batch_size']}")
    log_print(f"학습률: {pytorch_config['training']['learning_rate']}")
    log_print(f"옵티마이저: {optimizer_name.upper()}")
    log_print("=" * 50)
    
    # 모델 정보 저장용 딕셔너리
    model_info = {
        'model_name': save_file_name,
        'architecture': pytorch_config['architecture'],
        'training_config': pytorch_config['training'],
        'feature_config': pytorch_config['features'],
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    best_epoch = 0
    patience_counter = 0
    training_log = []
    training_start_time = time.time()
    
    # 테스트 샘플 선정 (추적용)
    sample_indices = random.sample(range(len(test_dataset)), 
                                 min(5, len(test_dataset)))
    
    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start_time = time.time()
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, pytorch_config)
        
        # 검증
        val_loss, val_errors, all_val_errors = validate(model, val_loader, criterion, device)
        
        # 스케줄러 업데이트
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 에폭 시간 계산
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        elapsed_h = int(total_elapsed // 3600)
        elapsed_m = int((total_elapsed % 3600) // 60)
        elapsed_s = int(total_elapsed % 60)
        elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"
        
        # 로깅
        current_lr = optimizer.param_groups[0]['lr']
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'center_error': val_errors['center'],
            'floor_error': val_errors['floor'],
            'front_error': val_errors['front'],
            'side_error': val_errors['side'],
            'avg_error': np.mean(list(val_errors.values()))
        }
        training_log.append(log_entry)
        
        # 콘솔 출력 (progress_interval에 따라, 첫 번째, 마지막)
        progress_interval = pytorch_config['logging'].get('progress_interval', 100)
        if epoch == 1 or epoch % progress_interval == 0 or epoch == epochs:
            print(f"Epoch [{epoch:4d}/{epochs}] | "
                  f"시간: {epoch_time:5.1f}s | "
                  f"누적: {elapsed_str} | "
                  f"LR: {current_lr:.6f} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Err: {log_entry['avg_error']:.1f}px")
        
        # 상세 로깅 (detail_interval에 따라)
        detail_interval = pytorch_config['logging'].get('detail_interval', 1000)
        if epoch % detail_interval == 0:
            log_print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 에폭 {epoch} 상세 정보")
            log_print(f"손실: Train={train_loss:.6f}, Val={val_loss:.6f}")
            log_print(f"좌표별 오차 (pixels):")
            for name in ['center', 'floor', 'front', 'side']:
                if name in all_val_errors and all_val_errors[name]['x']:
                    x_mean = np.mean(all_val_errors[name]['x'])
                    x_std = np.std(all_val_errors[name]['x'])
                    y_mean = np.mean(all_val_errors[name]['y'])
                    y_std = np.std(all_val_errors[name]['y'])
                    log_print(f"  {name.upper():6s} - X 오차: 평균={x_mean:.1f} 표준편차={x_std:.1f} | "
                            f"Y 오차: 평균={y_mean:.1f} 표준편차={y_std:.1f}")
            log_print(f"  전체 평균 거리 오차: {log_entry['avg_error']:.2f}px")
        
        # 100 에폭마다 체크포인트 저장
        if epoch % 100 == 0:
            checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = f"{save_file_name}_{checkpoint_timestamp}_{val_loss:.6f}.pth"
            checkpoint_path = checkpoint_dir / checkpoint_filename
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_errors': val_errors,
                'model_config': pytorch_config,
                'feature_mean': detector.feature_mean,
                'feature_std': detector.feature_std,
                'timestamp': checkpoint_timestamp
            }
            torch.save(checkpoint_data, str(checkpoint_path))
            log_print(f"\n[체크포인트 저장] 에폭 {epoch}: {checkpoint_filename}")
            
            # 샘플 추적
            if sample_indices:
                log_print("\n샘플 예측 추적:")
                model.eval()
                for idx in sample_indices[:3]:  # 3개만 표시
                    # metadata에서 실제 데이터 정보 가져오기
                    metadata = test_dataset.metadata[idx]
                    sample_data = test_dataset[idx]
                    features = sample_data['features'].unsqueeze(0).to(device)
                    targets = sample_data['targets']
                    
                    with torch.no_grad():
                        outputs = model(features)
                    
                    # 정규화된 좌표를 원본 픽셀 좌표로 변환
                    outputs_norm = outputs.cpu().numpy()[0]
                    targets_norm = targets.numpy()
                    
                    outputs = []
                    targets = []
                    for j in range(0, 8, 2):
                        # 예측 좌표 역정규화
                        pred_x, pred_y = test_dataset.denormalize_coordinates(
                            outputs_norm[j], outputs_norm[j+1]
                        )
                        outputs.extend([pred_x, pred_y])
                        
                        # 실제 좌표 역정규화
                        true_x, true_y = test_dataset.denormalize_coordinates(
                            targets_norm[j], targets_norm[j+1]
                        )
                        targets.extend([true_x, true_y])
                    
                    outputs = np.array(outputs)
                    targets = np.array(targets)
                    
                    log_print(f"  [{metadata['id']}]")
                    
                    # 4개 포인트 모두 표시
                    point_names = ['Center', 'Floor', 'Front', 'Side']
                    for i, name in enumerate(point_names):
                        true_x, true_y = targets[i*2], targets[i*2+1]
                        pred_x, pred_y = outputs[i*2], outputs[i*2+1]
                        error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                        log_print(f"    {name:6s}: ({true_x:.1f}, {true_y:.1f}) → "
                                f"({pred_x:.1f}, {pred_y:.1f}) | 오차: {error:.1f}px")
                
                # 전체 test 데이터셋에 대한 통계 정보 추가
                log_print("\n  Test 데이터셋 전체 예측 통계:")
                # 모든 test 데이터의 오차 수집
                test_errors = {'Center': {'x': [], 'y': []}, 
                              'Floor': {'x': [], 'y': []},
                              'Front': {'x': [], 'y': []},
                              'Side': {'x': [], 'y': []}}
                
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        features = batch['features'].to(device)
                        targets = batch['targets'].to(device)
                        
                        outputs = model(features)
                        
                        outputs_np = outputs.cpu().numpy()
                        targets_np = targets.cpu().numpy()
                        
                        for i in range(len(outputs_np)):
                            # 좌표 역정규화
                            for j, name in enumerate(['Center', 'Floor', 'Front', 'Side']):
                                pred_x, pred_y = test_dataset.denormalize_coordinates(
                                    outputs_np[i][j*2], outputs_np[i][j*2+1]
                                )
                                true_x, true_y = test_dataset.denormalize_coordinates(
                                    targets_np[i][j*2], targets_np[i][j*2+1]
                                )
                                test_errors[name]['x'].append(abs(pred_x - true_x))
                                test_errors[name]['y'].append(abs(pred_y - true_y))
                
                # 통계 출력
                for name in ['Center', 'Floor', 'Front', 'Side']:
                    x_mean = np.mean(test_errors[name]['x'])
                    x_std = np.std(test_errors[name]['x'])
                    y_mean = np.mean(test_errors[name]['y'])
                    y_std = np.std(test_errors[name]['y'])
                    log_print(f"  {name}: x: 평균 {x_mean:.2f}, 표준편차 {x_std:.2f}, "
                            f"y: 평균 {y_mean:.2f}, 표준편차 {y_std:.2f}")
        
        # 체크포인트 저장
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # 최고 모델 저장
            checkpoint_path = checkpoint_dir / f"{save_file_name}_best.pth"
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_errors': val_errors,
                'model_config': pytorch_config,
                'feature_mean': detector.feature_mean,
                'feature_std': detector.feature_std
            }
            torch.save(checkpoint_data, str(checkpoint_path))
            
            # Best 모델 저장 메시지 출력
            improvement = (previous_best - val_loss) if 'previous_best' in locals() else val_loss
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            log_print(f"\n{'='*60}")
            log_print(f"✓ BEST 모델 저장 완료!")
            log_print(f"  • 에폭: {epoch}")
            log_print(f"  • 검증 손실: {val_loss:.6f} (개선: {improvement:.6f})")
            log_print(f"  • 평균 오차: {log_entry['avg_error']:.2f}px")
            log_print(f"  • 파일: {checkpoint_path.name} ({file_size_mb:.2f}MB)")
            log_print(f"  • 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # ONNX로도 저장
            onnx_path = save_model_as_onnx(model, pytorch_config, checkpoint_path, device)
            if onnx_path and onnx_path.exists():
                onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                log_print(f"✓ ONNX 모델 저장: {onnx_path.name} ({onnx_size_mb:.2f}MB)")
            log_print(f"{'='*60}\n")
            
            previous_best = val_loss
            
            # best 로거에 기록
            if best_logger:
                best_logger.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
                best_logger.write(f"에폭 {epoch}: 손실={val_loss:.6f}, ")
                best_logger.write(f"평균 오차={log_entry['avg_error']:.2f}px\n")
            
            # 최고 모델 갱신 메시지 (100번에 한 번만 또는 중요한 개선일 때)
            if epoch % 100 == 0 or val_loss < best_val_loss * 0.9:
                log_print(f"\n*** 최고 모델 갱신! 에폭 {epoch}, Val Loss: {val_loss:.6f} ***")
        else:
            patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                # Early stopping (메시지 없이 조용히 종료)
                break
        
        # 주기적 저장
        if epoch % pytorch_config['checkpointing']['save_frequency'] == 0:
            if not pytorch_config['checkpointing']['save_best_only']:
                checkpoint_path = checkpoint_dir / f"{save_file_name}_epoch{epoch}.pth"
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_errors': val_errors,
                    'model_config': pytorch_config,
                    'feature_mean': detector.feature_mean,
                    'feature_std': detector.feature_std
                }
                torch.save(checkpoint_data, str(checkpoint_path))
                log_print(f"체크포인트 저장: {checkpoint_path}")
    
    # 학습 완료 시간 계산
    total_training_time = time.time() - training_start_time
    total_hours = int(total_training_time // 3600)
    total_minutes = int((total_training_time % 3600) // 60)
    total_seconds = int(total_training_time % 60)
    
    # 학습 로그 저장
    if pytorch_config['logging']['save_csv']:
        log_csv_path = log_dir / f"{save_file_name}_training_log.csv"
        save_training_log(training_log, str(log_csv_path))
    
    # 모델 정보 JSON 저장
    model_info['best_epoch'] = best_epoch
    model_info['best_val_loss'] = float(best_val_loss)
    model_info['total_training_time'] = f"{total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}"
    model_info['final_epoch'] = epoch
    
    model_info_path = checkpoint_dir / f"{save_file_name}_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    
    # 테스트 데이터 평가
    log_print("\n=== 테스트 데이터 최종 평가 ===")
    test_loss, test_errors, all_test_errors = validate(model, test_loader, criterion, device)
    
    log_print(f"Test Loss: {test_loss:.6f}")
    log_print("\n각 포인트별 오차 분석 (pixels):")
    for name in ['center', 'floor', 'front', 'side']:
        if name in all_test_errors and all_test_errors[name]['x']:
            x_mean = np.mean(all_test_errors[name]['x'])
            x_std = np.std(all_test_errors[name]['x'])
            y_mean = np.mean(all_test_errors[name]['y'])
            y_std = np.std(all_test_errors[name]['y'])
            dist_mean = np.mean(all_test_errors[name]['dist'])
            log_print(f"  {name.upper():6s}:")
            log_print(f"    X 오차: 평균={x_mean:.1f} 표준편차={x_std:.1f}")
            log_print(f"    Y 오차: 평균={y_mean:.1f} 표준편차={y_std:.1f}")
            log_print(f"    거리: 평균={dist_mean:.1f}px")
    log_print(f"\n  전체 평균 거리 오차: {np.mean(list(test_errors.values())):.2f}px")
    
    # 상세 통계
    log_print("\n상세 통계 (거리 기준):")
    for name, errors in all_test_errors.items():
        if errors['dist']:
            dist_array = np.array(errors['dist'])
            log_print(f"\n{name.upper()}:")
            log_print(f"  평균: {np.mean(dist_array):.2f}")
            log_print(f"  표준편차: {np.std(dist_array):.2f}")
            log_print(f"  중앙값: {np.median(dist_array):.2f}")
            log_print(f"  최소: {np.min(dist_array):.2f}")
            log_print(f"  최대: {np.max(dist_array):.2f}")
    
    # 테스트 결과 CSV 저장
    if pytorch_config['logging']['save_csv']:
        test_results = []
        model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)
                filenames = batch['filename']
                ids = batch['id']
                
                outputs = model(features)
                
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                for i in range(len(outputs_np)):
                    # 정규화된 좌표를 원본 픽셀 좌표로 변환
                    pred_coords = []
                    true_coords = []
                    
                    for j in range(0, 8, 2):
                        # 예측 좌표 역정규화
                        pred_x, pred_y = test_dataset.denormalize_coordinates(
                            outputs_np[i][j], outputs_np[i][j+1]
                        )
                        pred_coords.extend([pred_x, pred_y])
                        
                        # 실제 좌표 역정규화
                        true_x, true_y = test_dataset.denormalize_coordinates(
                            targets_np[i][j], targets_np[i][j+1]
                        )
                        true_coords.extend([true_x, true_y])
                    
                    pred_coords = np.array(pred_coords)
                    true_coords = np.array(true_coords)
                    
                    test_results.append({
                        'id': ids[i],
                        'filename': filenames[i],
                        'true_center_x': true_coords[0],
                        'true_center_y': true_coords[1],
                        'true_floor_x': true_coords[2],
                        'true_floor_y': true_coords[3],
                        'true_front_x': true_coords[4],
                        'true_front_y': true_coords[5],
                        'true_side_x': true_coords[6],
                        'true_side_y': true_coords[7],
                        'pred_center_x': pred_coords[0],
                        'pred_center_y': pred_coords[1],
                        'pred_floor_x': pred_coords[2],
                        'pred_floor_y': pred_coords[3],
                        'pred_front_x': pred_coords[4],
                        'pred_front_y': pred_coords[5],
                        'pred_side_x': pred_coords[6],
                        'pred_side_y': pred_coords[7],
                        'center_error': np.sqrt((pred_coords[0] - true_coords[0])**2 + 
                                              (pred_coords[1] - true_coords[1])**2),
                        'floor_error': np.sqrt((pred_coords[2] - true_coords[2])**2 + 
                                             (pred_coords[3] - true_coords[3])**2),
                        'front_error': np.sqrt((pred_coords[4] - true_coords[4])**2 + 
                                             (pred_coords[5] - true_coords[5])**2),
                        'side_error': np.sqrt((pred_coords[6] - true_coords[6])**2 + 
                                            (pred_coords[7] - true_coords[7])**2)
                    })
        
        test_csv_path = log_dir / f"{save_file_name}_test_results.csv"
        save_training_log(test_results, str(test_csv_path))
        log_print(f"\n테스트 결과 CSV 저장: {test_csv_path}")
    
    # 로거 닫기
    log_print("\n" + "=" * 60)
    log_print("학습 완료!")
    log_print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"전체 학습 시간: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}")
    log_print(f"최고 검증 손실: {best_val_loss:.6f} (Epoch {best_epoch})")
    log_print(f"최종 테스트 손실: {test_loss:.6f}")
    log_print(f"최종 평균 오차: {np.mean(list(test_errors.values())):.2f} pixels")
    log_print(f"\n저장된 파일:")
    log_print(f"  - 최고 모델: {checkpoint_dir / f'{save_file_name}_best.pth'}")
    log_print(f"  - 모델 정보: {model_info_path}")
    log_print(f"  - 학습 로그: {log_csv_path if pytorch_config['logging']['save_csv'] else 'N/A'}")
    log_print(f"  - 테스트 결과: {test_csv_path if pytorch_config['logging']['save_csv'] else 'N/A'}")
    log_print(f"  - 실행 로그: {log_path}")
    log_print("=" * 60)
    
    logger.close()
    best_logger.close()
    
    # 학습 완료 후 예측 및 시각화 수행
    predict_and_visualize()


def predict_and_visualize():
    """학습된 모델로 예측하고 시각화하여 저장"""
    print("\n" + "=" * 60)
    print("학습 데이터 예측 및 시각화")
    print("=" * 60)
    
    # 설정 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 경로 설정
    base_dir = Path(__file__).parent
    source_path = (base_dir / config['source']['source_folder']).resolve()
    output_dir = base_dir.parent / 'result' / 'learning_output'
    
    # 출력 디렉토리 생성 및 기존 파일 삭제
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리: {output_dir}")
    
    # PyTorch 설정
    pytorch_config = config['pytorch_model']
    
    # 디바이스 설정
    if pytorch_config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{pytorch_config['device']['cuda_device']}")
    else:
        device = torch.device('cpu')
    
    # 모델 파일명 설정
    save_file_name = pytorch_config['checkpointing'].get('save_file_name', 'multi_point_7x7')
    
    # 최고 모델 체크포인트 로드
    best_checkpoint_path = checkpoint_dir / f"{save_file_name}_best.pth"
    if not best_checkpoint_path.exists():
        print(f"모델 파일을 찾을 수 없습니다: {best_checkpoint_path}")
        return
    
    print(f"모델 로드: {best_checkpoint_path}")
    checkpoint = torch.load(str(best_checkpoint_path), map_location=device, weights_only=False)
    
    # 검출기 및 모델 생성
    detector = MultiPointDetectorPyTorch(pytorch_config, device)
    model = detector.model
    
    # 모델 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 특징 정규화 파라미터 로드
    if 'feature_mean' in checkpoint:
        detector.feature_mean = checkpoint['feature_mean']
        detector.feature_std = checkpoint['feature_std']
        feature_mean = checkpoint['feature_mean']
        feature_std = checkpoint['feature_std']
    else:
        feature_mean = None
        feature_std = None
    
    # 좌표 정규화 범위 설정 (학습시와 동일)
    coord_range = pytorch_config['training']['augmentation'].get('coordinate_range', {})
    image_size = tuple(pytorch_config['features']['image_size'])
    width = image_size[0]
    height = image_size[1]
    coord_min_x = coord_range.get('x_min_ratio', -1.0) * width
    coord_max_x = coord_range.get('x_max_ratio', 2.0) * width
    coord_min_y = coord_range.get('y_min_ratio', 0.0) * height
    coord_max_y = coord_range.get('y_max_ratio', 2.0) * height
    
    # coordinate_transform 모듈 import
    sys.path.append(str(Path(__file__).parent / 'utils'))
    from coordinate_transform import convert_predictions_to_original
    
    # labels.txt 파일 읽기
    labels_path = source_path / 'labels.txt'
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 색상 정의 (BGR)
    colors = {
        'center': (0, 0, 255),    # 빨간색
        'floor': (0, 255, 0),      # 초록색
        'front': (255, 0, 0),      # 파란색
        'side': (0, 255, 255)      # 노란색
    }
    
    # test 데이터만 필터링
    test_id_suffix = str(pytorch_config['data_split']['test_id_suffix'])
    test_count = 0
    processed_count = 0
    
    # test 데이터 개수 먼저 카운트
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 11 and parts[0].endswith(test_id_suffix):
            test_count += 1
    
    print(f"\nTest 이미지 처리 시작 (총 {test_count}개)...")
    
    # CSV 저장을 위한 데이터 수집
    csv_data = []
    
    # 헤더 스킵하고 처리
    for idx, line in enumerate(lines[1:], 1):            
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        if len(parts) < 11:
            continue
        
        # 데이터 파싱
        data_id = parts[0]
        filename = parts[2]
        
        # test 데이터만 처리
        if not data_id.endswith(test_id_suffix):
            continue
        
        # 실제 좌표 (원본 이미지 픽셀 좌표)
        true_coords = {
            'center': (int(parts[3]), int(parts[4])),
            'floor': (int(parts[5]) if parts[5] != '0' else int(parts[3]), 
                     int(parts[6]) if parts[6] != '0' else int(parts[4])),
            'front': (int(parts[7]) if parts[7] != '0' else int(parts[3]),
                     int(parts[8]) if parts[8] != '0' else int(parts[4])),
            'side': (int(parts[9]) if parts[9] != '0' else int(parts[3]),
                    int(parts[10]) if parts[10] != '0' else int(parts[4]))
        }
        
        # 이미지 로드
        image_path = source_path / filename
        if not image_path.exists():
            print(f"이미지 파일을 찾을 수 없음: {filename}")
            continue
        
        orig_image = cv2.imread(str(image_path))
        if orig_image is None:
            print(f"이미지 로드 실패: {filename}")
            continue
        
        orig_h, orig_w = orig_image.shape[:2]
        
        # 112x112로 리사이즈
        image_112 = cv2.resize(orig_image, image_size)
        
        # 특징 추출
        features = detector.extract_features(image_112)
        
        # 특징 정규화
        if feature_mean is not None and feature_std is not None:
            features = (features - feature_mean) / (feature_std + 1e-8)
        
        # 텐서 변환 및 배치 차원 추가
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # 예측
        with torch.no_grad():
            outputs = model(features_tensor)
        
        # 정규화된 예측값을 numpy로 변환
        outputs_np = outputs.cpu().numpy()[0]
        
        # 예측 좌표를 원본 이미지 크기로 변환
        pred_coords_dict = convert_predictions_to_original(
            outputs_np,
            orig_w, orig_h,
            coord_min_x, coord_max_x,
            coord_min_y, coord_max_y,
            width, height
        )
        
        # 시각화용 이미지 복사
        vis_image = orig_image.copy()
        
        # 실제 좌표 그리기 (크기 10, 빈 원)
        for name, (x, y) in true_coords.items():
            cv2.circle(vis_image, (int(x), int(y)), 10, colors[name], 1)
        
        # 예측 좌표 그리기 (크기 5, 채워진 원)
        for name, (x, y) in pred_coords_dict.items():
            cv2.circle(vis_image, (int(x), int(y)), 5, colors[name], -1)
        
        # 이미지 저장
        output_path = output_dir / f"pred_{filename}"
        cv2.imwrite(str(output_path), vis_image)
        
        # CSV 데이터 수집
        csv_data.append({
            'filename': filename,
            'label_center_x': true_coords['center'][0],
            'label_center_y': true_coords['center'][1],
            'label_floor_x': true_coords['floor'][0],
            'label_floor_y': true_coords['floor'][1],
            'label_front_x': true_coords['front'][0],
            'label_front_y': true_coords['front'][1],
            'label_side_x': true_coords['side'][0],
            'label_side_y': true_coords['side'][1],
            'pred_center_x': pred_coords_dict['center'][0],
            'pred_center_y': pred_coords_dict['center'][1],
            'pred_floor_x': pred_coords_dict['floor'][0],
            'pred_floor_y': pred_coords_dict['floor'][1],
            'pred_front_x': pred_coords_dict['front'][0],
            'pred_front_y': pred_coords_dict['front'][1],
            'pred_side_x': pred_coords_dict['side'][0],
            'pred_side_y': pred_coords_dict['side'][1]
        })
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  {processed_count}개 이미지 처리 완료...")
    
    # CSV 파일로 저장
    if csv_data:
        csv_path = output_dir / 'output.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['filename', 
                         'label_center_x', 'label_center_y',
                         'label_floor_x', 'label_floor_y',
                         'label_front_x', 'label_front_y',
                         'label_side_x', 'label_side_y',
                         'pred_center_x', 'pred_center_y',
                         'pred_floor_x', 'pred_floor_y',
                         'pred_front_x', 'pred_front_y',
                         'pred_side_x', 'pred_side_y']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"좌표 데이터 CSV 저장: {csv_path}")
        
        # 오차 계산 및 상위 10% 분류
        for data in csv_data:
            # 바닥 중심점 오차
            floor_error = np.sqrt(
                (data['pred_floor_x'] - data['label_floor_x'])**2 + 
                (data['pred_floor_y'] - data['label_floor_y'])**2
            )
            data['floor_error'] = floor_error
            
            # 전체 4개점 평균 오차
            errors = []
            for point in ['center', 'floor', 'front', 'side']:
                error = np.sqrt(
                    (data[f'pred_{point}_x'] - data[f'label_{point}_x'])**2 + 
                    (data[f'pred_{point}_y'] - data[f'label_{point}_y'])**2
                )
                errors.append(error)
            data['total_avg_error'] = np.mean(errors)
        
        # 상위 10% 선별
        top_10_percent = max(1, int(len(csv_data) * 0.1))
        
        # 바닥 중심점 기준 정렬
        worst_floor = sorted(csv_data, key=lambda x: x['floor_error'], reverse=True)[:top_10_percent]
        
        # 전체 평균 기준 정렬
        worst_total = sorted(csv_data, key=lambda x: x['total_avg_error'], reverse=True)[:top_10_percent]
        
        # 디렉토리 생성
        bad_floor_dir = output_dir / 'bed_center'
        bad_total_dir = output_dir / 'bed'
        bad_floor_dir.mkdir(exist_ok=True)
        bad_total_dir.mkdir(exist_ok=True)
        
        # 바닥 중심점 기준 저장
        for i, data in enumerate(worst_floor):
            src = output_dir / f"pred_{data['filename']}"
            if src.exists():
                dst = bad_floor_dir / f"{i+1:03d}_err{data['floor_error']:.1f}_{data['filename']}"
                shutil.copy2(src, dst)
        
        # 바닥 중심점 CSV
        with open(bad_floor_dir / 'analysis.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(worst_floor[0].keys()))
            writer.writeheader()
            writer.writerows(worst_floor)
        
        # 전체 기준 저장
        for i, data in enumerate(worst_total):
            src = output_dir / f"pred_{data['filename']}"
            if src.exists():
                dst = bad_total_dir / f"{i+1:03d}_err{data['total_avg_error']:.1f}_{data['filename']}"
                shutil.copy2(src, dst)
        
        # 전체 CSV
        with open(bad_total_dir / 'analysis.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(worst_total[0].keys()))
            writer.writeheader()
            writer.writerows(worst_total)
        
        print(f"바닥 중심점 오차 상위 10% ({len(worst_floor)}개): {bad_floor_dir}")
        print(f"전체 오차 상위 10% ({len(worst_total)}개): {bad_total_dir}")
    
    print(f"\n총 {processed_count}개 이미지 처리 완료")
    print(f"결과 저장 위치: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()