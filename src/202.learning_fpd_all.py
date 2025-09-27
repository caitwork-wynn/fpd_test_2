# -*- coding: utf-8 -*-
"""
FPD 특징 기반 좌표 회귀 모델 학습 스크립트
- config_fpd.yml에서 모든 설정 읽기
- FPDFeatureExtractor로 특징 추출
- FPDCoordinateRegression 모델 학습
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
from model_defs.fpd_feature_extractor import FPDFeatureExtractor
from model_defs.fpd_coordinate_regression import FPDCoordinateRegression


class DualLogger:
    """화면과 파일에 동시 출력하는 로거 클래스"""

    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message: str):
        """메시지를 화면과 파일에 동시 출력 (타임스탬프 포함)"""
        # 빈 줄이 아닌 경우에만 타임스탬프 추가
        if message.strip():
            timestamp = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
            message_with_time = f"{timestamp} {message}"
        else:
            message_with_time = message

        print(message_with_time, end='')
        if self.log_file:
            self.log_file.write(message_with_time)
            self.log_file.flush()
    
    def close(self):
        """파일 핸들 닫기"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class FPDDataset(Dataset):
    """FPD 특징 기반 좌표 회귀 데이터셋"""
    
    def __init__(self, 
                 config: Dict,
                 feature_extractor: FPDFeatureExtractor,
                 mode: str = 'train',  # 'train', 'val', 'test'
                 augment: bool = True):
        """
        Args:
            config: 설정 딕셔너리
            feature_extractor: FPD 특징 추출기
            mode: 'train', 'val', 'test'
            augment: 데이터 증강 여부
        """
        self.config = config
        self.feature_extractor = feature_extractor
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        # 설정 읽기
        self.source_folder = config['data']['source_folder']
        self.labels_file = config['data']['labels_file']
        self.test_id_suffix = str(config['data_split']['test_id_suffix'])
        self.val_ratio = config['data_split']['validation_ratio']
        self.max_train_images = config['data_split'].get('max_train_images', 0)  # 학습 이미지 수 제한
        self.augment_count = config['training']['augmentation']['augment_count'] if self.augment else 0
        self.noise_std = config['training']['augmentation']['noise_std']
        self.image_size = tuple(config['feature_extraction']['image_size'])
        
        # 크롭 증강 설정
        self.crop_config = config['training']['augmentation'].get('crop', {})
        self.crop_enabled = self.crop_config.get('enabled', False) and self.augment
        
        # 좌표 정규화 범위 설정
        coord_range = config['training']['augmentation'].get('coordinate_range', {})
        width = self.image_size[0]
        height = self.image_size[1]
        self.coord_min_x = coord_range.get('x_min_ratio', -1.0) * width
        self.coord_max_x = coord_range.get('x_max_ratio', 2.0) * width
        self.coord_min_y = coord_range.get('y_min_ratio', 0.0) * height
        self.coord_max_y = coord_range.get('y_max_ratio', 2.0) * height
        
        # 데이터 로드
        self.data = []
        self.load_data()
        
        # 특징과 타겟을 미리 추출하여 저장
        self.features = []
        self.targets = []
        self.metadata = []
        self.precompute_features()
        
        print(f"{mode.upper()} 데이터셋: {len(self.features)}개 샘플")
    
    def load_data(self):
        """레이블 파일에서 데이터 로드 및 분할"""
        labels_path = os.path.join(self.source_folder, self.labels_file)
        
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
            data_id = parts[0]
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

        # 학습 이미지 수 제한 적용
        if self.max_train_images > 0 and len(all_train_data) > self.max_train_images:
            random.seed(self.config['data_split']['random_seed'])
            random.shuffle(all_train_data)
            all_train_data = all_train_data[:self.max_train_images]
            print(f"학습 데이터를 {self.max_train_images}개로 제한했습니다.")

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
        """이미지 크롭 및 좌표 조정"""
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
        
        # 좌표 조정
        adjusted_coords = {}
        scale_x = self.image_size[0] / crop_w
        scale_y = self.image_size[1] / crop_h
        
        for key in ['center', 'floor', 'front', 'side']:
            orig_x = coords_orig[f'{key}_x']
            orig_y = coords_orig[f'{key}_y']
            
            # 크롭 영역 기준으로 조정 후 112x112 스케일 적용
            new_x = (orig_x - x1) * scale_x
            new_y = (orig_y - y1) * scale_y
            
            adjusted_coords[f'{key}_x'] = new_x
            adjusted_coords[f'{key}_y'] = new_y
        
        return cropped_image, adjusted_coords
    
    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """좌표를 min-max 범위로 정규화"""
        norm_x = (x - self.coord_min_x) / (self.coord_max_x - self.coord_min_x)
        norm_y = (y - self.coord_min_y) / (self.coord_max_y - self.coord_min_y)
        
        # 클리핑 (0~1 범위)
        norm_x = np.clip(norm_x, 0.0, 1.0)
        norm_y = np.clip(norm_y, 0.0, 1.0)
        
        return norm_x, norm_y
    
    def denormalize_coordinates(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """정규화된 좌표를 원본 픽셀 좌표로 변환"""
        x = norm_x * (self.coord_max_x - self.coord_min_x) + self.coord_min_x
        y = norm_y * (self.coord_max_y - self.coord_min_y) + self.coord_min_y
        return x, y
    
    def precompute_features(self):
        """모든 이미지에 대해 특징 벡터를 미리 추출"""
        print(f"특징 추출 중... ({self.mode} mode)")
        
        for sample in tqdm(self.data, desc=f"Processing {self.mode} data"):
            img_path = os.path.join(self.source_folder, sample['filename'])
            
            # 이미지 로드
            image = cv2.imread(img_path)
            if image is None:
                print(f"이미지 로드 실패: {img_path}")
                continue
            
            # 원본 샘플 처리
            if not self.crop_enabled:
                # 단순 리사이즈
                image_resized = cv2.resize(image, self.image_size)
                
                # 112x112로 스케일된 좌표 계산
                h_orig, w_orig = image.shape[:2]
                scale_x = self.image_size[0] / w_orig
                scale_y = self.image_size[1] / h_orig
                
                coords = {}
                for key in ['center', 'floor', 'front', 'side']:
                    coords[f'{key}_x'] = sample[f'{key}_x'] * scale_x
                    coords[f'{key}_y'] = sample[f'{key}_y'] * scale_y
                
                self._add_sample(image_resized, coords, sample, is_augmented=False)
            else:
                # 크롭 없이 원본도 추가
                image_resized = cv2.resize(image, self.image_size)
                h_orig, w_orig = image.shape[:2]
                scale_x = self.image_size[0] / w_orig
                scale_y = self.image_size[1] / h_orig
                
                coords = {}
                for key in ['center', 'floor', 'front', 'side']:
                    coords[f'{key}_x'] = sample[f'{key}_x'] * scale_x
                    coords[f'{key}_y'] = sample[f'{key}_y'] * scale_y
                
                self._add_sample(image_resized, coords, sample, is_augmented=False)
            
            # 증강 샘플 생성
            if self.augment:
                for _ in range(self.augment_count):
                    if self.crop_enabled:
                        # 크롭 증강 적용
                        aug_image, aug_coords = self.apply_crop_augmentation(image, sample)
                    else:
                        # 크롭 없이 노이즈만 적용
                        aug_image = image_resized.copy()
                        aug_coords = coords.copy()
                    
                    self._add_sample(aug_image, aug_coords, sample, is_augmented=True)
    
    def _add_sample(self, image: np.ndarray, coords: Dict[str, float], 
                    original_sample: Dict, is_augmented: bool):
        """샘플 추가 (특징 추출 및 저장)"""
        # FPDFeatureExtractor로 특징 추출
        features = self.feature_extractor.get_features(image)
        
        # 노이즈 추가 (증강된 샘플에만)
        if is_augmented and self.noise_std > 0:
            # global features에 노이즈 추가
            global_noise = np.random.normal(0, self.noise_std, features['global_features'].shape)
            features['global_features'] = features['global_features'] + global_noise
            
            # patch features에 노이즈 추가
            opencv_noise = np.random.normal(0, self.noise_std, features['patch_opencv_features'].shape)
            features['patch_opencv_features'] = features['patch_opencv_features'] + opencv_noise
        
        # 타겟 좌표 정규화
        target = []
        for key in ['center', 'floor', 'front', 'side']:
            x_norm, y_norm = self.normalize_coordinates(coords[f'{key}_x'], coords[f'{key}_y'])
            target.extend([x_norm, y_norm])
        
        # 저장
        self.features.append(features)
        self.targets.append(np.array(target, dtype=np.float32))
        self.metadata.append({
            'id': original_sample['id'],
            'filename': original_sample['filename'],
            'is_augmented': is_augmented,
            'raw_coords': coords  # 정규화 전 좌표 (픽셀 단위)
        })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.targets[idx]
        
        # numpy to tensor
        features_tensor = {
            'global_features': torch.FloatTensor(features['global_features']),
            'patch_opencv_features': torch.FloatTensor(features['patch_opencv_features']),
            'patch_latent_features': torch.FloatTensor(features['patch_latent_features'])
        }
        target_tensor = torch.FloatTensor(target)
        
        return features_tensor, target_tensor


def create_combined_loss(config: Dict):
    """조합된 손실 함수 생성"""
    class_weight = config['training']['loss']['classification_weight']
    reg_weight = config['training']['loss']['regression_weight']
    
    def combined_loss(model: FPDCoordinateRegression, 
                      outputs: Dict[str, torch.Tensor], 
                      targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model: FPDCoordinateRegression 모델
            outputs: 모델 출력 (logits, coordinates)
            targets: 정규화된 타겟 좌표 [B, 8]
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # 타겟을 픽셀 좌표로 변환 (soft label 생성용)
        targets_pixel = targets.clone()
        for i in range(batch_size):
            for j in range(8):
                if j % 2 == 0:  # x 좌표
                    # 정규화된 값을 픽셀 값으로 변환
                    targets_pixel[i, j] = targets[i, j] * (model.x_range[1] - model.x_range[0]) + model.x_range[0]
                else:  # y 좌표
                    targets_pixel[i, j] = targets[i, j] * (model.y_range[1] - model.y_range[0]) + model.y_range[0]
        
        # Classification loss
        class_loss = 0
        for i, head in enumerate(model.coordinate_heads):
            # Soft label 생성
            soft_labels = head.create_soft_label(
                targets_pixel[:, i], 
                sigma=config['training']['loss']['soft_label_sigma']
            )
            
            # Cross entropy loss
            logits = outputs['logits'][:, i]  # [B, max_classes]
            if i % 2 == 0:  # x 좌표
                logits = logits[:, :model.x_classes]
            else:  # y 좌표
                logits = logits[:, :model.y_classes]
            
            class_loss += -(soft_labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        
        class_loss /= 8  # 평균
        
        # Regression loss (예측된 픽셀 좌표와 타겟 픽셀 좌표 간 MSE)
        predicted_pixel = outputs['coordinates']  # 이미 픽셀 좌표
        reg_loss = nn.functional.mse_loss(predicted_pixel, targets_pixel)
        
        # 조합된 손실
        total_loss = class_weight * class_loss + reg_weight * reg_loss
        
        return total_loss
    
    return combined_loss


def train_epoch(model: FPDCoordinateRegression,
                dataloader: DataLoader,
                criterion,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                config: Dict) -> float:
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (features, targets) in enumerate(dataloader):
        # 디바이스로 이동
        features = {k: v.to(device) for k, v in features.items()}
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(features)
        
        # Loss 계산
        loss = criterion(model, outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          config['training']['gradient_clip'])
        
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model: FPDCoordinateRegression,
            dataloader: DataLoader,
            criterion,
            device: torch.device,
            config: Dict,
            dataset: FPDDataset,
            mode: str = 'val') -> Dict[str, float]:
    """검증/테스트 수행 (상세 오차 분석 포함)"""
    model.eval()
    running_loss = 0.0
    mae_sum = 0.0
    mae_coords = {'center': 0.0, 'floor': 0.0, 'front': 0.0, 'side': 0.0}
    total_samples = 0

    # 상세 오차 수집용
    all_errors = {
        'center': {'x': [], 'y': [], 'dist': []},
        'floor': {'x': [], 'y': [], 'dist': []},
        'front': {'x': [], 'y': [], 'dist': []},
        'side': {'x': [], 'y': [], 'dist': []}
    }

    # 테스트 결과 저장용
    test_results = [] if mode == 'test' else None

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(dataloader):
            # 디바이스로 이동
            features = {k: v.to(device) for k, v in features.items()}
            targets = targets.to(device)
            batch_size = targets.size(0)

            # 메타데이터 가져오기 (test 모드에서만)
            if mode == 'test' and test_results is not None:
                start_idx = batch_idx * dataloader.batch_size
                batch_metadata = dataset.metadata[start_idx:start_idx + batch_size]

            # Forward pass
            outputs = model(features)

            # Loss 계산
            loss = criterion(model, outputs, targets)
            running_loss += loss.item()

            # MAE 계산 (픽셀 단위)
            predicted = outputs['coordinates'].cpu().numpy()  # 이미 픽셀 좌표
            targets_np = targets.cpu().numpy()

            # 타겟을 픽셀 좌표로 변환
            targets_pixel = np.zeros_like(targets_np)
            for i in range(batch_size):
                for j in range(8):
                    norm_val = targets_np[i, j]
                    if j % 2 == 0:  # x 좌표
                        targets_pixel[i, j] = norm_val * (model.x_range[1] - model.x_range[0]) + model.x_range[0]
                    else:  # y 좌표
                        targets_pixel[i, j] = norm_val * (model.y_range[1] - model.y_range[0]) + model.y_range[0]

            # MAE 계산
            mae = np.abs(predicted - targets_pixel).mean()
            mae_sum += mae * batch_size

            # 각 포인트별 상세 오차 수집
            for i in range(batch_size):
                sample_result = {} if test_results is not None else None

                if test_results is not None:
                    metadata = batch_metadata[i]
                    sample_result['id'] = metadata['id']
                    sample_result['filename'] = metadata['filename']

                for point_idx, point_name in enumerate(['center', 'floor', 'front', 'side']):
                    x_idx = point_idx * 2
                    y_idx = point_idx * 2 + 1

                    pred_x = predicted[i, x_idx]
                    pred_y = predicted[i, y_idx]
                    true_x = targets_pixel[i, x_idx]
                    true_y = targets_pixel[i, y_idx]

                    # 오차 계산
                    x_error = pred_x - true_x
                    y_error = pred_y - true_y
                    dist_error = np.sqrt(x_error**2 + y_error**2)

                    # 상세 오차 저장
                    all_errors[point_name]['x'].append(abs(x_error))
                    all_errors[point_name]['y'].append(abs(y_error))
                    all_errors[point_name]['dist'].append(dist_error)

                    mae_coords[point_name] += dist_error

                    # 테스트 결과 저장
                    if test_results is not None:
                        sample_result[f'true_{point_name}_x'] = true_x
                        sample_result[f'true_{point_name}_y'] = true_y
                        sample_result[f'pred_{point_name}_x'] = pred_x
                        sample_result[f'pred_{point_name}_y'] = pred_y
                        sample_result[f'{point_name}_error'] = dist_error

                if test_results is not None:
                    test_results.append(sample_result)

            total_samples += batch_size

    avg_loss = running_loss / len(dataloader)
    avg_mae = mae_sum / total_samples

    # 각 포인트별 평균 MAE
    for key in mae_coords:
        mae_coords[key] /= total_samples

    results = {
        'loss': avg_loss,
        'mae': avg_mae,
        'mae_center': mae_coords['center'],
        'mae_floor': mae_coords['floor'],
        'mae_front': mae_coords['front'],
        'mae_side': mae_coords['side'],
        'all_errors': all_errors,
        'test_results': test_results
    }

    return results


def save_error_raw_data(epoch: int, test_metrics: Dict, config: Dict, timestamp: str, logger: DualLogger):
    """오차 원천 데이터를 CSV로 저장"""
    error_config = config['training'].get('error_analysis', {})
    if not error_config.get('save_raw_data', False):
        return

    # 디렉토리 생성
    results_dir = error_config.get('results_dir', '../results')
    model_name = config['checkpointing']['save_file_name']
    session_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    # CSV 파일 경로 (final일 경우 특별 처리)
    if isinstance(epoch, str) and epoch == 'final':
        csv_path = os.path.join(session_dir, "final_test_errors.csv")
    else:
        csv_path = os.path.join(session_dir, f"epoch_{epoch:06d}_errors.csv")

    # 데이터를 긴 형식(long format)으로 변환
    rows = []
    if test_metrics.get('test_results'):
        for result in test_metrics['test_results']:
            sample_id = result['id']
            filename = result['filename']

            for point in ['center', 'floor', 'front', 'side']:
                rows.append({
                    'epoch': epoch,
                    'id': sample_id,
                    'filename': filename,
                    'point': point,
                    'true_x': result[f'true_{point}_x'],
                    'true_y': result[f'true_{point}_y'],
                    'pred_x': result[f'pred_{point}_x'],
                    'pred_y': result[f'pred_{point}_y'],
                    'error_x': result[f'pred_{point}_x'] - result[f'true_{point}_x'],
                    'error_y': result[f'pred_{point}_y'] - result[f'true_{point}_y'],
                    'error_dist': result[f'{point}_error']
                })

    # CSV 저장
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.write(f"오차 원천 데이터 저장: {csv_path}\n")


def perform_error_analysis(epoch: int, model: FPDCoordinateRegression,
                          test_loader: DataLoader, criterion, device: torch.device,
                          config: Dict, test_dataset: FPDDataset, timestamp: str,
                          logger: DualLogger):
    """주기적 오차 분석 및 저장"""

    # 테스트 데이터 검증
    test_metrics = validate(model, test_loader, criterion, device, config,
                           test_dataset, 'test')

    # 오차 분석 로그 출력을 위한 문자열 준비
    all_errors = test_metrics['all_errors']
    output_lines = []
    output_lines.append(f"=== Epoch {epoch} 오차 분석 ===")

    # 각 포인트별 통계 출력
    for name in ['center', 'floor', 'front', 'side']:
        if name in all_errors and all_errors[name]['x']:
            x_mean = np.mean(all_errors[name]['x'])
            x_std = np.std(all_errors[name]['x'])
            y_mean = np.mean(all_errors[name]['y'])
            y_std = np.std(all_errors[name]['y'])
            dist_mean = np.mean(all_errors[name]['dist'])
            dist_std = np.std(all_errors[name]['dist'])

            # 탭으로 정렬된 형식
            line = f"{name.upper():6s}: X={x_mean:4.1f}±{x_std:4.1f},\tY={y_mean:4.1f}±{y_std:4.1f},\tDist={dist_mean:4.1f}±{dist_std:4.1f}"
            output_lines.append(line)

    # 전체 평균 거리 오차
    all_distances = []
    for errors in all_errors.values():
        all_distances.extend(errors['dist'])
    if all_distances:
        overall_mean = np.mean(all_distances)
        overall_std = np.std(all_distances)
        output_lines.append(f"전체: 평균={overall_mean:.2f}±{overall_std:.2f} pixels")

    # 비고 추가
    output_lines.append("비고) 값:좌표 오차 평균±표준편차, Dist:유클리드 거리 오차")

    # 한 번에 출력 (타임스탬프는 첫 줄과 마지막 줄에만)
    logger.write("\n" + "\n".join(output_lines) + "\n")
    logger.write("=====================\n")

    # 원천 데이터 저장
    save_error_raw_data(epoch, test_metrics, config, timestamp, logger)

    return test_metrics


def save_model_as_onnx(model: FPDCoordinateRegression,
                       feature_extractor: FPDFeatureExtractor,
                       save_path: str,
                       config: Dict):
    """모델을 ONNX 형식으로 저장"""
    model.eval()
    
    # 더미 입력 생성
    dummy_features = {
        'global_features': torch.randn(1, 21),
        'patch_opencv_features': torch.randn(1, 49, 21),
        'patch_latent_features': torch.randn(1, 49, 16)
    }
    
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
        model = model.to(device)
        dummy_features = {k: v.to(device) for k, v in dummy_features.items()}
    
    # ONNX 변환을 위한 wrapper 클래스
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, global_feat, opencv_feat, latent_feat):
            features = {
                'global_features': global_feat,
                'patch_opencv_features': opencv_feat,
                'patch_latent_features': latent_feat
            }
            outputs = self.model(features)
            return outputs['coordinates']
    
    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()
    
    # ONNX 변환
    torch.onnx.export(
        wrapped_model,
        (dummy_features['global_features'], 
         dummy_features['patch_opencv_features'],
         dummy_features['patch_latent_features']),
        save_path,
        input_names=['global_features', 'patch_opencv_features', 'patch_latent_features'],
        output_names=['coordinates'],
        dynamic_axes={
            'global_features': {0: 'batch_size'},
            'patch_opencv_features': {0: 'batch_size'},
            'patch_latent_features': {0: 'batch_size'},
            'coordinates': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"ONNX 모델 저장 완료: {save_path}")


def main(resume_checkpoint=None):
    """메인 학습 함수

    Args:
        resume_checkpoint: 재개할 체크포인트 경로 (None이면 새로 시작)
    """
    # 설정 로드
    config_path = 'config_fpd.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 체크포인트 경로 설정 (우선순위: 명령줄 > config > 자동 탐색)
    checkpoint_dir = config['checkpointing']['save_dir']
    checkpoint_name = config['checkpointing']['save_file_name']

    if resume_checkpoint:
        # 1. 명령줄 인자가 최우선
        config['resume_from_checkpoint'] = resume_checkpoint
    elif config.get('checkpointing', {}).get('resume_from'):
        resume_setting = config['checkpointing']['resume_from']
        if resume_setting == 'auto':
            # 2. 'auto' 설정시 자동 탐색
            best_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_best.pth")
            regular_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pth")

            if os.path.exists(best_path):
                config['resume_from_checkpoint'] = best_path
                print(f"자동으로 best 체크포인트 발견: {best_path}")
            elif os.path.exists(regular_path):
                config['resume_from_checkpoint'] = regular_path
                print(f"자동으로 체크포인트 발견: {regular_path}")
            else:
                config['resume_from_checkpoint'] = None
                print("체크포인트를 찾을 수 없어 새로 시작합니다.")
        elif resume_setting and resume_setting != 'null':
            # 3. 특정 경로가 지정된 경우
            config['resume_from_checkpoint'] = resume_setting
        else:
            # 4. null이면 새로 시작
            config['resume_from_checkpoint'] = None
    else:
        config['resume_from_checkpoint'] = None
    
    # 로그 디렉토리 생성
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['checkpointing']['save_dir'], exist_ok=True)

    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 오차 분석 결과 디렉토리 생성
    error_config = config['training'].get('error_analysis', {})
    if error_config.get('enabled', False) and error_config.get('save_raw_data', False):
        results_dir = error_config.get('results_dir', '../results')
        model_name = config['checkpointing']['save_file_name']
        session_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        print(f"오차 분석 데이터 저장 디렉토리: {session_dir}")
    
    # 로거 설정
    log_path = os.path.join(config['logging']['log_dir'], 
                            f'training_fpd_all_{timestamp}.log')
    logger = DualLogger(log_path)
    
    logger.write(f"=== FPD Coordinate Regression Training ===\n")
    logger.write(f"Timestamp: {timestamp}\n")
    logger.write(f"Config: {config_path}\n\n")
    
    # 디바이스 설정
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
        logger.write(f"Using CUDA device: {device}\n")
    else:
        device = torch.device('cpu')
        logger.write("Using CPU\n")
    
    # FPD Feature Extractor 초기화
    logger.write("\n특징 추출기 초기화 중...\n")
    feature_extractor = FPDFeatureExtractor(config_path)
    
    # 데이터셋 생성
    logger.write("\n데이터셋 생성 중...\n")
    train_dataset = FPDDataset(config, feature_extractor, mode='train', augment=True)
    val_dataset = FPDDataset(config, feature_extractor, mode='val', augment=False)
    test_dataset = FPDDataset(config, feature_extractor, mode='test', augment=False)
    
    logger.write(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, 
                             batch_size=config['training']['batch_size'],
                             shuffle=True,
                             num_workers=0)
    val_loader = DataLoader(val_dataset, 
                           batch_size=config['training']['batch_size'],
                           shuffle=False,
                           num_workers=0)
    test_loader = DataLoader(test_dataset, 
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            num_workers=0)
    
    # 모델 생성
    logger.write("\n모델 생성 중...\n")
    model = FPDCoordinateRegression(config)
    model = model.to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write(f"Total parameters: {total_params:,}\n")
    logger.write(f"Trainable parameters: {trainable_params:,}\n")
    
    # 손실 함수 및 옵티마이저 설정
    criterion = create_combined_loss(config)
    
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['training']['learning_rate'],
                              weight_decay=config['training']['weight_decay'])
    elif config['training']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                               lr=config['training']['learning_rate'],
                               weight_decay=config['training']['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), 
                             lr=config['training']['learning_rate'],
                             momentum=0.9,
                             weight_decay=config['training']['weight_decay'])
    
    # 스케줄러 설정
    if config['training']['scheduler']['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            min_lr=config['training']['scheduler']['min_lr']
        )
    else:
        scheduler = None
    
    # 체크포인트 로드 (있는 경우)
    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'test_mae': [],
        'learning_rate': []
    }

    checkpoint_path = config.get('resume_from_checkpoint', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.write(f"\n=== 체크포인트 로드 ===\n")
        logger.write(f"체크포인트 파일: {checkpoint_path}\n")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        best_epoch = checkpoint.get('best_epoch', checkpoint['epoch'])
        patience_counter = checkpoint.get('patience_counter', 0)

        # 학습 기록이 있으면 로드
        if 'training_history' in checkpoint:
            training_history = checkpoint['training_history']

        logger.write(f"이전 학습 상태 복원 완료\n")
        logger.write(f"- 시작 에폭: {start_epoch}\n")
        logger.write(f"- Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})\n")
        logger.write(f"- Patience counter: {patience_counter}\n")

        # 옵티마이저를 디바이스로 이동
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        if checkpoint_path:
            logger.write(f"\n경고: 지정된 체크포인트를 찾을 수 없음: {checkpoint_path}\n")
        logger.write("\n=== 새로운 학습 시작 ===\n")

    logger.write(f"\n=== 학습 진행 중 ===\n")
    start_time = time.time()

    # 학습 루프
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        epoch_start = time.time()
        
        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # 검증
        val_metrics = validate(model, val_loader, criterion, device, config, val_dataset, 'val')
        
        # 현재 학습률
        current_lr = optimizer.param_groups[0]['lr']
        
        # 기록 저장
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_mae'].append(val_metrics['mae'])
        training_history['learning_rate'].append(current_lr)
        
        # 스케줄러 업데이트
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])
        
        # Best model 체크
        if val_metrics['loss'] < best_val_loss - config['training']['early_stopping']['min_delta']:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            
            # Best model 저장
            if config['checkpointing']['save_best_only']:
                save_path = os.path.join(config['checkpointing']['save_dir'],
                                       f"{config['checkpointing']['save_file_name']}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_mae': val_metrics['mae'],
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'training_history': training_history,
                    'config': config
                }, save_path)
                logger.write(f"Best model saved at epoch {epoch}\n")
        else:
            patience_counter += 1
        
        # 정기 저장
        if epoch % config['checkpointing']['save_frequency'] == 0:
            save_path = os.path.join(config['checkpointing']['save_dir'],
                                   f"{config['checkpointing']['save_file_name']}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'training_history': training_history,
                'config': config
            }, save_path)
        
        # 진행 상황 출력
        if epoch % config['logging']['progress_interval'] == 0:
            epoch_time = time.time() - epoch_start
            logger.write(f"Epoch [{epoch}/{config['training']['epochs']}] "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_metrics['loss']:.6f}, "
                       f"Val MAE: {val_metrics['mae']:.2f}, LR: {current_lr:.6f}, "
                       f"Time: {epoch_time:.2f}s\n")
        
        # 상세 정보 출력
        if epoch % config['logging']['detail_interval'] == 0:
            test_metrics = validate(model, test_loader, criterion, device, config, test_dataset, 'test')
            training_history['test_mae'].append(test_metrics['mae'])

            logger.write(f"\n=== Epoch {epoch} Detail ===\n")
            logger.write(f"Val MAE by point: Center={val_metrics['mae_center']:.2f}, "
                       f"Floor={val_metrics['mae_floor']:.2f}, "
                       f"Front={val_metrics['mae_front']:.2f}, "
                       f"Side={val_metrics['mae_side']:.2f}\n")
            logger.write(f"Test MAE: {test_metrics['mae']:.2f}\n")
            logger.write(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.6f})\n\n")

        # 오차 분석 실행
        error_config = config['training'].get('error_analysis', {})
        if error_config.get('enabled', False):
            interval = error_config.get('interval', 500)
            if epoch % interval == 0:
                logger.write(f"\n주기적 오차 분석 실행 (Epoch {epoch})\n")
                perform_error_analysis(epoch, model, test_loader, criterion, device,
                                     config, test_dataset, timestamp, logger)
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            logger.write(f"\nEarly stopping triggered at epoch {epoch}\n")
            break
    
    # 학습 완료
    total_time = time.time() - start_time
    logger.write(f"\n=== 학습 완료 ===\n")
    logger.write(f"Total training time: {total_time/3600:.2f} hours\n")
    logger.write(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}\n")
    
    # 최종 테스트 평가
    logger.write("\n=== 테스트 데이터 최종 평가 ===\n")

    # Best model 로드
    best_model_path = os.path.join(config['checkpointing']['save_dir'],
                                  f"{config['checkpointing']['save_file_name']}_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.write(f"Best model loaded from epoch {checkpoint['epoch']}\n")

    test_metrics = validate(model, test_loader, criterion, device, config, test_dataset, 'test')

    # 기본 메트릭
    logger.write(f"\nTest Loss: {test_metrics['loss']:.6f}\n")
    logger.write(f"Test MAE: {test_metrics['mae']:.2f} pixels\n")

    # 각 포인트별 오차 분석
    logger.write("\n각 포인트별 오차 분석 (pixels):\n")
    all_errors = test_metrics['all_errors']
    for name in ['center', 'floor', 'front', 'side']:
        if name in all_errors and all_errors[name]['x']:
            x_mean = np.mean(all_errors[name]['x'])
            x_std = np.std(all_errors[name]['x'])
            y_mean = np.mean(all_errors[name]['y'])
            y_std = np.std(all_errors[name]['y'])
            dist_mean = np.mean(all_errors[name]['dist'])
            dist_std = np.std(all_errors[name]['dist'])

            logger.write(f"  {name.upper():6s}:\n")
            logger.write(f"    X 오차: 평균={x_mean:.1f} 표준편차={x_std:.1f}\n")
            logger.write(f"    Y 오차: 평균={y_mean:.1f} 표준편차={y_std:.1f}\n")
            logger.write(f"    거리: 평균={dist_mean:.1f} 표준편차={dist_std:.1f}\n")

    # 전체 평균 거리 오차
    all_distances = []
    for errors in all_errors.values():
        all_distances.extend(errors['dist'])
    overall_mean_dist = np.mean(all_distances) if all_distances else 0
    logger.write(f"\n전체 평균 거리 오차: {overall_mean_dist:.2f} pixels\n")

    # 상세 통계
    logger.write("\n상세 통계 (거리 기준):\n")
    for name in ['center', 'floor', 'front', 'side']:
        if name in all_errors and all_errors[name]['dist']:
            dist_array = np.array(all_errors[name]['dist'])
            logger.write(f"\n{name.upper()}:\n")
            logger.write(f"  평균: {np.mean(dist_array):.2f}\n")
            logger.write(f"  표준편차: {np.std(dist_array):.2f}\n")
            logger.write(f"  중앙값: {np.median(dist_array):.2f}\n")
            logger.write(f"  최소: {np.min(dist_array):.2f}\n")
            logger.write(f"  최대: {np.max(dist_array):.2f}\n")
            logger.write(f"  25% 분위수: {np.percentile(dist_array, 25):.2f}\n")
            logger.write(f"  75% 분위수: {np.percentile(dist_array, 75):.2f}\n")
    
    # 테스트 결과 CSV 저장
    if config['logging']['save_csv'] and test_metrics.get('test_results'):
        test_csv_path = os.path.join(config['logging']['log_dir'],
                                    f'{config["checkpointing"]["save_file_name"]}_test_results_{timestamp}.csv')
        test_results = test_metrics['test_results']

        if test_results:
            fieldnames = list(test_results[0].keys())
            with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(test_results)
            logger.write(f"\n테스트 결과 CSV 저장: {test_csv_path}\n")
            logger.write(f"  - 총 {len(test_results)}개 샘플\n")

    # 최종 오차 원천 데이터 저장
    error_config = config['training'].get('error_analysis', {})
    if error_config.get('save_raw_data', False):
        save_error_raw_data('final', test_metrics, config, timestamp, logger)

    # 학습 기록 저장 (CSV)
    if config['logging']['save_csv']:
        csv_path = os.path.join(config['logging']['log_dir'],
                              f'training_history_fpd_all_{timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=training_history.keys())
            writer.writeheader()
            for i in range(len(training_history['epoch'])):
                row = {key: training_history[key][i] if i < len(training_history[key]) else ''
                      for key in training_history.keys()}
                writer.writerow(row)
        logger.write(f"\n학습 기록 저장: {csv_path}\n")
    
    # 모델 정보 JSON 저장
    model_info = {
        'config_file': config_path,
        'timestamp': timestamp,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'final_epoch': epoch,
        'total_training_time': f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}",
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'test_results': {
            'loss': float(test_metrics['loss']),
            'mae': float(test_metrics['mae']),
            'mae_center': float(test_metrics['mae_center']),
            'mae_floor': float(test_metrics['mae_floor']),
            'mae_front': float(test_metrics['mae_front']),
            'mae_side': float(test_metrics['mae_side']),
            'overall_mean_distance': float(overall_mean_dist)
        }
    }

    model_info_path = os.path.join(config['checkpointing']['save_dir'],
                                  f"{config['checkpointing']['save_file_name']}_info.json")
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    logger.write(f"\n모델 정보 JSON 저장: {model_info_path}\n")

    # ONNX 변환
    logger.write("\nONNX 모델 변환 중...\n")
    onnx_path = os.path.join(config['checkpointing']['save_dir'],
                            f"{config['checkpointing']['save_file_name']}.onnx")
    try:
        save_model_as_onnx(model, feature_extractor, onnx_path, config)
    except Exception as e:
        logger.write(f"ONNX 변환 실패: {e}\n")

    # 최종 요약
    logger.write("\n" + "=" * 60 + "\n")
    logger.write("학습 완료!\n")
    logger.write(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.write(f"전체 학습 시간: {model_info['total_training_time']}\n")
    logger.write(f"최고 검증 손실: {best_val_loss:.6f} (Epoch {best_epoch})\n")
    logger.write(f"최종 테스트 손실: {test_metrics['loss']:.6f}\n")
    logger.write(f"최종 평균 오차: {overall_mean_dist:.2f} pixels\n")
    logger.write(f"\n저장된 파일:\n")
    logger.write(f"  - 최고 모델: {best_model_path}\n")
    logger.write(f"  - 모델 정보: {model_info_path}\n")
    logger.write(f"  - 학습 로그: {csv_path if config['logging']['save_csv'] else 'N/A'}\n")
    if config['logging']['save_csv'] and test_metrics.get('test_results'):
        logger.write(f"  - 테스트 결과: {test_csv_path}\n")
    logger.write(f"  - 실행 로그: {log_path}\n")
    logger.write("=" * 60 + "\n")

    logger.close()
    print("\n학습이 완료되었습니다!")


def predict_and_visualize(model_path: str, 
                          config_path: str,
                          test_images: List[str] = None,
                          output_dir: str = '../output'):
    """학습된 모델로 예측 및 시각화"""
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 디바이스 설정
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
    else:
        device = torch.device('cpu')
    
    # Feature extractor 및 모델 로드
    feature_extractor = FPDFeatureExtractor(config_path)
    model = FPDCoordinateRegression(config)
    
    # 모델 가중치 로드
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 테스트 이미지 처리
    if test_images is None:
        # 데이터셋에서 몇 개 샘플 가져오기
        test_dataset = FPDDataset(config, feature_extractor, mode='test', augment=False)
        test_images = [test_dataset.metadata[i]['filename'] for i in range(min(5, len(test_dataset)))]
    
    for img_file in test_images:
        img_path = os.path.join(config['data']['source_folder'], img_file)
        
        # 이미지 로드 및 전처리
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # 112x112로 리사이즈
        image_resized = cv2.resize(image, tuple(config['feature_extraction']['image_size']))
        
        # 특징 추출
        features = feature_extractor.get_features(image_resized)
        
        # Tensor 변환
        features_tensor = {
            'global_features': torch.FloatTensor(features['global_features']).unsqueeze(0).to(device),
            'patch_opencv_features': torch.FloatTensor(features['patch_opencv_features']).unsqueeze(0).to(device),
            'patch_latent_features': torch.FloatTensor(features['patch_latent_features']).unsqueeze(0).to(device)
        }
        
        # 예측
        with torch.no_grad():
            outputs = model(features_tensor)
            predicted_coords = outputs['coordinates'][0].cpu().numpy()  # [8] 픽셀 좌표
        
        # 시각화
        vis_image = image_resized.copy()
        
        # 원본 이미지 크기에 맞게 스케일 조정
        h_orig, w_orig = image.shape[:2]
        scale_x = w_orig / 112
        scale_y = h_orig / 112
        
        # 4개 포인트 그리기
        point_names = ['Center', 'Floor', 'Front', 'Side']
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, (name, color) in enumerate(zip(point_names, colors)):
            x = int(predicted_coords[i * 2])
            y = int(predicted_coords[i * 2 + 1])
            
            # 112x112 이미지에 그리기
            cv2.circle(vis_image, (x, y), 3, color, -1)
            cv2.putText(vis_image, name, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 저장
        output_path = os.path.join(output_dir, f"pred_{os.path.basename(img_file)}")
        cv2.imwrite(output_path, vis_image)
        print(f"Saved prediction: {output_path}")
        
        # 좌표 출력
        print(f"\nPredicted coordinates for {img_file}:")
        for i, name in enumerate(point_names):
            x = predicted_coords[i * 2]
            y = predicted_coords[i * 2 + 1]
            print(f"  {name}: ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FPD Coordinate Regression Training')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict'],
                       help='Mode: train or predict')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path for prediction')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint path to resume training from')
    parser.add_argument('--config', type=str, default='config_fpd.yml',
                       help='Config file path')
    parser.add_argument('--output', type=str, default='../output',
                       help='Output directory for predictions')

    args = parser.parse_args()

    if args.mode == 'train':
        main(resume_checkpoint=args.resume)
    elif args.mode == 'predict':
        if args.model is None:
            # 기본 모델 경로 사용
            args.model = '../model/fpd_all_best.pth'
        predict_and_visualize(args.model, args.config, output_dir=args.output)