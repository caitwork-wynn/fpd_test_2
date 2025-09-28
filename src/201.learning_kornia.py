# -*- coding: utf-8 -*-
"""
Kornia 기반 다중 포인트 검출 모델 학습 스크립트
- config_kornia.yml에서 모든 설정 읽기
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
import warnings

# TracerWarning 및 ONNX 관련 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
warnings.filterwarnings("ignore", category=UserWarning, module="kornia")
warnings.filterwarnings("ignore", message=".*TracerWarning.*")
warnings.filterwarnings("ignore", message=".*trace might not generalize.*")
warnings.filterwarnings("ignore", message=".*constant folding.*")

# 모델 import
sys.path.append(str(Path(__file__).parent))
from model_defs.multi_point_model_kornia import MultiPointDetectorKornia


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
    """다중 포인트 검출 데이터셋 (Kornia 버전)"""

    def __init__(self,
                 source_folder: str,
                 labels_file: str,
                 detector: MultiPointDetectorKornia,
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

        # 이미지와 타겟 저장 (메모리에 미리 로드)
        self.images = []
        self.targets = []
        self.metadata = []
        self.precompute_data()

        print(f"{mode.upper()} 데이터셋: {len(self.images)}개 샘플")

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

    def precompute_data(self):
        """모든 데이터를 미리 로드하여 저장 (Kornia는 이미지 직접 처리)"""
        print(f"{self.mode.upper()} 데이터셋 로드 중...")

        for sample in tqdm(self.data, desc="데이터 로드", disable=False):
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

            # BGR to RGB
            image_rgb = cv2.cvtColor(image_112, cv2.COLOR_BGR2RGB)

            # HWC to CHW
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()

            # 좌표 정규화 (min-max 방식)
            targets = []
            for key in ['center', 'floor', 'front', 'side']:
                x = coords_112[f'{key}_x']
                y = coords_112[f'{key}_y']
                norm_x, norm_y = self.normalize_coordinates(x, y)
                targets.extend([norm_x, norm_y])

            targets = np.array(targets, dtype=np.float32)

            # 원본 데이터 저장
            self.images.append(image_tensor)
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

                    # BGR to RGB
                    cropped_rgb = cv2.cvtColor(cropped_112, cv2.COLOR_BGR2RGB)

                    # HWC to CHW
                    cropped_tensor = torch.from_numpy(cropped_rgb).permute(2, 0, 1).float()

                    # 좌표 정규화
                    targets_aug = []
                    for key in ['center', 'floor', 'front', 'side']:
                        x = coords_cropped[f'{key}_x']
                        y = coords_cropped[f'{key}_y']
                        norm_x, norm_y = self.normalize_coordinates(x, y)
                        targets_aug.extend([norm_x, norm_y])

                    targets_aug = np.array(targets_aug, dtype=np.float32)

                    # 증강 데이터 저장
                    self.images.append(cropped_tensor)
                    self.targets.append(torch.FloatTensor(targets_aug))
                    self.metadata.append({
                        'filename': sample['filename'],
                        'id': sample['id'],
                        'augmented': True,
                        'aug_idx': aug_idx + 1
                    })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """미리 로드된 이미지와 타겟 반환"""
        image = self.images[idx].clone()
        targets = self.targets[idx].clone()
        metadata = self.metadata[idx]

        return {
            'image': image,
            'targets': targets,
            'filename': metadata['filename'],
            'id': metadata['id']
        }


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """1 에폭 학습"""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)

        # Forward (모델이 내부적으로 특징 추출)
        outputs = model(images)

        # FPD 모델: 분류 기반 회귀 손실
        loss = model.compute_loss(outputs, targets, sigma=1.0)

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
            images = batch['image'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(images)

            # FPD 모델: 분류 기반 회귀 손실
            loss = model.compute_loss(outputs, targets, sigma=1.0)
            # 연속 좌표 추출
            outputs_coords = outputs['coordinates']  # [B, 8]

            total_loss += loss.item()

            # 픽셀 단위 오차 계산
            outputs_np = outputs_coords.cpu().numpy()
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

        # 모델을 CPU로 이동하고 평가 모드로 설정 (디바이스 충돌 방지)
        original_device = device  # 원래 디바이스 저장
        model = model.cpu()
        model.eval()

        # 더미 입력 생성 (batch_size=1, 이미지 입력, CPU에서)
        dummy_input = torch.randn(1, 3, 112, 112)  # device 지정 제거

        # ONNX로 변환 (Kornia 호환성 개선)
        with torch.no_grad():
            # ONNX export 시 모든 경고 억제
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,  # 최신 opset 사용
                    do_constant_folding=True,
                    input_names=['image'],
                    output_names=['coordinates'],
                    dynamic_axes={
                        'image': {0: 'batch_size'},
                        'coordinates': {0: 'batch_size'}
                    },
                    # ONNX_ATEN_FALLBACK 대신 표준 ONNX 사용 (ATen 연산자 없이)
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                    verbose=False  # verbose 출력 비활성화
                )

        # 모델을 원래 디바이스로 복원하고 학습 모드로 설정
        model = model.to(original_device)
        model.train()

        return onnx_path
    except Exception as e:
        print(f"ONNX 변환 실패: {e}")
        # 에러 발생 시에도 모델을 원래 디바이스로 복원
        if 'original_device' in locals():
            model = model.to(original_device)
            model.train()
        return None


def main():
    """메인 함수"""
    print("=" * 60)
    print("Kornia 기반 다중 포인트 검출 모델")
    print("- config_kornia.yml 기반 설정")
    print("- ID 끝자리로 train/test 분리")
    print("=" * 60)

    # 설정 로드 (config_kornia.yml 생성)
    config_path = Path(__file__).parent / 'config_kornia.yml'

    # config_kornia.yml이 없으면 config.yml을 복사하여 생성
    if not config_path.exists():
        original_config_path = Path(__file__).parent / 'config.yml'
        if original_config_path.exists():
            shutil.copy(str(original_config_path), str(config_path))
            print(f"config_kornia.yml 생성됨 (config.yml 복사)")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 경로 설정
    base_dir = Path(__file__).parent
    source_path = (base_dir / config['source']['source_folder']).resolve()

    # PyTorch 설정 (Kornia 모델용으로 수정)
    pytorch_config = config['pytorch_model']

    # 모델 파일명 설정
    save_file_name = 'multi_point_kornia'

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
    checkpoint_dir = Path(pytorch_config['checkpointing']['save_dir'])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(__file__).parent / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 검출기 및 모델 생성
    log_print("\n모델 초기화...")
    detector = MultiPointDetectorKornia(pytorch_config, device)
    model = detector.model

    # 모델 아키텍처 정보 출력
    log_print("\n=== 모델 아키텍처 (Kornia + FPD) ===")
    log_print(f"특징 추출: Kornia 기반")
    log_print(f"입력: 3x112x112 이미지")
    log_print(f"아키텍처: FPD (분류 기반 회귀)")
    log_print(f"위치 인코딩: 7x7 그리드")
    log_print(f"Cross-Attention 융합")
    log_print(f"좌표 클래스: X={model.x_classes if hasattr(model, 'x_classes') else 337}, Y={model.y_classes if hasattr(model, 'y_classes') else 225}")
    log_print(f"출력 차원: {pytorch_config['architecture']['output_dim']}")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"\n총 파라미터 수: {total_params:,}")
    log_print(f"학습 가능한 파라미터 수: {trainable_params:,}")

    # 데이터셋 생성
    print("\n데이터셋 생성 중...")
    train_dataset = MultiPointDataset(
        source_folder=str(source_path),
        labels_file='labels.txt',
        detector=detector,
        mode='train',
        config=pytorch_config,
        augment=pytorch_config['training']['augmentation']['enabled']
    )

    val_dataset = MultiPointDataset(
        source_folder=str(source_path),
        labels_file='labels.txt',
        detector=detector,
        mode='val',
        config=pytorch_config,
        augment=False
    )

    test_dataset = MultiPointDataset(
        source_folder=str(source_path),
        labels_file='labels.txt',
        detector=detector,
        mode='test',
        config=pytorch_config,
        augment=False
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
    learning_rate = pytorch_config['training']['learning_rate']

    # FPD 모델용 학습률 (config 값 그대로 사용)
    # learning_rate = learning_rate * 0.5  # 필요시 조정 가능

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=pytorch_config['training']['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=pytorch_config['training']['weight_decay']
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=pytorch_config['training']['weight_decay']
        )

    # 기존 체크포인트 확인 및 로드
    start_epoch = 0
    best_val_loss = float('inf')

    # 기존 최고 모델 체크
    best_checkpoint_path = checkpoint_dir / f"{save_file_name}_best.pth"
    if best_checkpoint_path.exists():
        log_print(f"\n기존 체크포인트 발견: {best_checkpoint_path}")
        log_print("체크포인트에서 학습을 재개합니다...")

        checkpoint = torch.load(str(best_checkpoint_path), map_location=device, weights_only=False)

        # 모델 상태 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'feature_mean' in checkpoint:
            detector.feature_mean = checkpoint['feature_mean']
            detector.feature_std = checkpoint['feature_std']

        # 옵티마이저 상태 로드
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 이전 학습 정보 복원
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        log_print(f"에폭 {start_epoch}부터 학습 재개")
        log_print(f"이전 최고 검증 손실: {best_val_loss:.6f}")
    else:
        log_print(f"\n체크포인트가 없습니다. 새로운 학습을 시작합니다...")
        log_print(f"체크포인트 경로: {best_checkpoint_path}")

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

    # 손실 함수 (FPD 모델은 자체 손실 함수 사용)
    criterion = None

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
    log_print(f"학습률: {learning_rate:.6f}")
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

        # 매 에폭마다 출력
        print(f"\nEpoch {epoch}/{pytorch_config['training']['epochs']} | "
              f"Time: {elapsed_str} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

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

        # 콘솔 출력 (progress_interval에 따라)
        progress_interval = pytorch_config['logging'].get('progress_interval', 100)
        detail_interval = pytorch_config['logging'].get('detail_interval', 500)

        if epoch == 1 or epoch % progress_interval == 0 or epoch == epochs:
            print(f"Epoch [{epoch:4d}/{epochs}] | "
                  f"시간: {epoch_time:5.1f}s | "
                  f"누적: {elapsed_str} | "
                  f"LR: {current_lr:.6f} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Err: {log_entry['avg_error']:.1f}px")

        # detail_interval마다 상세 오차 분석 출력
        if epoch % detail_interval == 0:
            print(f"\n{'='*60}")
            print(f"[상세 오차 분석] 에폭 {epoch}")
            print(f"{'='*60}")

            # 각 포인트별 오차 분석
            print("\n각 포인트별 오차 분석 (pixels):")
            for name in ['center', 'floor', 'front', 'side']:
                if name in all_val_errors and all_val_errors[name]['dist']:
                    dist_array = np.array(all_val_errors[name]['dist'])
                    x_array = np.array(all_val_errors[name]['x'])
                    y_array = np.array(all_val_errors[name]['y'])

                    print(f"\n  {name.upper():6s}:")
                    print(f"    X 오차: 평균={np.mean(x_array):.1f} 표준편차={np.std(x_array):.1f}")
                    print(f"    Y 오차: 평균={np.mean(y_array):.1f} 표준편차={np.std(y_array):.1f}")
                    print(f"    거리: 평균={np.mean(dist_array):.1f} 표준편차={np.std(dist_array):.1f}")

            # 전체 평균 거리 오차
            all_distances = []
            for errors in all_val_errors.values():
                all_distances.extend(errors['dist'])
            overall_mean_dist = np.mean(all_distances) if all_distances else 0
            print(f"\n전체 평균 거리 오차: {overall_mean_dist:.2f} pixels")

            # 상세 통계
            print("\n상세 통계 (거리 기준):")
            for name in ['center', 'floor', 'front', 'side']:
                if name in all_val_errors and all_val_errors[name]['dist']:
                    dist_array = np.array(all_val_errors[name]['dist'])
                    print(f"\n{name.upper()}:")
                    print(f"  평균: {np.mean(dist_array):.2f}")
                    print(f"  표준편차: {np.std(dist_array):.2f}")
                    print(f"  중앙값: {np.median(dist_array):.2f}")
                    print(f"  최소: {np.min(dist_array):.2f}")
                    print(f"  최대: {np.max(dist_array):.2f}")
                    print(f"  25% 분위수: {np.percentile(dist_array, 25):.2f}")
                    print(f"  75% 분위수: {np.percentile(dist_array, 75):.2f}")
            print(f"{'='*60}\n")

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
                'model_config': pytorch_config
            }
            torch.save(checkpoint_data, str(checkpoint_path))

            # Best 모델 저장 메시지 출력
            improvement = (previous_best - val_loss) if 'previous_best' in locals() else val_loss
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            log_print(f"\n{'='*60}")
            log_print(f"[BEST] 모델 저장 완료!")
            log_print(f"  • 에폭: {epoch}")
            log_print(f"  • 검증 손실: {val_loss:.6f} (개선: {improvement:.6f})")
            log_print(f"  • 평균 오차: {log_entry['avg_error']:.2f}px")
            log_print(f"  • 파일: {checkpoint_path.name} ({file_size_mb:.2f}MB)")
            log_print(f"  • 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # ONNX로도 저장
            onnx_path = save_model_as_onnx(model, pytorch_config, checkpoint_path, device)
            if onnx_path and onnx_path.exists():
                onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                log_print(f"[ONNX] 모델 저장: {onnx_path.name} ({onnx_size_mb:.2f}MB)")
            log_print(f"{'='*60}\n")

            previous_best = val_loss

            # best 로거에 기록
            if best_logger:
                best_logger.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
                best_logger.write(f"에폭 {epoch}: 손실={val_loss:.6f}, ")
                best_logger.write(f"평균 오차={log_entry['avg_error']:.2f}px\n")
        else:
            patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
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
                    'model_config': pytorch_config
                }
                torch.save(checkpoint_data, str(checkpoint_path))
                log_print(f"체크포인트 저장: {checkpoint_path}")

                # ONNX로도 저장
                onnx_path = save_model_as_onnx(model, pytorch_config, checkpoint_path, device)
                if onnx_path and onnx_path.exists():
                    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                    log_print(f"  - ONNX 저장: {onnx_path.name} ({onnx_size_mb:.2f}MB)")

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

    # 로거 닫기
    log_print("\n" + "=" * 60)
    log_print("학습 완료!")
    log_print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"전체 학습 시간: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}")
    log_print(f"최고 검증 손실: {best_val_loss:.6f} (Epoch {best_epoch})")
    log_print(f"최종 테스트 손실: {test_loss:.6f}")
    log_print(f"최종 평균 오차: {np.mean(list(test_errors.values())):.2f} pixels")
    # 최종 모델도 ONNX로 저장
    final_pth_path = checkpoint_dir / f"{save_file_name}_final.pth"
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_errors': test_errors,
        'model_config': pytorch_config
    }
    torch.save(final_checkpoint, str(final_pth_path))

    final_onnx_path = save_model_as_onnx(model, pytorch_config, final_pth_path, device)

    log_print(f"\n저장된 파일:")
    log_print(f"  - 최고 모델: {checkpoint_dir / f'{save_file_name}_best.pth'}")
    log_print(f"  - 최고 ONNX: {checkpoint_dir / f'{save_file_name}_best.onnx'}")
    log_print(f"  - 최종 모델: {final_pth_path}")
    if final_onnx_path and final_onnx_path.exists():
        log_print(f"  - 최종 ONNX: {final_onnx_path}")
    log_print(f"  - 모델 정보: {model_info_path}")
    log_print(f"  - 학습 로그: {log_csv_path if pytorch_config['logging']['save_csv'] else 'N/A'}")
    log_print(f"  - 실행 로그: {log_path}")
    log_print("=" * 60)

    logger.close()
    best_logger.close()


if __name__ == "__main__":
    main()