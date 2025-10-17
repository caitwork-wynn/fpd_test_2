# -*- coding: utf-8 -*-
"""
범용 다중 포인트 검출 모델 학습 스크립트
- config.yml의 learning_model->source에서 모델 동적 로드
- util 폴더의 유틸리티 활용
- 결과는 error_analysis->results_dir/save_file_name 폴더에 저장
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time
import json
import importlib.util
import warnings

# 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
warnings.filterwarnings("ignore", category=UserWarning, module="kornia")
warnings.filterwarnings("ignore", message=".*TracerWarning.*")
warnings.filterwarnings("ignore", message=".*trace might not generalize.*")
warnings.filterwarnings("ignore", message=".*constant folding.*")

# sys.path에 현재 디렉토리 추가
sys.path.append(str(Path(__file__).parent))

# 유틸리티 import
from util.dual_logger import DualLogger
from util.save_load_model import save_model, load_model
from util.error_analysis import display_error_analysis, save_error_analysis_json, save_training_log_csv


def train_epoch(model, dataloader, optimizer, device, config):
    """1 에폭 학습"""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        data = batch['data'].to(device)
        targets = batch['targets'].to(device)

        # Forward
        outputs = model(data)

        # 모델의 compute_loss 메서드 사용
        sigma = config['training']['loss'].get('soft_label_sigma', 1.0)
        loss = model.compute_loss(outputs, targets, sigma=sigma)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clip = config['training'].get('gradient_clip', 0)
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """검증/테스트 평가"""
    model.eval()
    total_loss = 0

    # 첫 번째 배치에서 포인트 정보 자동 감지
    sample_batch = next(iter(dataloader))
    sample_output = model(sample_batch['data'].to(device))

    # 포인트 이름 자동 감지
    if 'point_names' in sample_output:
        point_names = sample_output['point_names']
        # 리스트가 아니면 리스트로 변환
        if not isinstance(point_names, list):
            point_names = list(point_names) if hasattr(point_names, '__iter__') and not isinstance(point_names, str) else [point_names]
    elif 'point_names' in sample_batch:
        # DataLoader의 default_collate가 배치의 point_names를 zip하기 때문에
        # [('center', 'center', ...), ('floor', 'floor', ...), ...] 형태가 됨
        # 각 튜플의 첫 번째 요소만 가져와서 고유한 리스트를 만듦
        point_names_raw = sample_batch['point_names']
        if isinstance(point_names_raw, (list, tuple)) and len(point_names_raw) > 0:
            if isinstance(point_names_raw[0], (tuple, list)):
                # zip된 형태: [('center', 'center'), ('floor', 'floor'), ...]
                # 각 튜플의 첫 번째 요소만 가져옴
                point_names = [item[0] if isinstance(item, (tuple, list)) else item for item in point_names_raw]
            else:
                # 이미 단순 리스트: ['center', 'floor', 'front', 'side']
                point_names = list(point_names_raw)
        else:
            point_names = list(point_names_raw) if hasattr(point_names_raw, '__iter__') and not isinstance(point_names_raw, str) else [point_names_raw]
    else:
        # 좌표 개수로부터 추론
        num_coords = sample_output['coordinates'].shape[1]
        if num_coords == 2:
            point_names = ['floor']
        elif num_coords == 8:
            point_names = ['center', 'floor', 'front', 'side']
        else:
            point_names = [f'point_{i//2}' for i in range(0, num_coords, 2)]

    # point_names가 리스트인지 확인 및 기본값 설정
    if not isinstance(point_names, list) or len(point_names) == 0:
        point_names = ['center', 'floor', 'front', 'side']  # 기본값

    all_errors = {}
    for name in point_names:
        all_errors[name] = {'x': [], 'y': [], 'dist': []}

    # 데이터셋 객체 가져오기 (좌표 범위 정보 필요)
    dataset = dataloader.dataset

    with torch.no_grad():
        for batch in dataloader:
            data = batch['data'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(data)

            # 손실 계산
            loss = model.compute_loss(outputs, targets, sigma=1.0)

            # 연속 좌표 추출
            outputs_coords = outputs['coordinates']  # [B, N] N은 좌표 개수

            total_loss += loss.item()

            # 픽셀 단위 오차 계산
            outputs_np = outputs_coords.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i in range(len(outputs_np)):
                # 정규화된 좌표를 원본 픽셀 좌표로 변환
                pred_coords = []
                true_coords = []

                for j in range(0, outputs_coords.shape[1], 2):
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

                # 각 포인트별 오차 계산 (동적 처리)
                point_errors = {}
                for idx, name in enumerate(point_names):
                    x_idx = idx * 2
                    y_idx = idx * 2 + 1
                    if x_idx < len(pred_coords) and y_idx < len(pred_coords):
                        point_errors[name] = {
                            'x': pred_coords[x_idx] - true_coords[x_idx],
                            'y': pred_coords[y_idx] - true_coords[y_idx]
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

    # 모든 포인트가 없는 경우를 대비해 기본값 설정
    for name in point_names:
        if name not in avg_errors:
            avg_errors[name] = 0

    return avg_loss, avg_errors, all_errors




def main():
    """메인 함수"""
    print("=" * 60)
    print("범용 다중 포인트 검출 모델 학습")
    print("- config.yml 기반 동적 모델 로딩")
    print("- util 폴더 유틸리티 활용")
    print("=" * 60)

    # 설정 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 동적 모델 import
    model_source = config['learning_model']['source']
    model_path = (Path(__file__).parent / model_source).resolve()

    print(f"\n모델 로딩: {model_path}")

    # 모듈 동적 import
    spec = importlib.util.spec_from_file_location("model_module", str(model_path))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 표준 클래스 import
    PointDetectorDataSet = model_module.PointDetectorDataSet
    PointDetector = model_module.PointDetector
    # Autoencoder 버전도 import (존재하는 경우)
    PointDetectorDataSetWithAutoencoder = getattr(model_module, 'PointDetectorDataSetWithAutoencoder', None)

    print("모델 클래스 로드 완료")

    # 모델 설정 가져오기 (get_model_config 함수가 있는 경우)
    if hasattr(model_module, 'get_model_config'):
        model_config = model_module.get_model_config()
        save_file_name = model_config['save_file_name']
        target_points = model_config['target_points']
        use_fpd_architecture = model_config['use_fpd_architecture']
        features_config = model_config['features']
        print(f"모델 설정 로드: {save_file_name}")
        print(f"타겟 포인트: {target_points}")
        print(f"FPD 아키텍처: {use_fpd_architecture}")
    else:
        # get_model_config가 없는 경우 config.yml에서 읽기 (하위 호환성)
        save_file_name = config.get('learning_model', {}).get('checkpointing', {}).get('save_file_name', 'model')
        target_points = config.get('learning_model', {}).get('target_points', None)
        use_fpd_architecture = config.get('learning_model', {}).get('architecture', {}).get('use_fpd_architecture', False)
        features_config = config.get('learning_model', {}).get('architecture', {}).get('features', {
            'image_size': [112, 112],
            'grid_size': 7
        })
        print(f"모델 설정 로드 (config.yml): {save_file_name}")

    # 경로 설정
    base_dir = Path(__file__).parent
    data_path = (base_dir / config['data']['source_folder']).resolve()

    # 결과 디렉토리 생성
    results_base_dir = Path(config['training']['error_analysis']['results_dir'])
    if not results_base_dir.is_absolute():
        results_base_dir = base_dir / results_base_dir
    result_dir = results_base_dir / save_file_name
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 디렉토리: {result_dir}")

    # 디바이스 설정
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        print(f"\nGPU 사용: {torch.cuda.get_device_name(device)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        print("\nCPU 사용")

    # 로그 디렉토리 생성
    log_dir = Path(config['logging']['log_dir'])
    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
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
    checkpoint_dir = Path(config['learning_model']['checkpointing']['save_dir'])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = base_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 검출기 및 모델 생성
    log_print("\n모델 초기화...")
    # learning_model 섹션과 features 섹션을 합쳐서 전달
    detector_config = config['learning_model'].copy()
    detector_config['features'] = features_config  # 모듈에서 가져온 설정 사용
    detector_config['training'] = config['training']
    detector_config['target_points'] = target_points  # 모듈에서 가져온 타겟 포인트
    detector_config['use_fpd_architecture'] = use_fpd_architecture  # 모듈에서 가져온 아키텍처 설정
    detector = PointDetector(detector_config, device)
    model = detector.model

    # 모델 아키텍처 정보 출력
    log_print("\n=== 모델 아키텍처 ===")
    log_print(f"모델 소스: {model_source}")
    log_print(f"입력: 3x{features_config['image_size'][0]}x{features_config['image_size'][1]} 이미지")
    log_print(f"그리드 크기: {features_config['grid_size']}x{features_config['grid_size']}")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"\n총 파라미터 수: {total_params:,}")
    log_print(f"학습 가능한 파라미터 수: {trainable_params:,}")

    # 데이터셋 생성
    log_print("\n데이터셋 생성 중...")

    # config 형식 맞추기 (DataSet 클래스가 필요로 하는 구조 생성)
    dataset_config = config.copy()
    if 'data_split' not in dataset_config:
        dataset_config['data_split'] = {
            'test_id_suffix': config['data']['test_id_suffix'],
            'validation_ratio': config['data']['validation_ratio'],
            'random_seed': config['data'].get('random_seed', 42)
        }

    # learning_model 섹션 추가/업데이트 (DataSet이 image_size에 접근할 수 있도록)
    if 'learning_model' not in dataset_config:
        dataset_config['learning_model'] = {}
    if 'architecture' not in dataset_config['learning_model']:
        dataset_config['learning_model']['architecture'] = {}
    dataset_config['learning_model']['architecture']['features'] = features_config

    # max_train_images 설정 읽기
    max_train_images = config['data'].get('max_train_images', 0)

    # extract_features 옵션 읽기
    extract_features = config['training'].get('extract_features', False)
    if extract_features:
        log_print("특징 사전 추출 모드 활성화 (학습 속도 향상)")

    # 타겟 포인트 정보 출력 (이미 모듈에서 로드됨)
    if target_points:
        log_print(f"타겟 포인트: {', '.join(target_points)}")

    # Autoencoder 사용 여부 확인 (모듈 설정에서 가져옴)
    use_autoencoder = features_config.get('use_autoencoder', False)

    if use_autoencoder and PointDetectorDataSetWithAutoencoder is not None:
        log_print("Autoencoder 기반 특성 추출 사용")
        encoder_path = features_config.get('encoder_path', '../model/autoencoder_16x16_best.pth')

        # DataSetWithAutoencoder 사용
        train_dataset = PointDetectorDataSetWithAutoencoder(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='train',
            config=dataset_config,
            augment=config['training']['augmentation']['enabled'],
            extract_features=True,  # Autoencoder는 항상 사전 추출
            encoder_path=encoder_path,
            target_points=target_points  # 타겟 포인트 전달
        )
    else:
        # 기존 DataSet 사용
        train_dataset = PointDetectorDataSet(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='train',
            config=dataset_config,
            augment=config['training']['augmentation']['enabled'],
            extract_features=extract_features,
            target_points=target_points  # 타겟 포인트 전달
        )

    # max_train_images 설정 확인 (DataSet에서 이미 처리됨)
    if max_train_images > 0:
        log_print(f"max_train_images 설정: {max_train_images}개")

    if use_autoencoder and PointDetectorDataSetWithAutoencoder is not None:
        val_dataset = PointDetectorDataSetWithAutoencoder(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='val',
            config=dataset_config,
            augment=False,
            extract_features=True,
            encoder_path=encoder_path,
            target_points=target_points  # 타겟 포인트 전달
        )

        test_dataset = PointDetectorDataSetWithAutoencoder(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='test',
            config=dataset_config,
            augment=False,
            extract_features=True,
            encoder_path=encoder_path,
            target_points=target_points  # 타겟 포인트 전달
        )
    else:
        val_dataset = PointDetectorDataSet(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='val',
            config=dataset_config,
            augment=False,
            extract_features=extract_features,
            target_points=target_points  # 타겟 포인트 전달
        )

        test_dataset = PointDetectorDataSet(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='test',
            config=dataset_config,
            augment=False,
            extract_features=extract_features,
            target_points=target_points  # 타겟 포인트 전달
        )

    # DataLoader 생성
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 옵티마이저 설정
    optimizer_name = config['training']['optimizer'].lower()
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    # 기존 체크포인트 확인 및 로드
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_time_loaded = None  # 로드된 시간 정보

    try:
        # 먼저 epoch 파일 로드 시도 (최신 epoch 파일 우선)
        loaded_epoch = load_model(
            model=model,
            optimizer=optimizer,
            save_dir=str(checkpoint_dir),
            model_name=save_file_name,
            load_best=False,  # epoch 파일 로드
            device=device
        )

        if loaded_epoch > 0:
            log_print(f"Epoch {loaded_epoch} 모델에서 학습 재개")
            start_epoch = loaded_epoch

            # epoch 파일에서 best_val_loss 정보 복원 시도
            epoch_checkpoint_path = checkpoint_dir / f"{save_file_name}_epoch{loaded_epoch}.pth"
            if epoch_checkpoint_path.exists():
                checkpoint = torch.load(str(epoch_checkpoint_path), map_location=device, weights_only=False)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_model_time_loaded = checkpoint.get('best_model_time', None)

                # feature_mean, feature_std 복원 (있는 경우)
                if 'feature_mean' in checkpoint:
                    detector.feature_mean = checkpoint['feature_mean']
                    detector.feature_std = checkpoint['feature_std']

        log_print(f"에폭 {start_epoch}부터 학습 재개")
        log_print(f"이전 최고 검증 손실: {best_val_loss:.6f}")

    except FileNotFoundError:
        # epoch 파일이 없으면 best 모델 로드 시도
        try:
            loaded_epoch = load_model(
                model=model,
                optimizer=optimizer,
                save_dir=str(checkpoint_dir),
                model_name=save_file_name,
                load_best=True,  # best 모델 로드
                device=device
            )

            if loaded_epoch == -1:  # best 모델
                log_print("Best 모델에서 학습 재개 (epoch 파일 없음)")
                # best 모델에서 재개할 때는 실제 epoch 정보를 체크포인트에서 읽어야 함
                best_checkpoint_path = checkpoint_dir / f"{save_file_name}_best.pth"
                checkpoint = torch.load(str(best_checkpoint_path), map_location=device, weights_only=False)
                start_epoch = checkpoint.get('epoch', 0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_model_time_loaded = checkpoint.get('best_model_time', None)

                # feature_mean, feature_std 복원 (있는 경우)
                if 'feature_mean' in checkpoint:
                    detector.feature_mean = checkpoint['feature_mean']
                    detector.feature_std = checkpoint['feature_std']

            log_print(f"에폭 {start_epoch}부터 학습 재개")
            log_print(f"이전 최고 검증 손실: {best_val_loss:.6f}")

        except FileNotFoundError:
            # epoch 파일도 없고 best 파일도 없으면 새로운 학습
            log_print("새로운 학습 시작")
            start_epoch = 0

    # 스케줄러 설정
    scheduler_config = config['training']['scheduler']
    min_lr = scheduler_config.get('min_lr', 0)

    # 스케줄러 비활성화 체크 (enabled 플래그 또는 type: 'none')
    if not scheduler_config.get('enabled', True) or scheduler_config['type'] == 'none':
        scheduler = None
        log_print("학습률 스케줄러 비활성화 - 고정 학습률 사용")
    elif scheduler_config['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_config['patience'],
            factor=scheduler_config['factor'],
            min_lr=min_lr
        )
        log_print(f"ReduceLROnPlateau 스케줄러 설정: patience={scheduler_config['patience']}, factor={scheduler_config['factor']}, min_lr={min_lr}")
    elif scheduler_config['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
        log_print(f"StepLR 스케줄러 설정: step_size={scheduler_config['step_size']}, gamma={scheduler_config['gamma']}, min_lr={min_lr}")
    elif scheduler_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=min_lr
        )
        log_print(f"CosineAnnealingLR 스케줄러 설정: T_max={config['training']['epochs']}, eta_min={min_lr}")
    else:
        scheduler = None
        log_print(f"알 수 없는 스케줄러 타입: {scheduler_config['type']} - 스케줄러 비활성화")

    # 학습 설정
    epochs = config['training']['epochs']
    early_stopping_patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']
    save_frequency = config['learning_model']['checkpointing']['save_frequency']

    # 오차 분석 설정
    error_analysis_config = config['training']['error_analysis']
    error_analysis_enabled = error_analysis_config['enabled']
    error_analysis_interval = error_analysis_config['interval']
    save_raw_data = error_analysis_config['save_raw_data']

    # 학습 시작
    log_print(f"\n===== 학습 시작 =====")
    log_print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"총 에폭: {epochs}")
    log_print(f"Train: {len(train_dataset)}개, Val: {len(val_dataset)}개, Test: {len(test_dataset)}개")
    log_print(f"배치 크기: {batch_size}")
    log_print(f"학습률: {learning_rate:.6f}")
    log_print(f"옵티마이저: {optimizer_name.upper()}")
    log_print("=" * 50)

    # 모델 정보 저장용 딕셔너리
    model_info = {
        'model_name': save_file_name,
        'model_source': model_source,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'config': config,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    best_epoch = 0
    patience_counter = 0
    training_log = []
    training_start_time = time.time()

    # 시간 기반 LR 감소 설정
    time_based_config = config['training']['scheduler'].get('time_based', {})
    time_based_enabled = time_based_config.get('enabled', False)
    time_patience_hours = time_based_config.get('patience_hours', 1.0)
    time_reduce_factor = time_based_config.get('factor', 0.5)

    # best_model_time 초기화 (체크포인트에서 로드되었으면 사용, 없으면 현재 시간)
    if best_model_time_loaded is not None:
        best_model_time = best_model_time_loaded
        log_print(f"Best 모델 시간 복원: {datetime.fromtimestamp(best_model_time).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        best_model_time = time.time()  # Best 모델 갱신 시각 초기화

    # 학습 루프
    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start_time = time.time()

        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, device, config)

        # 검증
        val_loss, val_errors, all_val_errors = validate(model, val_loader, device)

        # 스케줄러 업데이트 (스케줄러가 활성화되어 있을 때만)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # StepLR에서 min_lr 적용 (StepLR은 기본적으로 min_lr을 지원하지 않음)
            if isinstance(scheduler, optim.lr_scheduler.StepLR):
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < min_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = min_lr

        # 시간 기반 LR 감소 체크 (스케줄러 업데이트 이후)
        if time_based_enabled:
            current_time = time.time()
            hours_since_best = (current_time - best_model_time) / 3600

            if hours_since_best >= time_patience_hours:
                current_lr_before = optimizer.param_groups[0]['lr']
                new_lr = max(current_lr_before * time_reduce_factor, min_lr)

                if new_lr < current_lr_before:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    log_print(f"[시간 기반 LR 감소] Best 갱신 후 {hours_since_best:.1f}시간 경과 - LR: {current_lr_before:.2e} → {new_lr:.2e}")
                    best_model_time = current_time  # 타이머 리셋

        # 에폭 시간 계산
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        elapsed_h = int(total_elapsed // 3600)
        elapsed_m = int((total_elapsed % 3600) // 60)
        elapsed_s = int(total_elapsed % 60)
        elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"

        # 현재 학습률
        current_lr = optimizer.param_groups[0]['lr']

        # 평균 오차 (0이 아닌 값들만으로 계산)
        non_zero_errors = [v for v in val_errors.values() if v > 0]
        avg_error = np.mean(non_zero_errors) if non_zero_errors else 0

        # 로깅 (동적 포인트 처리)
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'avg_error': avg_error
        }
        # 각 포인트별 오차 추가
        for name in ['center', 'floor', 'front', 'side']:
            if name in val_errors:
                log_entry[f'{name}_error'] = val_errors[name]
            else:
                log_entry[f'{name}_error'] = 0
        training_log.append(log_entry)

        # 콘솔 출력 (error_analysis.interval 사용)
        if epoch == 1 or epoch % error_analysis_interval == 0 or epoch == epochs:
            print(f"\nEpoch [{epoch:4d}/{epochs}] | "
                  f"Time: {elapsed_str} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")
            error_str = f"Avg Error: {avg_error:.1f}px"
            for name in ['center', 'floor', 'front', 'side']:
                if name in val_errors:
                    error_str += f" | {name.capitalize()}: {val_errors[name]:.1f}px"
            print(error_str)

        # 오차 분석 출력 및 저장 (error_analysis_interval과 통합된 상세 출력)
        if error_analysis_enabled and epoch % error_analysis_interval == 0:
            log_print(f"\n{'='*60}")
            log_print(f"[Epoch {epoch}] 검증 데이터 오차 분석")
            log_print(f"{'='*60}")

            # 추가 정보 출력
            epochs_since_best = epoch - best_epoch
            hours_since_best_display = (time.time() - best_model_time) / 3600
            log_print(f"현재 에폭: {epoch}")
            log_print(f"마지막 Best 에폭: {best_epoch}")
            log_print(f"Best 에폭으로부터 경과: {epochs_since_best} 에폭 ({hours_since_best_display:.1f}시간)")
            log_print(f"현재 학습률: {current_lr:.2e}")
            log_print(f"현재 검증 손실: {val_loss:.6f}")
            log_print(f"Best 검증 손실: {best_val_loss:.6f}")
            if time_based_enabled:
                log_print(f"시간 기반 LR 감소: {time_patience_hours}시간마다, factor={time_reduce_factor}")
            log_print(f"-"*60)

            # 오차 분석 출력
            output_lines = display_error_analysis(
                all_val_errors,
                epoch=epoch
            )
            for line in output_lines:
                log_print(line)

            log_print(f"{'='*60}\n")

            # JSON 저장
            if save_raw_data:
                json_path = result_dir / f"error_epoch_{epoch}.json"
                save_error_analysis_json(
                    errors=all_val_errors,
                    epoch=epoch,
                    save_path=str(json_path),
                    additional_info={
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': current_lr,
                        'avg_error_pixels': avg_error,
                        'best_epoch': best_epoch,
                        'best_val_loss': best_val_loss,
                        'epochs_since_best': epoch - best_epoch
                    }
                )
                log_print(f"오차 분석 저장: {json_path}")

        # 체크포인트 저장
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_time = time.time()  # Best 모델 갱신 시각 업데이트

            # 추가 정보 포함한 체크포인트 데이터
            checkpoint_data = {
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_errors': val_errors,
                'model_config': config,
                'best_model_time': best_model_time  # 시간 정보 저장
            }

            # feature_mean, feature_std 저장 (있는 경우)
            if hasattr(detector, 'feature_mean'):
                checkpoint_data['feature_mean'] = detector.feature_mean
                checkpoint_data['feature_std'] = detector.feature_std

            # Best 모델 저장 (ONNX 자동 변환)
            checkpoint_path = save_model(
                model=model,
                optimizer=optimizer,
                save_dir=str(checkpoint_dir),
                model_name=save_file_name,
                is_best=True,
                device=device,
                log_func=log_print
            )

            # 추가 정보를 체크포인트에 저장
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            checkpoint.update(checkpoint_data)
            torch.save(checkpoint, str(checkpoint_path))

            # Best 모델 정보를 결과 디렉토리에도 기록 (상세 오차 정보 포함)
            best_info_path = result_dir / f"best_epoch_{epoch}.json"

            # 상세 오차 정보 계산
            detailed_errors = {}
            all_distances = []

            for point_name, errors in all_val_errors.items():
                # point_name이 튜플일 경우 문자열로 변환
                if isinstance(point_name, tuple):
                    point_name_str = str(point_name)
                else:
                    point_name_str = str(point_name)

                if errors.get('x') and errors.get('y') and errors.get('dist'):
                    detailed_errors[point_name_str] = {
                        'x': {
                            'mean': float(np.mean(errors['x'])),
                            'std': float(np.std(errors['x']))
                        },
                        'y': {
                            'mean': float(np.mean(errors['y'])),
                            'std': float(np.std(errors['y']))
                        },
                        'dist': {
                            'mean': float(np.mean(errors['dist'])),
                            'std': float(np.std(errors['dist']))
                        }
                    }
                    all_distances.extend(errors['dist'])

            # 전체 오차 계산
            overall_error = None
            if all_distances:
                overall_error = {
                    'mean': float(np.mean(all_distances)),
                    'std': float(np.std(all_distances))
                }

            with open(best_info_path, 'w', encoding='utf-8') as f:
                # val_errors 키를 문자열로 변환
                val_errors_simple = {}
                for k, v in val_errors.items():
                    key_str = str(k) if isinstance(k, tuple) else k
                    val_errors_simple[key_str] = float(v)

                json.dump({
                    'epoch': epoch,
                    'val_loss': float(val_loss),
                    'val_errors': detailed_errors,
                    'val_errors_simple': val_errors_simple,  # 기존 형식 호환성
                    'overall_error': overall_error,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)

            # Best 모델 저장 시 오차 분석 표시 (error_analysis와 동일 형식)
            log_print(f"\n{'='*60}")
            log_print(f"[BEST MODEL SAVED] Epoch {epoch}")
            log_print(f"{'='*60}")

            # display_error_analysis 함수 활용하여 동일한 형식으로 출력
            output_lines = display_error_analysis(
                all_val_errors,
                epoch=epoch,
                title=f"=== Epoch {epoch} 오차 분석 (BEST) ==="
            )
            for line in output_lines:
                log_print(line)

            # 추가 정보
            log_print(f"Validation Loss: {val_loss:.6f}")
            if 'previous_best_loss' in locals() and previous_best_loss != float('inf'):
                improvement = previous_best_loss - val_loss
                log_print(f"Improvement: {improvement:.6f} ({(improvement/previous_best_loss)*100:.2f}%)")

            log_print(f"{'='*60}\n")

            # 이전 best loss 저장
            previous_best_loss = val_loss

            # best 로거에 기록
            if best_logger:
                best_logger.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
                best_logger.write(f"에폭 {epoch}: 손실={val_loss:.6f}, ")
                best_logger.write(f"평균 오차={avg_error:.2f}px\n")
        else:
            patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                log_print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # 주기적 저장
        if epoch % save_frequency == 0:
            if not config['learning_model']['checkpointing']['save_best_only']:
                checkpoint_path = save_model(
                    model=model,
                    optimizer=optimizer,
                    save_dir=str(checkpoint_dir),
                    model_name=save_file_name,
                    is_best=False,
                    epoch=epoch,
                    log_func=log_print
                )
                log_print(f"체크포인트 저장: {checkpoint_path}")

    # 학습 완료 시간 계산
    total_training_time = time.time() - training_start_time
    total_hours = int(total_training_time // 3600)
    total_minutes = int((total_training_time % 3600) // 60)
    total_seconds = int(total_training_time % 60)

    # 학습 로그 CSV 저장 (항상 저장)
    log_csv_path = result_dir / "training_log.csv"
    save_training_log_csv(training_log, str(log_csv_path))

    # 테스트 데이터 최종 평가
    log_print("\n=== 테스트 데이터 최종 평가 ===")
    test_loss, test_errors, all_test_errors = validate(model, test_loader, device)

    # 최종 오차 분석 출력
    output_lines = display_error_analysis(
        all_test_errors,
        epoch='final',
        title="=== 최종 테스트 결과 오차 분석 ==="
    )
    for line in output_lines:
        log_print(line)

    log_print(f"\n최종 테스트 손실: {test_loss:.6f}")
    non_zero_test_errors = [v for v in test_errors.values() if v > 0]
    final_avg_error = np.mean(non_zero_test_errors) if non_zero_test_errors else 0
    log_print(f"최종 평균 오차: {final_avg_error:.2f} pixels")

    # 최종 결과 JSON 저장
    final_json_path = result_dir / "error_final.json"
    save_error_analysis_json(
        errors=all_test_errors,
        epoch='final',
        save_path=str(final_json_path),
        additional_info={
            'test_loss': float(test_loss),
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss),
            'total_epochs': epoch,
            'avg_error_pixels': float(final_avg_error),
            'model_path': str(checkpoint_dir / f"{save_file_name}_best.pth")
        }
    )

    # 최종 결과 텍스트 파일 저장 (result/result_모델명.txt)
    final_txt_path = results_base_dir / f"result_{save_file_name}.txt"
    with open(final_txt_path, 'w', encoding='utf-8') as f:
        f.write("=== 테스트 데이터 최종 평가 ===\n\n")
        for line in output_lines:
            f.write(line + "\n")
        f.write(f"\n최종 테스트 손실: {test_loss:.6f}\n")
        f.write(f"최종 평균 오차: {final_avg_error:.2f} pixels\n")
    log_print(f"최종 결과 텍스트 저장: {final_txt_path}")

    # 모델 정보 JSON 저장
    model_info['best_epoch'] = best_epoch
    model_info['best_val_loss'] = float(best_val_loss)
    model_info['final_test_loss'] = float(test_loss)
    model_info['total_training_time'] = f"{total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}"
    model_info['final_epoch'] = epoch

    model_info_path = result_dir / "model_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)

    # 로거 닫기
    log_print("\n" + "=" * 60)
    log_print("학습 완료!")
    log_print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"전체 학습 시간: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}")
    log_print(f"최고 검증 손실: {best_val_loss:.6f} (Epoch {best_epoch})")
    log_print(f"최종 테스트 손실: {test_loss:.6f}")
    log_print(f"최종 평균 오차: {final_avg_error:.2f} pixels")
    log_print(f"\n저장된 파일:")
    log_print(f"  - 최고 모델: {checkpoint_dir / f'{save_file_name}_best.pth'}")
    log_print(f"  - 최고 ONNX: {checkpoint_dir / f'{save_file_name}_best.onnx'}")
    log_print(f"  - 결과 디렉토리: {result_dir}")
    log_print(f"  - 실행 로그: {log_path}")
    log_print("=" * 60)

    logger.close()
    best_logger.close()


if __name__ == "__main__":
    main()