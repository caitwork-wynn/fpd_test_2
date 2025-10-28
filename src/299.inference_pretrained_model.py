# -*- coding: utf-8 -*-
"""
범용 다중 포인트 검출 모델 테스트 스크립트
- 학습된 모델을 로드하여 테스트 데이터셋 평가
- config.yml의 learning_model->source에서 모델 동적 로드
- 테스트 결과를 화면에 출력
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import importlib.util
import warnings
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time
import json

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
from util.error_analysis import display_error_analysis


def validate(model, dataloader, device, data_path=None):
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

    # 상세 결과 저장용 리스트
    detailed_results = []

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

            # 배치 내 파일명 가져오기
            filenames = batch.get('filename', [f'sample_{i}' for i in range(len(outputs_np))])

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
                labeled_points = {}
                predict_points = {}
                error_points = {}

                for idx, name in enumerate(point_names):
                    x_idx = idx * 2
                    y_idx = idx * 2 + 1
                    if x_idx < len(pred_coords) and y_idx < len(pred_coords):
                        point_errors[name] = {
                            'x': pred_coords[x_idx] - true_coords[x_idx],
                            'y': pred_coords[y_idx] - true_coords[y_idx]
                        }
                        labeled_points[name] = (float(true_coords[x_idx]), float(true_coords[y_idx]))
                        predict_points[name] = (float(pred_coords[x_idx]), float(pred_coords[y_idx]))
                        error_points[name] = (float(point_errors[name]['x']), float(point_errors[name]['y']))

                for key, err in point_errors.items():
                    all_errors[key]['x'].append(abs(err['x']))
                    all_errors[key]['y'].append(abs(err['y']))
                    all_errors[key]['dist'].append(np.sqrt(err['x']**2 + err['y']**2))

                # 상세 결과 저장
                current_filename = filenames[i] if isinstance(filenames, list) else filenames
                image_path = None
                if data_path:
                    # 이미지 파일 경로 생성
                    image_path = Path(data_path) / current_filename

                detailed_results.append({
                    'filename': current_filename,
                    'image_path': str(image_path) if image_path else None,
                    'labeled': labeled_points,
                    'predict': predict_points,
                    'errors': error_points
                })

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

    return avg_loss, avg_errors, all_errors, detailed_results


def create_visualization_grid(detailed_results, output_path, image_size, num_samples=40, grid_size=(8, 5)):
    """
    테스트 결과를 그리드 형태로 시각화하여 저장

    Args:
        detailed_results: 테스트 결과 리스트
        output_path: 저장할 파일 경로
        image_size: 모델 입력 이미지 사이즈 [width, height]
        num_samples: 샘플 개수 (기본값: 40)
        grid_size: 그리드 크기 (rows, cols) (기본값: (8, 5))
    """
    # 랜덤하게 샘플 선택
    if len(detailed_results) > num_samples:
        selected_results = random.sample(detailed_results, num_samples)
    else:
        selected_results = detailed_results[:num_samples]

    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.tight_layout(pad=1.0)

    for idx, result in enumerate(selected_results):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # 이미지 로드
        image_path = result.get('image_path')
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # 이미지 로드 실패 시 빈 이미지
                img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        else:
            # 이미지 경로가 없으면 빈 이미지
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255

        # 포인트 그리기 (이미지에 직접)
        img_copy = img.copy()

        # 원본 이미지 크기 가져오기
        orig_h, orig_w = img.shape[:2]

        # 모델 입력 좌표를 원본 이미지 크기로 스케일링
        # 모델 설정에서 읽어온 이미지 사이즈를 기준으로 스케일 계산
        model_size = float(image_size[0])  # 모델 설정에서 읽어온 이미지 사이즈 사용
        scale_x = orig_w / model_size
        scale_y = orig_h / model_size

        # 포인트별 색상 정의 (RGB)
        point_colors = {
            'center': (0, 255, 0),      # 초록색
            'floor': (255, 255, 0),     # 노란색
            'front': (255, 0, 255),     # 자홍색
            'side': (0, 255, 255)       # 청록색
        }

        # 파란색으로 실제 좌표 그리기 (이중 원 구조)
        labeled = result['labeled']
        for point_name, (x, y) in labeled.items():
            # 모델 입력 스케일의 좌표를 원본 크기로 변환
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            # 외각: 포인트별 색상 (큰 원, 테두리만)
            point_color = point_colors.get(point_name, (128, 128, 128))  # 기본값: 회색
            cv2.circle(img_copy, (scaled_x, scaled_y), 8, point_color, 2)
            # 내부: 파란색 (작은 원, 채움)
            cv2.circle(img_copy, (scaled_x, scaled_y), 5, (0, 0, 255), -1)

        # 빨간색으로 예측 좌표 그리기 (이중 원 구조)
        predict = result['predict']
        for point_name, (x, y) in predict.items():
            # 모델 입력 스케일의 좌표를 원본 크기로 변환
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            # 외각: 포인트별 색상 (큰 원, 테두리만)
            point_color = point_colors.get(point_name, (128, 128, 128))  # 기본값: 회색
            cv2.circle(img_copy, (scaled_x, scaled_y), 8, point_color, 2)
            # 내부: 빨간색 (작은 원, 채움)
            cv2.circle(img_copy, (scaled_x, scaled_y), 5, (255, 0, 0), -1)

        # 이미지 표시
        ax.imshow(img_copy)
        ax.axis('off')

        # 오차 텍스트 생성 및 표시 (오른쪽 위)
        errors = result['errors']
        error_text_lines = []
        for point_name, (x_err, y_err) in errors.items():
            error_text_lines.append(f"{point_name}: {x_err:.1f}, {y_err:.1f}")

        error_text = '\n'.join(error_text_lines)

        # 텍스트 배경 박스와 함께 표시
        ax.text(0.98, 0.02, error_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                family='monospace')

    # 빈 셀 숨기기
    for idx in range(len(selected_results), rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    # 저장
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"시각화 이미지 저장 완료: {output_path}")



def main():
    """메인 함수"""
    # Argument parser 설정
    parser = argparse.ArgumentParser(description='포인트 검출 모델 테스트')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='테스트할 모델 가중치 파일 경로 (기본값: config.yml의 모델 설정 기반 자동 결정)'
    )
    parser.add_argument(
        '--model_source',
        type=str,
        default=None,
        help='모델 정의 소스 파일 경로 (기본값: config.yml의 learning_model->source 사용)'
    )
    args = parser.parse_args()

    # 설정 로드 (모델 경로 결정을 위해 먼저 로드)
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 동적 모델 import
    # argument로 model_source가 지정되면 해당 파일 사용, 아니면 config.yml 사용
    if args.model_source:
        model_source = args.model_source
    else:
        model_source = config['learning_model']['source']

    # 모델 정의 파일을 먼저 import하여 save_file_name 가져오기
    base_dir = Path(__file__).parent
    model_path = (base_dir / model_source).resolve()

    # 모듈 동적 import
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", str(model_path))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 모델 설정 가져오기 (save_file_name 획득)
    if hasattr(model_module, 'get_model_config'):
        model_config = model_module.get_model_config()
        save_file_name = model_config['save_file_name']
    else:
        # get_model_config가 없는 경우 config.yml에서 읽기
        save_file_name = config.get('learning_model', {}).get('checkpointing', {}).get('save_file_name', 'model')

    # 모델 경로 결정: argument로 지정되지 않은 경우 자동 생성
    if args.model_path is None:
        args.model_path = f'../model/{save_file_name}_best.pth'

    print("=" * 60)
    print("범용 다중 포인트 검출 모델 테스트")
    print("- 동적 모델 로딩")
    print(f"- 모델 소스: {model_source}")
    print(f"- 모델 가중치 파일: {args.model_path}")
    print("=" * 60)

    print(f"\n모델 정의 로딩: {model_path}")
    print("모델 클래스 로드 완료")

    # 표준 클래스 import (이미 위에서 모듈 import 완료)
    PointDetectorDataSet = model_module.PointDetectorDataSet
    PointDetector = model_module.PointDetector
    # Autoencoder 버전도 import (존재하는 경우)
    PointDetectorDataSetWithAutoencoder = getattr(model_module, 'PointDetectorDataSetWithAutoencoder', None)

    # 모델 설정 가져오기 (이미 위에서 로드 완료)
    target_points = model_config.get('target_points', None) if hasattr(model_module, 'get_model_config') else config.get('learning_model', {}).get('target_points', None)
    use_fpd_architecture = model_config.get('use_fpd_architecture', False) if hasattr(model_module, 'get_model_config') else config.get('learning_model', {}).get('architecture', {}).get('use_fpd_architecture', False)
    features_config = model_config.get('features', {'image_size': [112, 112], 'grid_size': 7}) if hasattr(model_module, 'get_model_config') else config.get('learning_model', {}).get('architecture', {}).get('features', {'image_size': [112, 112], 'grid_size': 7})

    print(f"모델 설정 로드: {save_file_name}")
    print(f"타겟 포인트: {target_points}")
    print(f"FPD 아키텍처: {use_fpd_architecture}")

    # 경로 설정
    base_dir = Path(__file__).parent
    data_path = (base_dir / config['data']['source_folder']).resolve()

    # 디바이스 설정
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        print(f"\nGPU 사용: {torch.cuda.get_device_name(device)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        print("\nCPU 사용")

    def log_print(message: str = ""):
        """화면 출력"""
        print(message)

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

    # 테스트 데이터셋 생성
    log_print("\n테스트 데이터셋 생성 중...")

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

    # extract_features 옵션 읽기
    extract_features = config['training'].get('extract_features', False)

    # 타겟 포인트 정보 출력 (이미 모듈에서 로드됨)
    if target_points:
        log_print(f"타겟 포인트: {', '.join(target_points)}")

    # Autoencoder 사용 여부 확인 (모듈 설정에서 가져옴)
    use_autoencoder = features_config.get('use_autoencoder', False)

    if use_autoencoder and PointDetectorDataSetWithAutoencoder is not None:
        log_print("Autoencoder 기반 특성 추출 사용")
        encoder_path = features_config.get('encoder_path', '../model/autoencoder_16x16_best.pth')

        test_dataset = PointDetectorDataSetWithAutoencoder(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='test',
            config=dataset_config,
            augment=False,
            extract_features=True,
            encoder_path=encoder_path,
            target_points=target_points
        )
    else:
        test_dataset = PointDetectorDataSet(
            source_folder=str(data_path),
            labels_file=config['data']['labels_file'],
            detector=detector,
            mode='test',
            config=dataset_config,
            augment=False,
            extract_features=extract_features,
            target_points=target_points
        )

    # DataLoader 생성
    batch_size = config['training']['batch_size']
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    log_print(f"테스트 데이터: {len(test_dataset)}개")

    # 테스트 데이터 좌표 통계 계산 및 출력
    log_print("\n=== 테스트 데이터 레이블 좌표 통계 ===")

    # 테스트 데이터셋의 모든 타겟 좌표 수집
    all_targets = []
    # targets 속성이 있는지 확인 (특징 사전 추출 모드)
    if hasattr(test_dataset, 'targets') and test_dataset.targets is not None:
        for i in range(len(test_dataset.targets)):
            target = test_dataset.targets[i].numpy()
            if target.ndim == 1:
                all_targets.append(target)
            else:
                all_targets.append(target.flatten())
        all_targets = np.array(all_targets)  # [N, num_coords]
    else:
        # On-the-fly 처리 모드: DataLoader를 통해 타겟 수집
        for batch in test_loader:
            batch_targets = batch['targets'].numpy()  # [B, num_coords]
            for i in range(len(batch_targets)):
                all_targets.append(batch_targets[i])
        all_targets = np.array(all_targets) if len(all_targets) > 0 else np.array([])  # [N, num_coords]

    # 각 좌표별 통계 계산 (정규화된 값)
    coord_stats_normalized = {}
    coord_stats_pixel = {}

    for idx, point_name in enumerate(target_points):
        x_idx = idx * 2
        y_idx = idx * 2 + 1

        # 정규화된 좌표의 평균과 표준편차
        x_norm_mean = np.mean(all_targets[:, x_idx])
        x_norm_std = np.std(all_targets[:, x_idx])
        y_norm_mean = np.mean(all_targets[:, y_idx])
        y_norm_std = np.std(all_targets[:, y_idx])

        coord_stats_normalized[point_name] = {
            'x_mean': x_norm_mean,
            'x_std': x_norm_std,
            'y_mean': y_norm_mean,
            'y_std': y_norm_std
        }

        # 픽셀 좌표로 변환
        x_pixel_vals = all_targets[:, x_idx] * (test_dataset.coord_max_x - test_dataset.coord_min_x) + test_dataset.coord_min_x
        y_pixel_vals = all_targets[:, y_idx] * (test_dataset.coord_max_y - test_dataset.coord_min_y) + test_dataset.coord_min_y

        x_pixel_mean = np.mean(x_pixel_vals)
        x_pixel_std = np.std(x_pixel_vals)
        y_pixel_mean = np.mean(y_pixel_vals)
        y_pixel_std = np.std(y_pixel_vals)

        coord_stats_pixel[point_name] = {
            'x_mean': x_pixel_mean,
            'x_std': x_pixel_std,
            'y_mean': y_pixel_mean,
            'y_std': y_pixel_std
        }

        # 화면 출력
        log_print(f"\n[{point_name.upper()}]")
        log_print(f"  픽셀 좌표:")
        log_print(f"    X: 평균 = {x_pixel_mean:.2f}px, 표준편차 = {x_pixel_std:.2f}px")
        log_print(f"    Y: 평균 = {y_pixel_mean:.2f}px, 표준편차 = {y_pixel_std:.2f}px")

    log_print(f"\n총 테스트 샘플 수: {len(all_targets)}")
    log_print("=" * 60)

    # 모델 가중치 로드
    model_path_to_load = Path(base_dir) / args.model_path
    if not model_path_to_load.exists():
        log_print(f"\n오류: 모델 파일을 찾을 수 없습니다: {model_path_to_load}")
        return

    log_print(f"\n모델 가중치 로드 중: {model_path_to_load}")

    try:
        checkpoint = torch.load(str(model_path_to_load), map_location=device, weights_only=False)

        # 체크포인트에서 모델 설정 정보 읽기
        if 'model_config' in checkpoint:
            checkpoint_model_source = checkpoint['model_config'].get('learning_model', {}).get('source', 'Unknown')
            log_print(f"체크포인트 모델 소스: {checkpoint_model_source}")

            # 모델 소스 불일치 경고
            if checkpoint_model_source != 'Unknown' and checkpoint_model_source != model_source:
                log_print(f"\n{'='*60}")
                log_print(f"경고: 모델 소스 불일치 감지!")
                log_print(f"  - 현재 모델 소스: {model_source}")
                log_print(f"  - 체크포인트 모델 소스: {checkpoint_model_source}")
                log_print(f"")
                log_print(f"해결 방법:")
                log_print(f"  1. config.yml의 learning_model->source를 '{checkpoint_model_source}'로 변경하거나")
                log_print(f"  2. --model_source 옵션으로 올바른 모델 소스 파일 지정:")
                log_print(f"     python 299.inference_pretrained_model.py --model_source {checkpoint_model_source} --model_path {args.model_path}")
                log_print(f"{'='*60}\n")

        # 모델 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # feature_mean, feature_std 복원 (있는 경우)
        if 'feature_mean' in checkpoint:
            detector.feature_mean = checkpoint['feature_mean']
            detector.feature_std = checkpoint['feature_std']
            log_print("Feature normalization 파라미터 로드 완료")

        # 모델 정보 출력 (있는 경우)
        if 'epoch' in checkpoint:
            log_print(f"로드된 모델 에폭: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            log_print(f"모델 검증 손실: {checkpoint['best_val_loss']:.6f}")

        log_print("모델 가중치 로드 완료\n")

    except Exception as e:
        log_print(f"\n{'='*60}")
        log_print(f"오류: 모델 가중치 로드 실패")
        log_print(f"{'='*60}")
        log_print(f"오류 내용: {str(e)[:200]}...")
        log_print(f"")
        log_print(f"현재 설정:")
        log_print(f"  - 모델 소스: {model_source}")
        log_print(f"  - 모델 가중치: {model_path_to_load}")
        log_print(f"")
        log_print(f"가능한 원인:")
        log_print(f"  1. 모델 소스 파일과 가중치 파일이 서로 다른 아키텍처")
        log_print(f"  2. 가중치 파일이 손상되었거나 호환되지 않는 버전")
        log_print(f"")
        log_print(f"해결 방법:")
        log_print(f"  1. --model_source 옵션으로 올바른 모델 소스 지정")
        log_print(f"  2. --model_path 옵션으로 올바른 가중치 파일 지정")
        log_print(f"  3. 모델을 새로 학습하여 호환되는 가중치 생성")
        log_print(f"{'='*60}\n")
        return

    # 테스트 데이터 평가
    log_print("\n=== 테스트 데이터 평가 시작 ===")
    test_loss, test_errors, all_test_errors, detailed_results = validate(model, test_loader, device, data_path=str(data_path))

    # 최종 오차 분석 출력
    output_lines = display_error_analysis(
        all_test_errors,
        epoch='final',
        title="=== 최종 테스트 결과 오차 분석 ==="
    )
    for line in output_lines:
        log_print(line)

    log_print(f"\n테스트 손실: {test_loss:.6f}")
    non_zero_test_errors = [v for v in test_errors.values() if v > 0]
    final_avg_error = np.mean(non_zero_test_errors) if non_zero_test_errors else 0
    log_print(f"평균 오차: {final_avg_error:.2f} pixels")

    # 실제 예측 결과 기반 상세 오차 분석
    log_print("\n" + "=" * 60)
    log_print("=== 포인트별 상세 오차 분석 ===")
    log_print("=" * 60)

    # 각 포인트별 상세 통계 계산 및 출력
    all_distances = []
    for point_name in target_points:
        if point_name in all_test_errors and all_test_errors[point_name]['x']:
            x_errors = all_test_errors[point_name]['x']
            y_errors = all_test_errors[point_name]['y']
            dist_errors = all_test_errors[point_name]['dist']

            x_mean = np.mean(x_errors)
            x_std = np.std(x_errors)
            y_mean = np.mean(y_errors)
            y_std = np.std(y_errors)
            dist_mean = np.mean(dist_errors)
            dist_std = np.std(dist_errors)

            all_distances.extend(dist_errors)

            log_print(f"{point_name.upper():6s}: X={x_mean:4.1f}±{x_std:4.1f},    Y={y_mean:4.1f}±{y_std:4.1f},    Dist={dist_mean:4.1f}±{dist_std:4.1f}")

    # 전체 평균 계산
    if all_distances:
        overall_mean = np.mean(all_distances)
        overall_std = np.std(all_distances)
        log_print(f"전체: 평균={overall_mean:.2f}±{overall_std:.2f} pixels")

    log_print("비고) 값:좌표 오차 평균±표준편차, Dist:유클리드 거리 오차")
    log_print("=" * 60)

    # 모델 정보 출력
    log_print("\n=== 모델 정보 ===")
    log_print(f"모델 파일: {model_path_to_load}")
    log_print(f"모델 소스: {model_source}")
    log_print(f"총 파라미터: {total_params:,}")
    log_print(f"학습 가능 파라미터: {trainable_params:,}")

    # 체크포인트 정보 출력 (있는 경우)
    if 'epoch' in checkpoint:
        log_print(f"모델 에폭: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        log_print(f"검증 손실: {checkpoint['best_val_loss']:.6f}")

    # 결과 파일 저장
    result_dir = base_dir.parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"inference_pretrained_model_{timestamp}.txt"

    log_print(f"\n결과 파일 저장 중: {result_file}")

    with open(result_file, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("id,labeledXY,predictXY,X오차,Y오차\n")

        # 각 샘플 결과 작성
        for result in detailed_results:
            filename = result['filename']
            labeled = result['labeled']
            predict = result['predict']
            errors = result['errors']

            # labeledXY 포맷: point_name:(x,y);...
            labeled_str = ";".join([f"{name}:({x:.1f},{y:.1f})" for name, (x, y) in labeled.items()])

            # predictXY 포맷: point_name:(x,y);...
            predict_str = ";".join([f"{name}:({x:.1f},{y:.1f})" for name, (x, y) in predict.items()])

            # X오차,Y오차 포맷: point_name:(x_err,y_err);...
            errors_x_str = ";".join([f"{name}:({x:.1f},{y:.1f})" for name, (x, y) in errors.items()])

            # CSV 라인 작성
            f.write(f"{filename},{labeled_str},{predict_str},{errors_x_str}\n")

    log_print(f"결과 파일 저장 완료: {len(detailed_results)}개 샘플")

    # 시각화 이미지 생성
    log_print("\n시각화 이미지 생성 중...")
    visualization_file = result_dir / f"inference_pretrained_model_{timestamp}.jpg"

    try:
        create_visualization_grid(detailed_results, str(visualization_file), features_config['image_size'], num_samples=40, grid_size=(8, 5))
        log_print(f"시각화 이미지 저장 완료: {visualization_file}")
    except Exception as e:
        log_print(f"시각화 이미지 생성 실패: {e}")

    # 추론 속도 벤치마크 (GPU 환경)
    log_print("\n=== 추론 속도 벤치마크 (GPU) ===")

    # 속도 측정을 위한 샘플 수
    num_speed_samples = min(100, len(test_dataset))

    # 속도 측정용 DataLoader (batch_size=1)
    speed_test_dataset = torch.utils.data.Subset(test_dataset, range(num_speed_samples))
    speed_loader = DataLoader(
        speed_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model.eval()
    inference_times = []

    # GPU 워밍업 (10회)
    log_print(f"워밍업 중 (10회)...")
    warmup_count = min(10, num_speed_samples)
    warmup_iter = iter(speed_loader)
    for _ in range(warmup_count):
        try:
            batch = next(warmup_iter)
            data = batch['data'].to(device)
            with torch.no_grad():
                _ = model(data)
            if use_cuda:
                torch.cuda.synchronize()
        except StopIteration:
            break

    # 실제 속도 측정
    log_print(f"속도 측정 중 ({num_speed_samples}개 샘플)...")
    for batch in speed_loader:
        data = batch['data'].to(device)

        if use_cuda:
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            _ = model(data)

        if use_cuda:
            torch.cuda.synchronize()

        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000)  # ms로 변환

    # 통계 계산
    avg_time_ms = np.mean(inference_times)
    min_time_ms = np.min(inference_times)
    max_time_ms = np.max(inference_times)
    std_time_ms = np.std(inference_times)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0

    # GPU 메모리 사용량 측정
    gpu_memory_mb = 0
    if use_cuda:
        gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # 결과 출력
    log_print(f"\n샘플 수: {num_speed_samples}개 (각 이미지 개별 측정)")
    log_print(f"평균 추론 시간 (이미지당): {avg_time_ms:.2f} ms")
    log_print(f"초당 처리량 (FPS): {fps:.2f}")
    log_print(f"최소 시간 (이미지당): {min_time_ms:.2f} ms")
    log_print(f"최대 시간 (이미지당): {max_time_ms:.2f} ms")
    log_print(f"표준편차: {std_time_ms:.2f} ms")
    if use_cuda:
        log_print(f"GPU 메모리 사용: {gpu_memory_mb:.2f} MB")
    log_print("=" * 60)

    # 속도 벤치마크 결과 JSON 저장
    speed_result = {
        'num_samples': num_speed_samples,
        'measurement_unit': 'per_image',
        'device': str(device),
        'avg_time_ms_per_image': float(avg_time_ms),
        'fps': float(fps),
        'min_time_ms_per_image': float(min_time_ms),
        'max_time_ms_per_image': float(max_time_ms),
        'std_time_ms': float(std_time_ms),
        'gpu_memory_mb': float(gpu_memory_mb) if use_cuda else 0,
        'model_params': total_params,
        'batch_size': 1,
        'timestamp': datetime.now().isoformat()
    }

    speed_json_path = result_dir / f"inference_speed_{timestamp}.json"
    with open(speed_json_path, 'w', encoding='utf-8') as f:
        json.dump(speed_result, f, indent=4, ensure_ascii=False)
    log_print(f"\n속도 벤치마크 결과 저장: {speed_json_path}")

    # 통합 결과 텍스트 파일 생성 (result/inference_result_모델명_timestamp.txt)
    final_txt_path = result_dir / f"inference_result_{save_file_name}_{timestamp}.txt"

    with open(final_txt_path, 'w', encoding='utf-8') as f:
        # 테스트 데이터 레이블 좌표 통계 추가
        f.write("=" * 60 + "\n")
        f.write("테스트 데이터 레이블 좌표 통계\n")
        f.write("=" * 60 + "\n\n")

        for point_name in target_points:
            stats_norm = coord_stats_normalized[point_name]
            stats_pix = coord_stats_pixel[point_name]

            f.write(f"[{point_name.upper()}]\n")
            f.write(f"  픽셀 좌표:\n")
            f.write(f"    X: 평균 = {stats_pix['x_mean']:.2f}px, 표준편차 = {stats_pix['x_std']:.2f}px\n")
            f.write(f"    Y: 평균 = {stats_pix['y_mean']:.2f}px, 표준편차 = {stats_pix['y_std']:.2f}px\n\n")

        f.write(f"총 테스트 샘플 수: {len(all_targets)}\n")
        f.write("=" * 60 + "\n\n")

        # 테스트 결과
        f.write("=== 테스트 데이터 최종 평가 ===\n\n")
        for line in output_lines:
            f.write(line + "\n")
        f.write(f"\n테스트 손실: {test_loss:.6f}\n")
        f.write(f"평균 오차: {final_avg_error:.2f} pixels\n\n")

        # 포인트별 오차 출력
        f.write("=== 포인트별 오차 ===\n")
        for point_name, error in test_errors.items():
            f.write(f"{point_name}: {error:.2f} pixels\n")
        f.write("\n")

        # 추론 속도 벤치마크 결과 추가
        f.write("=" * 60 + "\n")
        f.write("추론 속도 벤치마크 (GPU)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"디바이스: {str(device)}\n")
        f.write(f"샘플 수: {num_speed_samples}개 (각 이미지 개별 측정)\n")
        f.write(f"평균 추론 시간 (이미지당): {avg_time_ms:.2f} ms\n")
        f.write(f"초당 처리량 (FPS): {fps:.2f}\n")
        f.write(f"최소 시간 (이미지당): {min_time_ms:.2f} ms\n")
        f.write(f"최대 시간 (이미지당): {max_time_ms:.2f} ms\n")
        f.write(f"표준편차: {std_time_ms:.2f} ms\n")
        if use_cuda:
            f.write(f"GPU 메모리 사용: {gpu_memory_mb:.2f} MB\n")
        f.write("\n")

        # 모델 정보 추가
        f.write("=" * 60 + "\n")
        f.write("모델 정보\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"모델 파일: {model_path_to_load}\n")
        f.write(f"모델 소스: {model_source}\n")
        f.write(f"총 파라미터: {total_params:,}\n")
        f.write(f"학습 가능 파라미터: {trainable_params:,}\n")
        if 'epoch' in checkpoint:
            f.write(f"모델 에폭: {checkpoint['epoch']}\n")
        if 'best_val_loss' in checkpoint:
            f.write(f"검증 손실: {checkpoint['best_val_loss']:.6f}\n")
        f.write("=" * 60 + "\n")

    log_print(f"통합 결과 텍스트 저장: {final_txt_path}")

    log_print("\n" + "=" * 60)
    log_print("테스트 완료!")
    log_print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"결과 파일 (CSV): {result_file}")
    log_print(f"결과 파일 (TXT): {final_txt_path}")
    log_print(f"시각화 파일: {visualization_file}")
    log_print(f"속도 벤치마크 파일: {speed_json_path}")
    log_print("=" * 60)


if __name__ == "__main__":
    main()