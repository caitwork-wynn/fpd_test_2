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
    elif 'point_names' in sample_batch:
        point_names = sample_batch['point_names']
    else:
        # 좌표 개수로부터 추론
        num_coords = sample_output['coordinates'].shape[1]
        if num_coords == 2:
            point_names = ['floor']
        elif num_coords == 8:
            point_names = ['center', 'floor', 'front', 'side']
        else:
            point_names = [f'point_{i//2}' for i in range(0, num_coords, 2)]

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


def create_visualization_grid(detailed_results, output_path, num_samples=40, grid_size=(8, 5)):
    """
    테스트 결과를 그리드 형태로 시각화하여 저장

    Args:
        detailed_results: 테스트 결과 리스트
        output_path: 저장할 파일 경로
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

        # 파란색으로 실제 좌표 그리기
        labeled = result['labeled']
        for point_name, (x, y) in labeled.items():
            cv2.circle(img_copy, (int(x), int(y)), 5, (0, 0, 255), -1)  # 파란색

        # 빨간색으로 예측 좌표 그리기
        predict = result['predict']
        for point_name, (x, y) in predict.items():
            cv2.circle(img_copy, (int(x), int(y)), 5, (255, 0, 0), -1)  # 빨간색

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
        default='../model/floor_attention_pre_trained.pth',
        help='테스트할 모델 파일 경로 (기본값: ../model/floor_attention_pre_trained.pth)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("범용 다중 포인트 검출 모델 테스트")
    print("- config.yml 기반 동적 모델 로딩")
    print(f"- 모델 파일: {args.model_path}")
    print("=" * 60)

    # 설정 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 설정 정보 읽기
    save_file_name = config['learning_model']['checkpointing']['save_file_name']

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
    detector_config['features'] = config['learning_model']['architecture']['features']
    detector_config['training'] = config['training']
    detector = PointDetector(detector_config, device)
    model = detector.model

    # 모델 아키텍처 정보 출력
    log_print("\n=== 모델 아키텍처 ===")
    log_print(f"모델 소스: {model_source}")
    features_config = config['learning_model']['architecture']['features']
    log_print(f"입력: 3x{features_config['image_size'][0]}x{features_config['image_size'][1]} 이미지")
    log_print(f"그리드 크기: {features_config['grid_size']}x{features_config['grid_size']}")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"\n총 파라미터 수: {total_params:,}")
    log_print(f"학습 가능한 파라미터 수: {trainable_params:,}")

    # 테스트 데이터셋 생성
    log_print("\n테스트 데이터셋 생성 중...")

    # config 형식 맞추기
    dataset_config = config.copy()
    if 'data_split' not in dataset_config:
        dataset_config['data_split'] = {
            'test_id_suffix': config['data']['test_id_suffix'],
            'validation_ratio': config['data']['validation_ratio'],
            'random_seed': config['data'].get('random_seed', 42)
        }

    # extract_features 옵션 읽기
    extract_features = config['training'].get('extract_features', False)

    # 타겟 포인트 설정 읽기
    target_points = config.get('learning_model', {}).get('target_points', None)
    if target_points:
        log_print(f"타겟 포인트: {', '.join(target_points)}")

    # Autoencoder 사용 여부 확인
    use_autoencoder = config['learning_model']['architecture']['features'].get('use_autoencoder', False)

    if use_autoencoder and PointDetectorDataSetWithAutoencoder is not None:
        log_print("Autoencoder 기반 특성 추출 사용")
        encoder_path = config['learning_model']['architecture']['features'].get('encoder_path', '../model/autoencoder_16x16_best.pth')

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

    # 모델 가중치 로드
    model_path_to_load = Path(base_dir) / args.model_path
    if not model_path_to_load.exists():
        log_print(f"\n오류: 모델 파일을 찾을 수 없습니다: {model_path_to_load}")
        return

    log_print(f"\n모델 로드 중: {model_path_to_load}")

    try:
        checkpoint = torch.load(str(model_path_to_load), map_location=device, weights_only=False)

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

        log_print("모델 로드 완료\n")

    except Exception as e:
        log_print(f"\n오류: 모델 로드 실패 - {e}")
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

    # 포인트별 오차 출력
    log_print("\n=== 포인트별 오차 ===")
    for point_name, error in test_errors.items():
        log_print(f"{point_name}: {error:.2f} pixels")

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
        create_visualization_grid(detailed_results, str(visualization_file), num_samples=40, grid_size=(8, 5))
        log_print(f"시각화 이미지 저장 완료: {visualization_file}")
    except Exception as e:
        log_print(f"시각화 이미지 생성 실패: {e}")

    log_print("\n" + "=" * 60)
    log_print("테스트 완료!")
    log_print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"결과 파일: {result_file}")
    log_print(f"시각화 파일: {visualization_file}")
    log_print("=" * 60)


if __name__ == "__main__":
    main()