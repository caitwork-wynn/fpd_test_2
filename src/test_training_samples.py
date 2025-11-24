# -*- coding: utf-8 -*-
"""
Training 샘플 예측 정확도 확인
- Training 데이터에 대한 모델 예측 성능 검증
"""

import sys
import torch
import numpy as np
import cv2
import yaml
import importlib.util
from pathlib import Path

# 모듈 로드
sys.path.append(str(Path(__file__).parent))


def preprocess_image(image_path, image_size=96):
    """
    이미지 전처리 (모델과 동일)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    # BGR -> Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, (image_size, image_size))

    # 정규화 [0, 255] -> [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # 학습 시 사용한 정규화 적용
    mean = 0.449
    std = 0.226
    normalized = (normalized - mean) / std

    # Tensor 변환: [H, W] -> [1, 1, H, W]
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)

    return tensor


def denormalize_coords(norm_x, norm_y, coord_ranges):
    """
    정규화된 좌표를 픽셀 좌표로 변환
    """
    x_min, x_max = coord_ranges['x']
    y_min, y_max = coord_ranges['y']

    pred_x = norm_x * (x_max - x_min) + x_min
    pred_y = norm_y * (y_max - y_min) + y_min

    return pred_x, pred_y


def main():
    print("=" * 80)
    print("Training 샘플 예측 정확도 확인")
    print("=" * 80)

    # Config 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 모듈 로드
    model_source = config['learning_model']['source']
    model_path = Path(__file__).parent / model_source
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 모델 config 가져오기
    if hasattr(model_module, 'get_model_config'):
        model_config = model_module.get_model_config()
        save_file_name = model_config['save_file_name']
        features_config = model_config['features']
    else:
        save_file_name = 'mpm_lightweight_floor_optim_96'
        features_config = {'image_size': [96, 96]}

    # 경로 설정
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "learning"
    labels_file = data_dir / "labels.txt"

    # 모델 경로
    checkpoint_dir = base_dir / "model"
    pth_path = checkpoint_dir / f"{save_file_name}_best.pth"

    if not pth_path.exists():
        print(f"[오류] PyTorch 모델을 찾을 수 없습니다: {pth_path}")
        return

    print(f"\n[모델] {pth_path.name}")

    # Labels 로드
    print(f"\n[데이터] {labels_file}")

    training_samples = []
    test_samples = []

    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 헤더 건너뛰기
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                try:
                    sample_id = int(parts[0])
                    filename = parts[2]
                    floor_x = float(parts[3])
                    floor_y = float(parts[4])

                    sample = {
                        'id': sample_id,
                        'filename': filename,
                        'label_x': floor_x,
                        'label_y': floor_y
                    }

                    # Test 샘플 기준: ID % 10 == 1
                    if sample_id % 10 == 1:
                        test_samples.append(sample)
                    else:
                        training_samples.append(sample)
                except ValueError:
                    continue

    print(f"Training 샘플 수: {len(training_samples)}")
    print(f"Test 샘플 수: {len(test_samples)}")

    # 좌표 범위
    coord_ranges = {
        'x': (-112, 224),
        'y': (0, 224)
    }

    image_size = features_config['image_size'][0]

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 모델 로드
    print("\n모델 로딩...")
    model_class = model_module.MPMMobileNetLightweightModel
    model_init_config = {
        'features': features_config,
        'training': config.get('training', {}),
        'target_points': model_config.get('target_points', ['floor']),
        'use_fpd_architecture': model_config.get('use_fpd_architecture', False)
    }
    model = model_class(model_init_config)
    model = model.to(device)

    checkpoint = torch.load(str(pth_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("모델 로딩 완료")

    # Training 샘플 예측 (처음 10개)
    print("\n" + "=" * 80)
    print("[Training 샘플 예측] (처음 10개)")
    print("=" * 80)

    training_errors = []

    for i, sample in enumerate(training_samples[:10]):
        image_path = data_dir / sample['filename']

        if not image_path.exists():
            print(f"[경고] 이미지 없음: {sample['filename']}")
            continue

        # 전처리
        input_tensor = preprocess_image(image_path, image_size)

        # 추론
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)
            coords_norm = outputs['coordinates'].cpu().numpy()[0]

        # 역정규화
        pred_x, pred_y = denormalize_coords(
            coords_norm[0], coords_norm[1], coord_ranges
        )

        # 오차 계산
        error = np.sqrt((pred_x - sample['label_x'])**2 + (pred_y - sample['label_y'])**2)
        training_errors.append(error)

        print(f"\n{i+1}. ID: {sample['id']:05d} ({sample['filename']})")
        print(f"   정답:  ({sample['label_x']:.2f}, {sample['label_y']:.2f})")
        print(f"   예측:  ({pred_x:.2f}, {pred_y:.2f})")
        print(f"   오차:  {error:.2f} pixels")

    # Training 샘플 통계
    print("\n" + "=" * 80)
    print("[Training 샘플 통계]")
    print("=" * 80)
    print(f"평균 오차: {np.mean(training_errors):.2f} pixels")
    print(f"최소 오차: {np.min(training_errors):.2f} pixels")
    print(f"최대 오차: {np.max(training_errors):.2f} pixels")
    print(f"표준편차: {np.std(training_errors):.2f} pixels")

    # Test 샘플 예측 (처음 10개)
    print("\n" + "=" * 80)
    print("[Test 샘플 예측] (처음 10개)")
    print("=" * 80)

    test_errors = []

    for i, sample in enumerate(test_samples[:10]):
        image_path = data_dir / sample['filename']

        if not image_path.exists():
            print(f"[경고] 이미지 없음: {sample['filename']}")
            continue

        # 전처리
        input_tensor = preprocess_image(image_path, image_size)

        # 추론
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)
            coords_norm = outputs['coordinates'].cpu().numpy()[0]

        # 역정규화
        pred_x, pred_y = denormalize_coords(
            coords_norm[0], coords_norm[1], coord_ranges
        )

        # 오차 계산
        error = np.sqrt((pred_x - sample['label_x'])**2 + (pred_y - sample['label_y'])**2)
        test_errors.append(error)

        print(f"\n{i+1}. ID: {sample['id']:05d} ({sample['filename']})")
        print(f"   정답:  ({sample['label_x']:.2f}, {sample['label_y']:.2f})")
        print(f"   예측:  ({pred_x:.2f}, {pred_y:.2f})")
        print(f"   오차:  {error:.2f} pixels")

    # Test 샘플 통계
    print("\n" + "=" * 80)
    print("[Test 샘플 통계]")
    print("=" * 80)
    print(f"평균 오차: {np.mean(test_errors):.2f} pixels")
    print(f"최소 오차: {np.min(test_errors):.2f} pixels")
    print(f"최대 오차: {np.max(test_errors):.2f} pixels")
    print(f"표준편차: {np.std(test_errors):.2f} pixels")

    # 비교
    print("\n" + "=" * 80)
    print("[Training vs Test 비교]")
    print("=" * 80)
    print(f"Training 평균 오차: {np.mean(training_errors):.2f} pixels")
    print(f"Test 평균 오차:     {np.mean(test_errors):.2f} pixels")
    print(f"차이:              {abs(np.mean(test_errors) - np.mean(training_errors)):.2f} pixels")

    if np.mean(training_errors) < 20 and np.mean(test_errors) > 40:
        print("\n[결론] Overfitting 의심!")
        print("-> Training 데이터는 잘 학습했지만, Test 데이터에는 일반화가 안 됨")
    elif np.mean(training_errors) > 40 and np.mean(test_errors) > 40:
        print("\n[결론] Underfitting 의심!")
        print("-> Training과 Test 모두 학습이 제대로 안 됨")
    elif abs(np.mean(test_errors) - np.mean(training_errors)) < 10:
        print("\n[결론] 일반화 성공!")
        print("-> Training과 Test의 성능이 비슷함")
    else:
        print("\n[결론] 추가 분석 필요")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
