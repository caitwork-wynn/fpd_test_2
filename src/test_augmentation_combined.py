# -*- coding: utf-8 -*-
"""
Flip + Crop 조합 증강 테스트 스크립트
- 데이터셋 크기 검증
- 증강 샘플 시각화
- 좌표 정확도 확인
"""

import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
import importlib.util

# sys.path에 현재 디렉토리 추가
sys.path.append(str(Path(__file__).parent))


def load_model_module(model_path):
    """모델 모듈 동적 로드"""
    spec = importlib.util.spec_from_file_location("model_module", str(model_path))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module


def visualize_augmented_samples(dataset, base_idx=0, save_path=None):
    """
    한 원본 이미지의 모든 증강 샘플 시각화

    Args:
        dataset: DataSet 객체
        base_idx: 원본 이미지 인덱스
        save_path: 저장 경로 (None이면 저장 안 함)
    """
    # 증강 설정
    if not dataset.augment:
        print("경고: 증강이 비활성화되어 있습니다.")
        return

    augment_count = dataset.augment_count
    total_variations = 2 + 2 * augment_count  # 원본 + crop×N + flip + flip+crop×N

    print(f"\n원본 이미지 인덱스 {base_idx}의 모든 증강 샘플 ({total_variations}개):")
    print(f"  - 0: 원본")
    print(f"  - 1~{augment_count}: 원본 + crop")
    print(f"  - {augment_count + 1}: flip")
    print(f"  - {augment_count + 2}~{total_variations-1}: flip + crop")

    # 모든 증강 샘플 수집
    samples = []
    labels = []

    for variation_idx in range(total_variations):
        idx = base_idx * total_variations + variation_idx

        if idx >= len(dataset):
            print(f"경고: 인덱스 {idx}가 데이터셋 크기({len(dataset)})를 초과합니다.")
            break

        sample = dataset[idx]

        # 이미지 텐서 변환 (Grayscale [1, H, W] → [H, W])
        image_tensor = sample['data']
        if image_tensor.shape[0] == 1:  # Grayscale
            image_np = image_tensor[0].numpy()
        else:  # RGB
            image_np = image_tensor.numpy().transpose(1, 2, 0)

        # 정규화 해제 (mean=0.449, std=0.226)
        image_np = (image_np * 0.226 + 0.449) * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        # Grayscale을 RGB로 변환 (시각화용)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # 좌표 역정규화
        targets = sample['targets'].numpy()
        coords = []
        for i in range(0, len(targets), 2):
            norm_x, norm_y = targets[i], targets[i+1]
            x, y = dataset.denormalize_coordinates(norm_x, norm_y)
            coords.append((int(x), int(y)))

        # 좌표 표시
        for coord in coords:
            cv2.circle(image_np, coord, 3, (0, 0, 255), -1)

        # 라벨 텍스트
        if variation_idx == 0:
            label = "Original"
        elif variation_idx <= augment_count:
            label = f"Crop #{variation_idx}"
        elif variation_idx == augment_count + 1:
            label = "Flip"
        else:
            label = f"Flip+Crop #{variation_idx - augment_count - 1}"

        samples.append(image_np)
        labels.append(label)

    # 그리드 형태로 배치
    num_samples = len(samples)
    cols = min(num_samples, 5)
    rows = (num_samples + cols - 1) // cols

    # 이미지 크기
    h, w = samples[0].shape[:2]

    # 그리드 이미지 생성
    grid_h = rows * (h + 30) + 30
    grid_w = cols * (w + 10) + 10
    grid_image = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for i, (img, label) in enumerate(zip(samples, labels)):
        row = i // cols
        col = i % cols

        y = row * (h + 30) + 30
        x = col * (w + 10) + 5

        grid_image[y:y+h, x:x+w] = img

        # 라벨 텍스트
        cv2.putText(grid_image, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 제목 추가
    title = f"Base Index {base_idx} - All Augmentations ({num_samples} samples)"
    cv2.putText(grid_image, title, (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 저장
    if save_path:
        cv2.imwrite(str(save_path), grid_image)
        print(f"\n시각화 이미지 저장: {save_path}")

    return grid_image


def main():
    """메인 함수"""
    print("=" * 60)
    print("Flip + Crop 조합 증강 테스트")
    print("=" * 60)

    # 설정 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 로드 (현재 사용 중인 131 모델)
    model_source = config['learning_model']['source']
    model_path = (Path(__file__).parent / model_source).resolve()

    print(f"\n모델 로딩: {model_path}")
    model_module = load_model_module(model_path)

    # 모델 설정 가져오기
    if hasattr(model_module, 'get_model_config'):
        model_config = model_module.get_model_config()
        target_points = model_config['target_points']
        features_config = model_config['features']
    else:
        print("경고: get_model_config 함수를 찾을 수 없습니다.")
        return

    # 데이터 경로
    base_dir = Path(__file__).parent
    data_path = (base_dir / config['data']['source_folder']).resolve()

    # config 형식 맞추기
    dataset_config = config.copy()
    if 'data_split' not in dataset_config:
        dataset_config['data_split'] = {
            'test_id_suffix': config['data']['test_id_suffix'],
            'validation_ratio': config['data']['validation_ratio'],
            'random_seed': config['data'].get('random_seed', 42)
        }

    if 'learning_model' not in dataset_config:
        dataset_config['learning_model'] = {}
    if 'architecture' not in dataset_config['learning_model']:
        dataset_config['learning_model']['architecture'] = {}
    dataset_config['learning_model']['architecture']['features'] = features_config

    # 증강 활성화 확인
    augment_enabled = config['training']['augmentation']['enabled']
    augment_count = config['training']['augmentation']['augment_count']

    print(f"\n증강 설정:")
    print(f"  - enabled: {augment_enabled}")
    print(f"  - augment_count: {augment_count}")
    print(f"  - crop.enabled: {config['training']['augmentation']['crop']['enabled']}")
    print(f"  - flip.enabled: {config['training']['augmentation']['flip']['enabled']}")

    # DataSet 생성
    print("\n데이터셋 생성 중...")
    DataSet = model_module.DataSet

    train_dataset = DataSet(
        source_folder=str(data_path),
        labels_file=config['data']['labels_file'],
        detector=None,  # 간단한 테스트이므로 None
        mode='train',
        config=dataset_config,
        augment=augment_enabled,
        target_points=target_points
    )

    # 데이터셋 크기 검증
    print("\n=== 데이터셋 크기 검증 ===")
    base_len = len(train_dataset.data)
    expected_len = base_len * (2 + 2 * augment_count) if augment_enabled else base_len
    actual_len = len(train_dataset)

    print(f"원본 데이터 수: {base_len}")
    if augment_enabled:
        print(f"증강 배수: 2 + 2×{augment_count} = {2 + 2 * augment_count}")
        print(f"예상 총 샘플 수: {base_len} × {2 + 2 * augment_count} = {expected_len}")
    print(f"실제 총 샘플 수: {actual_len}")

    if expected_len == actual_len:
        print("[OK] 데이터셋 크기가 올바릅니다!")
    else:
        print(f"[ERROR] 데이터셋 크기 불일치! (예상: {expected_len}, 실제: {actual_len})")
        return

    # 증강 샘플 시각화
    result_dir = base_dir / '..' / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)

    base_idx = 0  # 첫 번째 이미지
    save_path = result_dir / "augmentation_combined_test.png"

    visualize_augmented_samples(train_dataset, base_idx=base_idx, save_path=save_path)

    print("\n=== 테스트 완료 ===")
    print(f"결과 이미지: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
