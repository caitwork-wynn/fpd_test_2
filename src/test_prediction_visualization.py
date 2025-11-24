# -*- coding: utf-8 -*-
"""
예측 시각화 기능 테스트 스크립트
"""

import sys
from pathlib import Path
import numpy as np

# sys.path 설정
sys.path.append(str(Path(__file__).parent))

from util.onnx_inference import get_test_samples_by_id_filter, predict_with_onnx
from util.visualize_prediction import draw_prediction_and_label, create_progress_composite
import cv2


def test_filter_function():
    """ID 필터링 함수 테스트"""
    print("=" * 60)
    print("1. ID 필터링 함수 테스트")
    print("=" * 60)

    # 간단한 Mock 데이터셋
    class MockDataset:
        def __init__(self):
            self.samples = [
                {'id': '00001', 'filename': '000002.jpg', 'floor_x': 48.24, 'floor_y': 33.89},
                {'id': '00011', 'filename': '000012.jpg', 'floor_x': 56.88, 'floor_y': 44.80},
                {'id': '00021', 'filename': '000022.jpg', 'floor_x': 33.40, 'floor_y': 46.19},
                {'id': '00002', 'filename': '000003.jpg', 'floor_x': 38.07, 'floor_y': 49.99},
            ]
            self.source_folder = '../data/learning'

    test_dataset = MockDataset()
    filtered = get_test_samples_by_id_filter(test_dataset)

    print(f"전체 샘플 수: {len(test_dataset.samples)}")
    print(f"필터링된 샘플 수: {len(filtered)}")
    print("\n필터링된 샘플 ID:")
    for sample in filtered:
        print(f"  - ID: {sample['id']}, 파일명: {sample['filename']}")

    # 검증: ID가 00001, 00011, 00021만 선택되어야 함
    expected_ids = {'00001', '00011', '00021'}
    actual_ids = {s['id'] for s in filtered}
    assert actual_ids == expected_ids, f"필터링 실패: {actual_ids} != {expected_ids}"
    print("\n[OK] 필터링 함수 테스트 성공!\n")


def test_visualization():
    """시각화 함수 테스트 (실제 이미지 필요)"""
    print("=" * 60)
    print("2. 시각화 함수 테스트")
    print("=" * 60)

    # 테스트 이미지 경로
    base_dir = Path(__file__).parent.parent
    test_image_path = base_dir / 'data' / 'learning' / '000002.jpg'

    if not test_image_path.exists():
        print(f"[WARNING] 테스트 이미지가 없습니다: {test_image_path}")
        print("시각화 테스트를 건너뜁니다.\n")
        return

    print(f"테스트 이미지: {test_image_path}")

    # 가상의 예측 좌표와 정답 좌표
    pred_coords = {'floor_x': 50.0, 'floor_y': 35.0}
    label_coords = {'floor_x': 48.24, 'floor_y': 33.89}

    try:
        vis_image = draw_prediction_and_label(
            image_path=str(test_image_path),
            pred_coords=pred_coords,
            label_coords=label_coords,
            point_names=['floor'],
            output_size=None
        )

        # 테스트 출력 저장
        output_dir = base_dir / 'test_output'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'test_visualization.jpg'
        cv2.imwrite(str(output_path), vis_image)

        print(f"[OK] 시각화 이미지 저장: {output_path}")
        print(f"   이미지 크기: {vis_image.shape}\n")

    except Exception as e:
        print(f"[ERROR] 시각화 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()


def test_progress_composite():
    """진행 상태 합성 이미지 테스트"""
    print("=" * 60)
    print("3. 진행 상태 합성 이미지 테스트")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent
    test_image_dir = base_dir / 'test_output' / 'test_images'
    test_image_dir.mkdir(parents=True, exist_ok=True)

    # 테스트용 더미 이미지 생성 (실제 이미지 대신)
    sample_id = '00001'
    epochs = [10, 20, 30]

    print(f"더미 테스트 이미지 생성 (ID: {sample_id}, Epochs: {epochs})")

    for epoch in epochs:
        # 간단한 컬러 이미지 생성
        img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

        # Epoch 번호를 이미지에 표시
        cv2.putText(img, f"Epoch {epoch}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 저장
        img_path = test_image_dir / f"{sample_id}-{epoch}.jpg"
        cv2.imwrite(str(img_path), img)

    print(f"더미 이미지 저장 완료: {test_image_dir}")

    # 진행 상태 합성 이미지 생성
    try:
        output_path = base_dir / 'test_output' / f'test_model_{sample_id}_progress.jpg'
        create_progress_composite(
            test_image_dir=test_image_dir,
            sample_id=sample_id,
            output_path=str(output_path),
            max_images=None
        )

        # 결과 확인
        if output_path.exists():
            composite_img = cv2.imread(str(output_path))
            print(f"[OK] 진행 상태 합성 이미지 생성 성공")
            print(f"   저장 위치: {output_path}")
            print(f"   이미지 크기: {composite_img.shape}\n")
        else:
            print(f"[ERROR] 합성 이미지가 생성되지 않았습니다.\n")

    except Exception as e:
        print(f"[ERROR] 진행 상태 합성 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 60)
    print("예측 시각화 기능 테스트")
    print("=" * 60 + "\n")

    try:
        test_filter_function()
        test_visualization()
        test_progress_composite()

        print("=" * 60)
        print("[OK] 모든 테스트 완료!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("[ERROR] 테스트 실패")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
