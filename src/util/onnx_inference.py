# -*- coding: utf-8 -*-
"""
ONNX 모델 추론 유틸리티
- onnxruntime을 사용한 ONNX 모델 추론
- 테스트 샘플 필터링 (ID % 10 == 1)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import onnxruntime as ort


def predict_with_onnx(
    onnx_path: str,
    image_path: str,
    coord_ranges: Dict[str, Tuple[float, float]],
    image_size: int = 96
) -> Tuple[float, float]:
    """
    ONNX 모델로 이미지의 포인트 좌표 예측

    Parameters:
    -----------
    onnx_path : str
        ONNX 모델 파일 경로
    image_path : str
        입력 이미지 파일 경로
    coord_ranges : Dict[str, Tuple[float, float]]
        좌표 범위 {'x': (min_x, max_x), 'y': (min_y, max_y)}
    image_size : int
        모델 입력 이미지 크기 (기본 96)

    Returns:
    --------
    Tuple[float, float]
        예측된 픽셀 좌표 (x, y)
    """
    # ONNX 세션 생성
    sess = ort.InferenceSession(str(onnx_path))

    # 입력 이름 확인
    input_name = sess.get_inputs()[0].name

    # 이미지 전처리
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    # BGR -> Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 리사이즈
    resized = cv2.resize(gray, (image_size, image_size))

    # 정규화 [0, 255] -> [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # 학습 시 사용한 정규화 적용
    mean = 0.449
    std = 0.226
    normalized = (normalized - mean) / std

    # 배치 차원 추가: [H, W] -> [1, 1, H, W]
    input_tensor = normalized[np.newaxis, np.newaxis, :, :]

    # 추론
    outputs = sess.run(None, {input_name: input_tensor})

    # 출력 좌표 (정규화된 값 [0, 1])
    coords = outputs[0][0]  # [num_coords]

    # 역정규화: [0, 1] -> 픽셀 좌표
    x_min, x_max = coord_ranges['x']
    y_min, y_max = coord_ranges['y']

    pred_x = coords[0] * (x_max - x_min) + x_min
    pred_y = coords[1] * (y_max - y_min) + y_min

    return float(pred_x), float(pred_y)


def get_test_samples_by_id_filter(
    test_dataset,
    filter_func=None
) -> List[Dict]:
    """
    테스트 데이터셋에서 특정 조건의 샘플만 필터링

    Parameters:
    -----------
    test_dataset : Dataset
        테스트 데이터셋 객체
    filter_func : callable, optional
        필터 함수. None이면 기본값 (ID % 10 == 1) 사용
        함수 시그니처: filter_func(sample_id: str) -> bool

    Returns:
    --------
    List[Dict]
        필터링된 샘플 정보 리스트
        각 딕셔너리: {
            'id': str,           # 샘플 ID (예: "00001")
            'filename': str,     # 파일명 (예: "000002.jpg")
            'image_path': str,   # 전체 이미지 경로
            'label_coords': Dict # 정답 좌표 {'floor_x': float, 'floor_y': float, ...}
        }
    """
    if filter_func is None:
        # 기본 필터: ID를 정수로 변환해서 10으로 나눈 나머지가 1
        def filter_func(sample_id: str) -> bool:
            try:
                id_num = int(sample_id)
                return id_num % 10 == 1
            except ValueError:
                return False

    filtered_samples = []

    # 데이터셋의 samples 또는 data 속성에서 접근
    dataset_samples = None
    if hasattr(test_dataset, 'samples'):
        dataset_samples = test_dataset.samples
    elif hasattr(test_dataset, 'data'):
        dataset_samples = test_dataset.data

    if dataset_samples is not None:
        for sample in dataset_samples:
            sample_id = sample['id']
            if filter_func(sample_id):
                # 이미지 전체 경로 생성
                image_path = Path(test_dataset.source_folder) / sample['filename']

                # 정답 좌표 추출
                label_coords = {}
                for key, value in sample.items():
                    if key.endswith('_x') or key.endswith('_y'):
                        label_coords[key] = value

                filtered_samples.append({
                    'id': sample_id,
                    'filename': sample['filename'],
                    'image_path': str(image_path),
                    'label_coords': label_coords
                })

    return filtered_samples


def batch_predict_onnx(
    onnx_path: str,
    samples: List[Dict],
    coord_ranges: Dict[str, Tuple[float, float]],
    image_size: int = 96
) -> List[Dict]:
    """
    여러 샘플에 대해 일괄 ONNX 추론

    Parameters:
    -----------
    onnx_path : str
        ONNX 모델 경로
    samples : List[Dict]
        샘플 정보 리스트 (get_test_samples_by_id_filter 출력 형식)
    coord_ranges : Dict[str, Tuple[float, float]]
        좌표 범위
    image_size : int
        입력 이미지 크기

    Returns:
    --------
    List[Dict]
        예측 결과 리스트
        각 딕셔너리: {
            'id': str,
            'filename': str,
            'image_path': str,
            'label_coords': Dict,
            'pred_coords': Dict  # {'floor_x': float, 'floor_y': float}
        }
    """
    results = []

    for sample in samples:
        try:
            # ONNX 추론
            pred_x, pred_y = predict_with_onnx(
                onnx_path=onnx_path,
                image_path=sample['image_path'],
                coord_ranges=coord_ranges,
                image_size=image_size
            )

            # 결과 저장
            result = sample.copy()
            result['pred_coords'] = {
                'floor_x': pred_x,
                'floor_y': pred_y
            }
            results.append(result)

        except Exception as e:
            print(f"샘플 {sample['id']} 추론 실패: {e}")
            continue

    return results


if __name__ == "__main__":
    # 테스트 코드
    print("ONNX 추론 유틸리티 모듈")
    print("사용 예제:")
    print("""
    from util.onnx_inference import predict_with_onnx

    coord_ranges = {'x': (-112, 224), 'y': (0, 224)}
    pred_x, pred_y = predict_with_onnx(
        onnx_path='../model/model_best.onnx',
        image_path='../data/learning/000002.jpg',
        coord_ranges=coord_ranges,
        image_size=96
    )
    print(f"예측 좌표: ({pred_x:.2f}, {pred_y:.2f})")
    """)
