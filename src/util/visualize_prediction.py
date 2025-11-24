# -*- coding: utf-8 -*-
"""
예측 결과 시각화 유틸리티
- 예측 포인트와 정답 포인트를 이미지 위에 표시
- Epoch별 진행 상태 합성 이미지 생성
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional


def draw_prediction_and_label(
    image_path: str,
    pred_coords: Dict[str, float],
    label_coords: Dict[str, float],
    point_names: List[str],
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    이미지 위에 예측 포인트와 정답 포인트를 함께 표시

    Parameters:
    -----------
    image_path : str
        원본 이미지 경로
    pred_coords : Dict[str, float]
        예측 좌표 {'floor_x': float, 'floor_y': float, ...}
    label_coords : Dict[str, float]
        정답 좌표 {'floor_x': float, 'floor_y': float, ...}
    point_names : List[str]
        포인트 이름 리스트 (예: ['floor'])
    output_size : Tuple[int, int], optional
        출력 이미지 크기 (width, height). None이면 원본 크기 유지

    Returns:
    --------
    np.ndarray
        시각화된 이미지 (BGR)
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    # 출력 크기 조정
    if output_size is not None:
        image = cv2.resize(image, output_size)
        # 좌표도 비율에 맞게 조정
        h, w = cv2.imread(str(image_path)).shape[:2]
        scale_x = output_size[0] / w
        scale_y = output_size[1] / h
    else:
        scale_x = 1.0
        scale_y = 1.0

    # 포인트별 색상 정의
    point_colors = {
        'center': (0, 255, 0),      # 초록색 (BGR)
        'floor': (0, 255, 255),     # 노란색
        'front': (255, 0, 255),     # 자홍색
        'side': (255, 255, 0)       # 청록색
    }

    # 각 포인트 그리기
    total_error = 0.0
    error_count = 0

    for point_name in point_names:
        pred_x_key = f'{point_name}_x'
        pred_y_key = f'{point_name}_y'

        if pred_x_key not in pred_coords or pred_y_key not in pred_coords:
            continue
        if pred_x_key not in label_coords or pred_y_key not in label_coords:
            continue

        # 좌표 추출 및 스케일링
        pred_x = int(pred_coords[pred_x_key] * scale_x)
        pred_y = int(pred_coords[pred_y_key] * scale_y)
        label_x = int(label_coords[pred_x_key] * scale_x)
        label_y = int(label_coords[pred_y_key] * scale_y)

        # 오차 계산 (원본 좌표 기준)
        error = np.sqrt(
            (pred_coords[pred_x_key] - label_coords[pred_x_key])**2 +
            (pred_coords[pred_y_key] - label_coords[pred_y_key])**2
        )
        total_error += error
        error_count += 1

        # 포인트 색상
        color = point_colors.get(point_name, (255, 255, 255))

        # 정답 포인트 그리기 (초록색, 속이 빈 원)
        cv2.circle(image, (label_x, label_y), 5, (0, 255, 0), 2)
        cv2.circle(image, (label_x, label_y), 2, (0, 255, 0), -1)

        # 예측 포인트 그리기 (빨간색, 속이 찬 원)
        cv2.circle(image, (pred_x, pred_y), 5, (0, 0, 255), 2)
        cv2.circle(image, (pred_x, pred_y), 2, (0, 0, 255), -1)

        # 예측-정답 연결선 (회색 점선)
        cv2.line(image, (pred_x, pred_y), (label_x, label_y), (128, 128, 128), 1, cv2.LINE_AA)


    return image


def create_progress_composite(
    test_image_dir: Path,
    sample_id: str,
    output_path: str,
    max_images: Optional[int] = None
) -> None:
    """
    특정 ID의 모든 epoch 이미지를 하나로 합성 (오른쪽→왼쪽 시간순)

    Parameters:
    -----------
    test_image_dir : Path
        테스트 이미지가 저장된 디렉토리
    sample_id : str
        샘플 ID (예: "00001")
    output_path : str
        출력 합성 이미지 경로
    max_images : int, optional
        최대 표시할 이미지 개수. None이면 모두 표시
    """
    # 해당 ID의 모든 이미지 찾기 (패턴: {id}-{epoch}.jpg)
    image_files = sorted(
        test_image_dir.glob(f"{sample_id}-*.jpg"),
        key=lambda p: int(p.stem.split('-')[1])  # epoch 번호로 정렬
    )

    if not image_files:
        print(f"샘플 {sample_id}의 이미지를 찾을 수 없습니다.")
        return

    # 최대 개수 제한
    if max_images is not None and len(image_files) > max_images:
        # 균등 샘플링
        indices = np.linspace(0, len(image_files) - 1, max_images, dtype=int)
        image_files = [image_files[i] for i in indices]

    # 첫 번째 이미지로 크기 확인
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"이미지를 읽을 수 없습니다: {image_files[0]}")
        return

    img_h, img_w = first_image.shape[:2]

    # 여백 설정
    margin = 10
    text_height = 30

    # 합성 이미지 크기 계산
    composite_width = len(image_files) * (img_w + margin) + margin
    composite_height = img_h + text_height + 2 * margin

    # 빈 캔버스 생성 (흰색 배경)
    composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 255

    # 이미지를 오른쪽부터 왼쪽으로 배치
    for idx, img_path in enumerate(reversed(image_files)):  # 역순으로 배치
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 크기가 다르면 리사이즈
        if img.shape[:2] != (img_h, img_w):
            img = cv2.resize(img, (img_w, img_h))

        # 배치 위치 계산 (오른쪽부터)
        x_offset = composite_width - (idx + 1) * (img_w + margin)
        y_offset = margin

        # 이미지 배치
        composite[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = img

        # Epoch 번호 텍스트 추가 (이미지 하단)
        epoch_num = img_path.stem.split('-')[1]
        text = f"E{epoch_num}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        text_x = x_offset + (img_w - text_w) // 2
        text_y = y_offset + img_h + 20

        cv2.putText(composite, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)


def create_all_progress_composites(
    test_image_dir: Path,
    output_dir: Path,
    model_name: str,
    max_images: Optional[int] = None
) -> None:
    """
    test_image 디렉토리의 모든 ID에 대해 진행 상태 합성 이미지 생성

    Parameters:
    -----------
    test_image_dir : Path
        테스트 이미지 디렉토리
    output_dir : Path
        출력 디렉토리
    model_name : str
        모델 이름 (파일명에 사용)
    max_images : int, optional
        ID별 최대 표시 이미지 개수
    """
    # 모든 이미지 파일에서 고유 ID 추출
    image_files = list(test_image_dir.glob("*-*.jpg"))
    unique_ids = set()

    for img_file in image_files:
        sample_id = img_file.stem.split('-')[0]
        unique_ids.add(sample_id)

    # 각 ID별로 합성 이미지 생성
    for sample_id in sorted(unique_ids):
        output_path = output_dir / f"{model_name}_{sample_id}_progress.jpg"
        create_progress_composite(
            test_image_dir=test_image_dir,
            sample_id=sample_id,
            output_path=str(output_path),
            max_images=max_images
        )


if __name__ == "__main__":
    # 테스트 코드
    print("예측 시각화 유틸리티 모듈")
    print("사용 예제:")
    print("""
    from util.visualize_prediction import draw_prediction_and_label

    vis_image = draw_prediction_and_label(
        image_path='../data/learning/000002.jpg',
        pred_coords={'floor_x': 48.5, 'floor_y': 34.2},
        label_coords={'floor_x': 48.24, 'floor_y': 33.89},
        point_names=['floor']
    )
    cv2.imwrite('output.jpg', vis_image)
    """)
