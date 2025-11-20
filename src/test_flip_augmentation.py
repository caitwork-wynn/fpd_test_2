# -*- coding: utf-8 -*-
"""
좌우 반전 증강 테스트 스크립트
- 샘플 이미지에 flip augmentation 적용
- 원본과 반전된 이미지를 좌우로 나란히 시각화
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# sys.path에 현재 디렉토리 추가
sys.path.append(str(Path(__file__).parent))

from util.data_augmentation import apply_flip_augmentation


def draw_points_on_image(image, coords, point_names):
    """
    이미지에 좌표 점을 원으로 표시

    Args:
        image: 원본 이미지 (BGR)
        coords: 좌표 딕셔너리
        point_names: 포인트 이름 리스트

    Returns:
        표시된 이미지 (복사본)
    """
    # 이미지 복사
    vis_image = image.copy()

    # 색상 정의 (BGR)
    colors = {
        'floor': (0, 0, 255),      # 빨간색
        'center': (255, 0, 0),     # 파란색
        'front': (0, 255, 0),      # 초록색
        'side': (0, 255, 255)      # 노란색
    }

    # 각 포인트 그리기
    for point_name in point_names:
        x_key = f'{point_name}_x'
        y_key = f'{point_name}_y'

        if x_key in coords and y_key in coords:
            x = int(coords[x_key])
            y = int(coords[y_key])
            color = colors.get(point_name, (255, 255, 255))  # 기본: 흰색

            # 원 그리기 (반지름 5, 두께 2)
            cv2.circle(vis_image, (x, y), 5, color, 2)

            # 포인트 이름 텍스트 표시
            cv2.putText(vis_image, point_name, (x + 8, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return vis_image


def load_sample_from_labels(data_folder, labels_file, sample_index=0):
    """
    labels.txt에서 샘플 데이터 로드

    Args:
        data_folder: 데이터 폴더 경로
        labels_file: 라벨 파일명
        sample_index: 샘플 인덱스

    Returns:
        image: 이미지 (BGR)
        coords: 좌표 딕셔너리
        point_names: 포인트 이름 리스트
    """
    labels_path = Path(data_folder) / labels_file

    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 헤더 파싱
    header = lines[0].strip().split(',')

    # 샘플 데이터 파싱 (인덱스 범위 체크)
    if sample_index + 1 >= len(lines):
        sample_index = 0  # 범위 초과 시 첫 번째 샘플 사용

    data_line = lines[sample_index + 1].strip().split(',')

    # 딕셔너리 생성
    sample_data = {}
    for i, key in enumerate(header):
        if i < len(data_line):
            sample_data[key] = data_line[i]

    # 이미지 로드
    filename = sample_data['filename']
    image_path = Path(data_folder) / filename
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 좌표 추출 (동적으로 모든 *_x, *_y 찾기)
    coords = {}
    point_names = set()

    for key in header:
        if key.endswith('_x') or key.endswith('_y'):
            if key in sample_data:
                coords[key] = float(sample_data[key])
                # 포인트 이름 추출
                point_name = key[:-2]  # _x 또는 _y 제거
                point_names.add(point_name)

    point_names = sorted(list(point_names))

    return image, coords, point_names, filename


def create_side_by_side_visualization(original_image, original_coords,
                                      flipped_image, flipped_coords,
                                      point_names, filename):
    """
    원본과 반전 이미지를 좌우로 나란히 배치

    Args:
        original_image: 원본 이미지
        original_coords: 원본 좌표
        flipped_image: 반전 이미지
        flipped_coords: 반전 좌표
        point_names: 포인트 이름 리스트
        filename: 원본 파일명

    Returns:
        결합된 이미지
    """
    # 원본에 좌표 표시
    vis_original = draw_points_on_image(original_image, original_coords, point_names)

    # 반전에 좌표 표시
    vis_flipped = draw_points_on_image(flipped_image, flipped_coords, point_names)

    # 높이 맞추기
    h1, w1 = vis_original.shape[:2]
    h2, w2 = vis_flipped.shape[:2]
    max_h = max(h1, h2)

    # 패딩 추가 (높이 맞추기)
    if h1 < max_h:
        pad = max_h - h1
        vis_original = cv2.copyMakeBorder(vis_original, 0, pad, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if h2 < max_h:
        pad = max_h - h2
        vis_flipped = cv2.copyMakeBorder(vis_flipped, 0, pad, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 가운데 구분선 (흰색, 두께 3)
    separator = np.ones((max_h, 3, 3), dtype=np.uint8) * 255

    # 좌우로 결합
    combined = np.hstack([vis_original, separator, vis_flipped])

    # 텍스트 레이블 추가
    label_h = 40
    label_img = np.zeros((label_h, combined.shape[1], 3), dtype=np.uint8)

    # 왼쪽 레이블
    cv2.putText(label_img, f"Original: {filename}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 오른쪽 레이블
    right_x = w1 + 3 + 10
    cv2.putText(label_img, "Flipped (Horizontal)", (right_x, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 상단에 레이블 추가
    final_image = np.vstack([label_img, combined])

    return final_image


def main():
    """메인 함수"""
    print("=" * 60)
    print("좌우 반전(Flip) 증강 테스트")
    print("=" * 60)

    # 경로 설정
    base_dir = Path(__file__).parent
    data_folder = base_dir / '..' / 'data' / 'learning'
    labels_file = 'labels.txt'
    result_dir = base_dir / '..' / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 데이터 로드 (인덱스 10 사용)
    print("\n1. 샘플 데이터 로드 중...")
    sample_index = 10
    image, coords, point_names, filename = load_sample_from_labels(
        data_folder, labels_file, sample_index
    )

    print(f"   파일: {filename}")
    print(f"   이미지 크기: {image.shape}")
    print(f"   포인트: {point_names}")
    print(f"   좌표: {coords}")

    # Flip 증강 적용
    print("\n2. Flip 증강 적용 중...")
    flipped_image, flipped_coords, flipped_direction = apply_flip_augmentation(
        image, coords, direction=None
    )

    print(f"   반전된 좌표: {flipped_coords}")

    # 시각화 이미지 생성
    print("\n3. 시각화 이미지 생성 중...")
    vis_image = create_side_by_side_visualization(
        image, coords,
        flipped_image, flipped_coords,
        point_names, filename
    )

    # 결과 저장
    output_path = result_dir / "flip_augmentation_test.png"
    cv2.imwrite(str(output_path), vis_image)

    print(f"\n[OK] 시각화 이미지 저장 완료: {output_path}")
    print(f"   크기: {vis_image.shape}")

    # 좌표 검증
    print("\n4. 좌표 변환 검증:")
    img_width = image.shape[1]
    for point_name in point_names:
        x_key = f'{point_name}_x'
        if x_key in coords:
            original_x = coords[x_key]
            flipped_x = flipped_coords[x_key]
            expected_x = img_width - original_x

            print(f"   {point_name}_x: {original_x:.2f} → {flipped_x:.2f}")
            print(f"      (예상: {expected_x:.2f}, 오차: {abs(flipped_x - expected_x):.6f})")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
