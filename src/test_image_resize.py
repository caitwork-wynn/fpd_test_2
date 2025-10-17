#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_image_resize.py
이미지 리사이징 테스트 도구

data/learning의 이미지와 labels.txt를 사용하여
원본(높이 112 리사이즈)과 resize_image_with_coordinates 결과(112x112)를
비교하는 이미지를 생성합니다.
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from util.image_resize import resize_image_with_coordinates


def parse_labels_file(labels_path):
    """labels.txt 파일을 파싱하여 데이터를 반환합니다."""
    data = []

    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 헤더 건너뛰기
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        columns = line.split(',')
        if len(columns) < 11:
            continue

        record = {
            'id': columns[0],
            'transparency': columns[1],
            'filename': columns[2],
            'center_x': int(columns[3]),
            'center_y': int(columns[4]),
            'floor_x': int(columns[5]),
            'floor_y': int(columns[6]),
            'front_x': int(columns[7]),
            'front_y': int(columns[8]),
            'side_x': int(columns[9]),
            'side_y': int(columns[10])
        }
        data.append(record)

    return data


def draw_coordinates(image, coords_dict, is_resized=False):
    """
    이미지에 좌표를 표시합니다.

    Args:
        image: 이미지 (numpy array)
        coords_dict: 좌표 딕셔너리 {'center': [x, y], 'floor': [x, y], ...}
        is_resized: 리사이즈된 이미지인지 여부
    """
    # 좌표별 색상 (BGR)
    colors = {
        'center': (0, 0, 255),    # 빨강
        'floor': (0, 255, 0),      # 녹색
        'front': (255, 0, 0),      # 파랑
        'side': (0, 255, 255)      # 노랑
    }

    # 좌표 표시
    for name, coord in coords_dict.items():
        if coord is None:
            continue

        x, y = int(coord[0]), int(coord[1])
        color = colors.get(name, (255, 255, 255))

        # 원 그리기
        cv2.circle(image, (x, y), 3, color, -1)
        cv2.circle(image, (x, y), 5, color, 1)

        # 레이블 표시 (작은 글씨)
        label = name[0].upper()  # 첫 글자만 (C, F, F, S)
        cv2.putText(image, label, (x + 7, y - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def resize_original_image(image, target_height):
    """
    원본 이미지를 높이에 맞춰 리사이즈합니다 (비율 유지).

    Args:
        image: 원본 이미지
        target_height: 목표 높이 (112)

    Returns:
        리사이즈된 이미지
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_height = target_height
    new_width = int(target_height * aspect_ratio)

    resized = cv2.resize(image, (new_width, new_height))
    return resized


def scale_coordinates_for_original(coords_dict, original_size, resized_height):
    """
    원본 이미지의 좌표를 리사이즈된 이미지의 좌표로 변환합니다.

    Args:
        coords_dict: 원본 좌표 딕셔너리
        original_size: 원본 이미지 크기 (height, width)
        resized_height: 리사이즈된 높이

    Returns:
        변환된 좌표 딕셔너리
    """
    orig_h, orig_w = original_size
    scale = resized_height / orig_h

    scaled_coords = {}
    for name, coord in coords_dict.items():
        if coord is None:
            scaled_coords[name] = None
        else:
            scaled_coords[name] = [coord[0] * scale, coord[1] * scale]

    return scaled_coords


def process_single_image(record, learning_path, output_path, target_size=112):
    """
    단일 이미지를 처리하여 비교 이미지를 생성합니다.

    Args:
        record: 레이블 레코드
        learning_path: 학습 데이터 경로
        output_path: 출력 경로
        target_size: 목표 크기 (112)
    """
    image_path = learning_path / record['filename']

    # 이미지가 존재하지 않으면 건너뛰기
    if not image_path.exists():
        return False

    # 원본 이미지 읽기
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        return False

    orig_h, orig_w = original_img.shape[:2]

    # 원본 좌표
    original_coords = {
        'center': [record['center_x'], record['center_y']],
        'floor': [record['floor_x'], record['floor_y']],
        'front': [record['front_x'], record['front_y']],
        'side': [record['side_x'], record['side_y']]
    }

    # === 왼쪽: 원본 이미지 (높이 112로 리사이즈) ===
    left_img = resize_original_image(original_img.copy(), target_size)
    left_coords = scale_coordinates_for_original(
        original_coords, (orig_h, orig_w), target_size
    )
    draw_coordinates(left_img, left_coords, is_resized=True)

    # === 오른쪽: resize_image_with_coordinates 결과 (112x112) ===
    # 좌표를 [x, y] 형식의 리스트로 변환
    labels = [
        [record['center_x'], record['center_y']],
        [record['floor_x'], record['floor_y']],
        [record['front_x'], record['front_y']],
        [record['side_x'], record['side_y']]
    ]

    right_img, adjusted_labels = resize_image_with_coordinates(
        target_size, str(image_path), labels
    )

    # 조정된 좌표를 딕셔너리로 변환
    right_coords = {
        'center': adjusted_labels[0],
        'floor': adjusted_labels[1],
        'front': adjusted_labels[2],
        'side': adjusted_labels[3]
    }
    draw_coordinates(right_img, right_coords, is_resized=True)

    # === 좌우 합성 ===
    # 왼쪽 이미지 너비가 다를 수 있으므로 패딩 추가
    left_h, left_w = left_img.shape[:2]

    # 간격 추가 (10 픽셀)
    gap = 10
    gap_img = np.ones((target_size, gap, 3), dtype=np.uint8) * 255

    # 합성 이미지 생성
    combined = np.hstack([left_img, gap_img, right_img])

    # 제목 추가
    title_height = 30
    title_img = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 255

    # 제목 텍스트
    cv2.putText(title_img, "Original (H=112)", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(title_img, "Resized (112x112)", (left_w + gap + 10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 최종 합성
    final_img = np.vstack([title_img, combined])

    # 파일명 생성
    output_filename = f"test_{record['id'].replace('/', '_')}.png"
    output_file = output_path / output_filename

    # 저장
    cv2.imwrite(str(output_file), final_img)

    return True


def main():
    """메인 실행 함수"""
    print("=== 이미지 리사이징 테스트 도구 ===\n")

    # 경로 설정
    learning_path = ROOT_DIR / "data" / "learning"
    labels_path = learning_path / "labels.txt"
    output_path = ROOT_DIR / "result" / "resize"

    # 출력 폴더 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # labels.txt 파싱
    print("1. labels.txt 파싱 중...")
    if not labels_path.exists():
        print(f"오류: {labels_path}가 존재하지 않습니다.")
        sys.exit(1)

    all_data = parse_labels_file(labels_path)
    print(f"   총 {len(all_data)}개 레코드 발견\n")

    # 100개 랜덤 선택
    sample_size = min(100, len(all_data))
    selected_data = random.sample(all_data, sample_size)
    print(f"2. {sample_size}개 이미지를 랜덤하게 선택\n")

    # 각 이미지 처리
    print(f"3. 이미지 처리 중...")
    success_count = 0
    fail_count = 0

    for record in tqdm(selected_data, desc="   처리 중", unit="img"):
        if process_single_image(record, learning_path, output_path):
            success_count += 1
        else:
            fail_count += 1

    # 결과 출력
    print(f"\n=== 처리 완료 ===")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"\n결과 저장 위치: {output_path}")

    # 범례 출력
    print("\n[좌표 색상 범례]")
    print("  - Center (C): 빨강")
    print("  - Floor (F):  녹색")
    print("  - Front (F):  파랑")
    print("  - Side (S):   노랑")


if __name__ == "__main__":
    main()
