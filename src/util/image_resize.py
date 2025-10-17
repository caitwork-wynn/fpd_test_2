# -*- coding: utf-8 -*-
"""
이미지 리사이징 및 좌표 변환 유틸리티
- 비율 유지하면서 target_size로 리사이징
- 검은색 배경 패딩 추가
- 좌표 자동 변환
"""

import cv2
import numpy as np
from typing import List, Union, Tuple


def resize_image_with_coordinates(target_size: int,
                                image_path: str,
                                labels: List[Union[int, List[int]]]) -> Tuple[np.ndarray, List]:
    """
    이미지를 target_size x target_size로 리사이징하고 좌표를 조정하는 함수

    Args:
        target_size: 목표 크기 (64면 64x64, 112면 112x112)
        image_path: 원본 이미지 파일 경로
        labels: 좌표 데이터 ([floor] 또는 [center, floor, front, side] 좌표 배열)

    Returns:
        tuple: (리사이징된 이미지, 조정된 좌표 배열)

    Examples:
        >>> # 1 point 케이스
        >>> img, coords = resize_image_with_coordinates(64, "image.png", [[100, 150]])
        >>>
        >>> # 4 points 케이스
        >>> labels = [[50, 60], [100, 150], [80, 90], [120, 110]]
        >>> img, coords = resize_image_with_coordinates(112, "image.png", labels)
        >>>
        >>> # 단일 값 포함 케이스
        >>> labels = [5, [100, 150]]  # floor ID와 좌표
        >>> img, coords = resize_image_with_coordinates(64, "image.png", labels)
    """

    # 1. 원본 이미지 읽기
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    original_height, original_width = original_img.shape[:2]

    # 2. 배경 이미지 생성 (검은색)
    background = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 3. 비율 계산 및 리사이징
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:  # 가로가 더 긴 이미지
        # 가로를 target_size에 맞추고, 세로는 비율 유지
        new_width = target_size
        new_height = int(target_size / aspect_ratio)

        # 리사이징
        resized_img = cv2.resize(original_img, (new_width, new_height))

        # 아래쪽에 패딩이 들어가도록 배치 (위쪽 정렬)
        y_offset = 0
        x_offset = 0

        # 좌표 변환 계산
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        offset_x = x_offset
        offset_y = y_offset

    else:  # 세로가 더 긴 이미지 (정사각형 포함)
        # 세로를 target_size에 맞추고, 가로는 비율 유지
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

        # 리사이징
        resized_img = cv2.resize(original_img, (new_width, new_height))

        # 좌우에 패딩이 들어가도록 배치 (중앙 정렬)
        x_offset = (target_size - new_width) // 2
        y_offset = 0

        # 좌표 변환 계산
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        offset_x = x_offset
        offset_y = y_offset

    # 4. 배경 이미지에 리사이징된 이미지 복사
    background[y_offset:y_offset+new_height,
               x_offset:x_offset+new_width] = resized_img

    # 5. 좌표 조정
    adjusted_labels = []
    for label in labels:
        if isinstance(label, list) and len(label) >= 2:
            # [x, y] 좌표인 경우
            adjusted_x = label[0] * scale_x + offset_x
            adjusted_y = label[1] * scale_y + offset_y

            if len(label) == 2:
                adjusted_labels.append([adjusted_x, adjusted_y])
            else:
                # 추가 정보가 있는 경우 그대로 유지
                adjusted_label = [adjusted_x, adjusted_y] + label[2:]
                adjusted_labels.append(adjusted_label)
        else:
            # 단일 값인 경우 (floor 등)
            adjusted_labels.append(label)

    return background, adjusted_labels


# 테스트 코드
if __name__ == "__main__":
    import sys

    # 간단한 테스트 이미지 생성
    test_img_path = "/tmp/test_image.png"

    # 300x200 테스트 이미지 생성 (가로가 긴 경우)
    test_img_wide = np.ones((200, 300, 3), dtype=np.uint8) * 128
    cv2.imwrite(test_img_path, test_img_wide)

    # 테스트 1: 가로가 긴 이미지 (300x200 -> 64x64)
    print("=== 테스트 1: 가로가 긴 이미지 ===")
    labels_1 = [[150, 100]]  # 중앙 좌표
    result_img_1, result_labels_1 = resize_image_with_coordinates(64, test_img_path, labels_1)
    print(f"원본 이미지: 300x200")
    print(f"결과 이미지: {result_img_1.shape}")
    print(f"원본 좌표: {labels_1}")
    print(f"변환 좌표: {result_labels_1}")
    print(f"예상: 가로 맞춤(64), 세로 비율유지(약 43), 위쪽 정렬")
    print(f"예상 좌표: [32.0, 21.33] 정도\n")

    # 200x300 테스트 이미지 생성 (세로가 긴 경우)
    test_img_tall = np.ones((300, 200, 3), dtype=np.uint8) * 128
    cv2.imwrite(test_img_path, test_img_tall)

    # 테스트 2: 세로가 긴 이미지 (200x300 -> 64x64)
    print("=== 테스트 2: 세로가 긴 이미지 ===")
    labels_2 = [[100, 150]]  # 중앙 좌표
    result_img_2, result_labels_2 = resize_image_with_coordinates(64, test_img_path, labels_2)
    print(f"원본 이미지: 200x300")
    print(f"결과 이미지: {result_img_2.shape}")
    print(f"원본 좌표: {labels_2}")
    print(f"변환 좌표: {result_labels_2}")
    print(f"예상: 세로 맞춤(64), 가로 비율유지(약 43), 중앙 정렬")
    print(f"예상 좌표: [32.0, 32.0] 정도\n")

    # 테스트 3: 여러 좌표
    print("=== 테스트 3: 여러 좌표 (4 points) ===")
    labels_3 = [[100, 150], [100, 200], [50, 150], [150, 160]]
    result_img_3, result_labels_3 = resize_image_with_coordinates(64, test_img_path, labels_3)
    print(f"원본 좌표: {labels_3}")
    print(f"변환 좌표: {result_labels_3}\n")

    # 테스트 4: 혼합 타입
    print("=== 테스트 4: 혼합 타입 (단일 값 + 좌표) ===")
    labels_4 = [5, [100, 150]]
    result_img_4, result_labels_4 = resize_image_with_coordinates(64, test_img_path, labels_4)
    print(f"원본 레이블: {labels_4}")
    print(f"변환 레이블: {result_labels_4}")
    print("단일 값(5)은 그대로 유지, 좌표만 변환됨\n")

    print("모든 테스트 완료!")
