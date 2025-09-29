#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
user.json 데이터 처리 테스트
"""

import os
import shutil
from pathlib import Path
import sys
import json
import cv2
import numpy as np

# 프로젝트 루트 디렉토리 설정 (src의 상위 디렉토리)
ROOT_DIR = Path(__file__).parent.parent

def calculate_floor_coords(detected, bbox, target_size):
    """detected 좌표를 crop된 이미지의 좌표로 변환합니다."""
    floor_x = (detected['x'] - bbox['x']) / bbox['width'] * target_size
    floor_y = (detected['y'] - bbox['y']) / bbox['height'] * target_size
    return floor_x, floor_y

def crop_and_resize_image(image, bbox, target_size):
    """이미지에서 bbox 영역을 crop하고 리사이즈합니다."""
    x = max(0, int(bbox['x']))
    y = max(0, int(bbox['y']))
    w = int(bbox['width'])
    h = int(bbox['height'])

    # 이미지 경계 체크
    img_h, img_w = image.shape[:2]
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # Crop
    cropped = image[y:y2, x:x2]

    # 리사이즈
    if cropped.size > 0:
        resized = cv2.resize(cropped, (target_size, target_size))
        return resized
    return None

def test_single_image():
    """단일 이미지 테스트"""
    # 20250812_lc 폴더의 첫 번째 이미지 테스트
    test_folder = ROOT_DIR / "data" / "base" / "20250812_lc" / "001"
    json_path = test_folder / "20250812-000000-001.jpg.user.json"
    image_path = test_folder / "20250812-000000-001.jpg"

    print(f"테스트 JSON: {json_path}")
    print(f"테스트 이미지: {image_path}")

    if not json_path.exists() or not image_path.exists():
        print("테스트 파일이 없습니다.")
        return

    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print("이미지 로드 실패")
        return

    print(f"원본 이미지 크기: {image.shape}")

    # 테스트 출력 디렉토리 생성
    output_dir = ROOT_DIR / "data" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = 224

    # 첫 번째 객체만 테스트
    if json_data.get('objects'):
        obj = json_data['objects'][0]
        print(f"\n객체 정보:")
        print(f"  - objectId: {obj.get('objectId')}")
        print(f"  - objectType: {obj.get('objectType')}")
        print(f"  - confidence: {obj.get('confidence')}")

        bbox = obj.get('bbox')
        detected = obj.get('detected')

        if bbox and detected:
            print(f"  - bbox: x={bbox['x']:.1f}, y={bbox['y']:.1f}, w={bbox['width']:.1f}, h={bbox['height']:.1f}")
            print(f"  - detected: x={detected['x']:.1f}, y={detected['y']:.1f}")

            # 이미지 crop 및 리사이즈
            cropped_image = crop_and_resize_image(image, bbox, target_size)

            if cropped_image is not None:
                # floor 좌표 계산
                floor_x, floor_y = calculate_floor_coords(detected, bbox, target_size)
                print(f"  - floor (crop 이미지 기준): x={floor_x:.1f}, y={floor_y:.1f}")

                # 결과 이미지 저장
                output_path = output_dir / f"test_obj{obj.get('objectId')}.jpg"
                cv2.imwrite(str(output_path), cropped_image)
                print(f"\nCrop 이미지 저장: {output_path}")

                # floor 위치 표시한 이미지 생성
                marked_image = cropped_image.copy()
                cv2.circle(marked_image, (int(floor_x), int(floor_y)), 5, (0, 0, 255), -1)  # 빨간 점
                cv2.circle(marked_image, (int(floor_x), int(floor_y)), 10, (0, 255, 0), 2)  # 초록 원

                marked_path = output_dir / f"test_obj{obj.get('objectId')}_marked.jpg"
                cv2.imwrite(str(marked_path), marked_image)
                print(f"Floor 표시 이미지 저장: {marked_path}")

                # 원본 이미지에서 bbox와 detected 표시
                original_marked = image.copy()
                # bbox 그리기
                x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                cv2.rectangle(original_marked, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 파란 박스
                # detected 점 그리기
                cv2.circle(original_marked, (int(detected['x']), int(detected['y'])), 10, (0, 0, 255), -1)  # 빨간 점

                original_marked_path = output_dir / f"test_original_marked.jpg"
                cv2.imwrite(str(original_marked_path), original_marked)
                print(f"원본 이미지 표시: {original_marked_path}")

if __name__ == "__main__":
    print("=== user.json 데이터 처리 테스트 ===")
    test_single_image()
    print("\n테스트 완료!")