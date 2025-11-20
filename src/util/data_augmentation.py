# -*- coding: utf-8 -*-
"""
데이터 증강 유틸리티
- 유연한 포인트 개수 처리 (1개 ~ N개)
- 다양한 포인트 이름 지원
- 크롭, 회전, 노이즈 등 다양한 증강 기법
"""

import numpy as np
import cv2
import random
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path


class AugmentationConfig:
    """증강 설정 관리 클래스"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 증강 설정 딕셔너리
        """
        self.enabled = config.get('enabled', False)
        self.augment_count = config.get('augment_count', 0)

        # 크롭 설정
        self.crop_config = config.get('crop', {})
        self.crop_enabled = self.crop_config.get('enabled', False)
        self.crop_min_ratio = self.crop_config.get('min_ratio', 0.8)
        self.crop_max_ratio = self.crop_config.get('max_ratio', 1.0)
        self.crop_max_shift = self.crop_config.get('max_shift', 0.15)

        # 회전 설정 (향후 확장)
        self.rotation_config = config.get('rotation', {})
        self.rotation_enabled = self.rotation_config.get('enabled', False)

        # 노이즈 설정 (향후 확장)
        self.noise_config = config.get('noise', {})
        self.noise_enabled = self.noise_config.get('enabled', False)


class FlexibleAugmentation:
    """다양한 포인트 개수를 지원하는 증강 클래스"""

    @staticmethod
    def detect_point_names(coords: Dict[str, float]) -> List[str]:
        """
        좌표 딕셔너리에서 포인트 이름 자동 추출

        Args:
            coords: 좌표 딕셔너리 (예: {'floor_x': 100, 'floor_y': 150, ...})

        Returns:
            포인트 이름 리스트 (예: ['floor', 'center', ...])
        """
        points = set()
        for key in coords.keys():
            if key.endswith('_x') or key.endswith('_y'):
                point_name = key[:-2]  # _x 또는 _y 제거
                points.add(point_name)
        return sorted(list(points))

    @staticmethod
    def count_points(coords: Dict[str, float]) -> int:
        """
        좌표 딕셔너리에서 포인트 개수 계산

        Args:
            coords: 좌표 딕셔너리

        Returns:
            포인트 개수
        """
        return len(FlexibleAugmentation.detect_point_names(coords))

    @staticmethod
    def apply_augmentation(
        image: np.ndarray,
        coords: Dict[str, float],
        aug_type: str,
        target_size: Tuple[int, int],
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        포인트 개수에 관계없이 증강 적용

        Args:
            image: 입력 이미지
            coords: 좌표 딕셔너리
            aug_type: 증강 타입 ('crop', 'flip', 'rotate', 'noise' 등)
            target_size: 목표 이미지 크기
            config: 증강 설정

        Returns:
            증강된 이미지와 변환된 좌표
        """
        if aug_type == 'crop':
            return apply_crop_augmentation(image, coords, target_size, config.get('crop', {}))
        elif aug_type == 'flip':
            # flip은 target_size를 사용하지 않음
            direction = config.get('flip', {}).get('direction', None)
            flipped_image, flipped_coords, _ = apply_flip_augmentation(image, coords, direction)
            return flipped_image, flipped_coords
        elif aug_type == 'rotate':
            # 향후 구현
            raise NotImplementedError("Rotation augmentation not implemented yet")
        elif aug_type == 'noise':
            # 향후 구현
            raise NotImplementedError("Noise augmentation not implemented yet")
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")


def apply_flip_augmentation(
    image: np.ndarray,
    coords: Dict[str, float],
    direction: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, float], Optional[int]]:
    """
    좌우 반전 증강 및 좌표 조정 (유연한 포인트 개수 지원)

    Args:
        image: 원본 이미지 (H, W, C)
        coords: 원본 좌표 딕셔너리 (픽셀 좌표)
                - floor는 필수, center/front/side는 선택적
                - 예: {'floor_x': 484, 'floor_y': 446, 'center_x': 484, 'center_y': 210}
        direction: 128방위 (0~255), 선택적. 0이 북쪽.

    Returns:
        flipped_image: 좌우 반전된 이미지
        flipped_coords: 반전된 좌표 딕셔너리
        flipped_direction: 반전된 방위 (None이면 None 반환)
    """
    h, w = image.shape[:2]

    # 이미지 좌우 반전 (1은 좌우 반전을 의미)
    flipped_image = cv2.flip(image, 1)

    # 모든 좌표를 동적으로 처리
    flipped_coords = {}
    for key, value in coords.items():
        if key.endswith('_x'):  # X 좌표만 반전
            # flipped_x = image_width - original_x
            flipped_coords[key] = w - value
        elif key.endswith('_y'):  # Y 좌표는 그대로
            flipped_coords[key] = value
        else:
            # 좌표가 아닌 다른 메타데이터는 그대로 유지
            flipped_coords[key] = value

    # 방위 반전 (128방위 시스템: 0~255)
    flipped_direction = None
    if direction is not None:
        # 좌우 반전: direction' = 256 - direction (mod 256)
        flipped_direction = (256 - direction) % 256

    return flipped_image, flipped_coords, flipped_direction


def apply_crop_augmentation(
    image: np.ndarray,
    coords: Dict[str, float],
    target_size: Tuple[int, int],
    crop_config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    이미지 크롭 증강 및 좌표 조정 (유연한 포인트 개수 지원)

    Args:
        image: 원본 이미지 (H, W, C)
        coords: 원본 좌표 딕셔너리
                - 1개 포인트: {'floor_x': 100, 'floor_y': 150}
                - 4개 포인트: {'center_x': 50, 'center_y': 60, 'floor_x': 100, ...}
                - 커스텀: {'left_eye_x': 40, 'left_eye_y': 50, ...}
        target_size: 목표 크기 (width, height)
        crop_config: 크롭 설정 {'min_ratio': 0.8, 'max_ratio': 1.0, 'max_shift': 0.15}

    Returns:
        크롭된 이미지와 조정된 좌표
    """
    if crop_config is None:
        crop_config = {}

    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 크롭 파라미터 설정
    min_ratio = crop_config.get('min_ratio', 0.8)
    max_ratio = crop_config.get('max_ratio', 1.0)
    max_shift = crop_config.get('max_shift', 0.15)

    # 랜덤 크롭 크기 결정
    scale = random.uniform(min_ratio, max_ratio)
    crop_h = int(h * scale)
    crop_w = int(w * scale)

    # 랜덤 크롭 위치 결정
    max_shift_x = int(w * max_shift)
    max_shift_y = int(h * max_shift)

    # 크롭 영역이 이미지를 벗어나지 않도록 조정
    x1 = random.randint(0, min(max_shift_x, max(0, w - crop_w)))
    y1 = random.randint(0, min(max_shift_y, max(0, h - crop_h)))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # 이미지 크롭
    cropped_image = image[y1:y2, x1:x2]

    # 목표 크기로 리사이즈
    cropped_image = cv2.resize(cropped_image, target_size)

    # 스케일 계산
    scale_x = target_w / crop_w
    scale_y = target_h / crop_h

    # 모든 좌표를 동적으로 처리
    adjusted_coords = {}
    for key, value in coords.items():
        if key.endswith('_x'):  # X 좌표
            # 크롭 영역 기준으로 조정 후 스케일 적용
            adjusted_coords[key] = (value - x1) * scale_x
        elif key.endswith('_y'):  # Y 좌표
            # 크롭 영역 기준으로 조정 후 스케일 적용
            adjusted_coords[key] = (value - y1) * scale_y
        else:
            # 좌표가 아닌 다른 메타데이터는 그대로 유지
            adjusted_coords[key] = value

    return cropped_image, adjusted_coords


def apply_random_augmentations(
    image: np.ndarray,
    coords: Dict[str, float],
    target_size: Tuple[int, int],
    aug_config: AugmentationConfig,
    count: int = 1
) -> List[Tuple[np.ndarray, Dict[str, float]]]:
    """
    여러 개의 랜덤 증강 적용

    Args:
        image: 원본 이미지
        coords: 원본 좌표
        target_size: 목표 크기
        aug_config: 증강 설정
        count: 생성할 증강 샘플 수

    Returns:
        증강된 (이미지, 좌표) 튜플의 리스트
    """
    augmented_samples = []

    for _ in range(count):
        aug_image = image.copy()
        aug_coords = coords.copy()

        # 크롭 증강 적용
        if aug_config.crop_enabled:
            aug_image, aug_coords = apply_crop_augmentation(
                aug_image, aug_coords, target_size,
                {
                    'min_ratio': aug_config.crop_min_ratio,
                    'max_ratio': aug_config.crop_max_ratio,
                    'max_shift': aug_config.crop_max_shift
                }
            )

        # 다른 증강 기법들 (향후 추가)
        # if aug_config.rotation_enabled:
        #     aug_image, aug_coords = apply_rotation_augmentation(...)

        augmented_samples.append((aug_image, aug_coords))

    return augmented_samples


def validate_coordinates(coords: Dict[str, float]) -> bool:
    """
    좌표 딕셔너리 유효성 검사

    Args:
        coords: 좌표 딕셔너리

    Returns:
        유효하면 True, 아니면 False
    """
    point_names = FlexibleAugmentation.detect_point_names(coords)

    # 각 포인트가 x, y 좌표를 모두 가지고 있는지 확인
    for point in point_names:
        if f"{point}_x" not in coords or f"{point}_y" not in coords:
            return False

    return len(point_names) > 0


# 테스트 코드 (개발 시 사용)
if __name__ == "__main__":
    # 테스트용 더미 데이터
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # 1개 포인트 테스트
    coords_1 = {'floor_x': 112.0, 'floor_y': 150.0}
    aug_img, aug_coords = apply_crop_augmentation(test_image, coords_1, (112, 112))
    print(f"1 point test: {FlexibleAugmentation.detect_point_names(coords_1)}")

    # 4개 포인트 테스트
    coords_4 = {
        'center_x': 50.0, 'center_y': 60.0,
        'floor_x': 112.0, 'floor_y': 150.0,
        'front_x': 80.0, 'front_y': 90.0,
        'side_x': 120.0, 'side_y': 110.0
    }
    aug_img, aug_coords = apply_crop_augmentation(test_image, coords_4, (112, 112))
    print(f"4 points test: {FlexibleAugmentation.detect_point_names(coords_4)}")

    # 커스텀 포인트 테스트
    coords_custom = {
        'left_eye_x': 40.0, 'left_eye_y': 50.0,
        'right_eye_x': 70.0, 'right_eye_y': 52.0,
        'nose_x': 55.0, 'nose_y': 70.0
    }
    aug_img, aug_coords = apply_crop_augmentation(test_image, coords_custom, (112, 112))
    print(f"Custom points test: {FlexibleAugmentation.detect_point_names(coords_custom)}")