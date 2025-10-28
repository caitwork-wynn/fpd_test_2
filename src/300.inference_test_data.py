# -*- coding: utf-8 -*-
"""
테스트 데이터 추론 스크립트
- data/test 폴더의 이미지와 JSON 파일 읽기
- JSON에서 object bbox로 이미지 crop
- 학습된 모델로 4개 좌표 추론
- 추론 좌표를 이미지에 표시하여 result/test 폴더에 저장
"""

import os
import sys
import yaml
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List
import importlib.util
import warnings

# 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
warnings.filterwarnings("ignore", category=UserWarning, module="kornia")

# sys.path에 현재 디렉토리 추가
sys.path.append(str(Path(__file__).parent))


def load_model_and_detector(config_path: Path, model_weight_path: Path, device):
    """모델 및 검출기 로드"""
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 소스 가져오기
    model_source = config['learning_model']['source']

    # 모델 정의 파일 동적 import
    base_dir = config_path.parent
    model_path = (base_dir / model_source).resolve()

    spec = importlib.util.spec_from_file_location("model_module", str(model_path))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 모델 설정 가져오기
    if hasattr(model_module, 'get_model_config'):
        model_config = model_module.get_model_config()
        target_points = model_config.get('target_points', ['center', 'floor', 'front', 'side'])
        use_fpd_architecture = model_config.get('use_fpd_architecture', False)
        features_config = model_config.get('features', {'image_size': [96, 96], 'grid_size': 7})
    else:
        target_points = config.get('learning_model', {}).get('target_points', ['center', 'floor', 'front', 'side'])
        use_fpd_architecture = config.get('learning_model', {}).get('architecture', {}).get('use_fpd_architecture', False)
        features_config = config.get('learning_model', {}).get('architecture', {}).get('features', {'image_size': [96, 96], 'grid_size': 7})

    # PointDetector 및 PointDetectorDataSet 클래스 가져오기
    PointDetector = model_module.PointDetector
    PointDetectorDataSet = model_module.PointDetectorDataSet

    # 검출기 생성
    detector_config = config['learning_model'].copy()
    detector_config['features'] = features_config
    detector_config['training'] = config['training']
    detector_config['target_points'] = target_points
    detector_config['use_fpd_architecture'] = use_fpd_architecture

    detector = PointDetector(detector_config, device)
    model = detector.model

    # 모델 가중치 로드
    checkpoint = torch.load(str(model_weight_path), map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # feature_mean, feature_std 복원 (있는 경우)
    if 'feature_mean' in checkpoint:
        detector.feature_mean = checkpoint['feature_mean']
        detector.feature_std = checkpoint['feature_std']

    model.eval()

    return model, detector, target_points, features_config


def preprocess_image(img_crop, target_size):
    """이미지 전처리 (모델 입력 형식으로 변환)"""
    # 리사이즈
    img_resized = cv2.resize(img_crop, tuple(target_size))

    # Grayscale로 변환
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 정규화 [0, 255] -> [0, 1]
    img_normalized = img_gray.astype(np.float32) / 255.0

    # 채널 차원 추가 (H, W) -> (1, H, W)
    img_with_channel = np.expand_dims(img_normalized, axis=0)

    # 배치 차원 추가 (1, H, W) -> (1, 1, H, W)
    img_tensor = torch.from_numpy(img_with_channel).unsqueeze(0)

    return img_tensor


def denormalize_coordinates(normalized_coords, coord_min_x=-112, coord_max_x=224, coord_min_y=0, coord_max_y=224):
    """
    정규화된 좌표를 픽셀 좌표로 역정규화

    모델은 112x112 픽셀 기준의 확장된 좌표 범위를 사용:
    - X: -112 ~ 224 (음수 허용, 이미지 크기의 2배까지)
    - Y: 0 ~ 224 (이미지 크기의 2배까지)
    """
    pixel_coords = []
    for i in range(0, len(normalized_coords), 2):
        x_norm = normalized_coords[i]
        y_norm = normalized_coords[i + 1]

        x_pixel = x_norm * (coord_max_x - coord_min_x) + coord_min_x
        y_pixel = y_norm * (coord_max_y - coord_min_y) + coord_min_y

        pixel_coords.extend([x_pixel, y_pixel])

    return pixel_coords


def draw_points_on_image(img, points_dict, bbox=None):
    """이미지에 포인트 그리기"""
    img_copy = img.copy()

    # 포인트별 색상 정의 (BGR)
    point_colors = {
        'center': (0, 255, 0),      # 초록색
        'floor': (0, 255, 255),     # 노란색
        'front': (255, 0, 255),     # 자홍색
        'side': (255, 255, 0)       # 청록색
    }

    # bbox 그리기 (선택사항)
    if bbox:
        left, top, width, height = bbox['left'], bbox['top'], bbox['width'], bbox['height']
        cv2.rectangle(img_copy, (int(left), int(top)), (int(left + width), int(top + height)), (255, 0, 0), 2)

    # 포인트 그리기
    for point_name, (x, y) in points_dict.items():
        color = point_colors.get(point_name, (128, 128, 128))

        # 외각 원 (큰 원, 테두리만)
        cv2.circle(img_copy, (int(x), int(y)), 8, color, 2)
        # 내부 원 (작은 원, 채움)
        cv2.circle(img_copy, (int(x), int(y)), 5, color, -1)

        # 포인트 이름 텍스트
        cv2.putText(img_copy, point_name, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img_copy


def add_legend(img, target_points):
    """이미지에 색상 범례 추가"""
    img_copy = img.copy()

    # 포인트별 색상 정의 (BGR)
    point_colors = {
        'center': (0, 255, 0),      # 초록색
        'floor': (0, 255, 255),     # 노란색
        'front': (255, 0, 255),     # 자홍색
        'side': (255, 255, 0)       # 청록색
    }

    # 범례 위치 및 크기 설정
    legend_x = 10
    legend_y = 10
    legend_width = 180
    legend_height = 30 + len(target_points) * 30

    # 반투명 배경 그리기
    overlay = img_copy.copy()
    cv2.rectangle(overlay, (legend_x, legend_y),
                  (legend_x + legend_width, legend_y + legend_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_copy, 0.3, 0, img_copy)

    # 테두리 그리기
    cv2.rectangle(img_copy, (legend_x, legend_y),
                  (legend_x + legend_width, legend_y + legend_height),
                  (255, 255, 255), 2)

    # 제목
    cv2.putText(img_copy, "Point Legend",
                (legend_x + 10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 각 포인트 설명
    for idx, point_name in enumerate(target_points):
        y_pos = legend_y + 50 + idx * 30
        color = point_colors.get(point_name, (128, 128, 128))

        # 색상 원 그리기
        cv2.circle(img_copy, (legend_x + 20, y_pos), 8, color, 2)
        cv2.circle(img_copy, (legend_x + 20, y_pos), 5, color, -1)

        # 포인트 이름
        cv2.putText(img_copy, point_name.capitalize(),
                    (legend_x + 40, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img_copy


def process_test_data(test_data_dir: Path, result_dir: Path, model, detector, target_points, features_config, device):
    """테스트 데이터 처리"""
    # result/test 폴더 생성
    result_test_dir = result_dir / 'test'
    result_test_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 크기
    image_size = features_config['image_size']

    # JSON 파일 목록
    json_files = list(test_data_dir.glob('*.json'))

    print(f"\n총 {len(json_files)}개의 JSON 파일 발견")
    print("=" * 60)

    for json_file in json_files:
        # JSON 로드
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 파일 경로
        image_file = json_file.with_suffix('.jpg')

        if not image_file.exists():
            print(f"이미지 파일 없음: {image_file}")
            continue

        # 이미지 로드
        img_original = cv2.imread(str(image_file))

        if img_original is None:
            print(f"이미지 로드 실패: {image_file}")
            continue

        print(f"\n처리 중: {image_file.name}")
        print(f"  객체 수: {len(data['objects'])}")

        # 결과 이미지 복사 (모든 객체를 하나의 이미지에 그리기)
        img_result = img_original.copy()

        # 각 객체 처리
        for obj in data['objects']:
            object_id = obj['object_id']
            bbox = obj['bbox']

            # bbox 좌표
            left = int(bbox['left'])
            top = int(bbox['top'])
            width = int(bbox['width'])
            height = int(bbox['height'])

            # 이미지 crop
            img_crop = img_original[top:top+height, left:left+width]

            if img_crop.size == 0:
                print(f"  객체 {object_id}: Crop 실패 (빈 영역)")
                continue

            # 전처리
            img_tensor = preprocess_image(img_crop, image_size)
            img_tensor = img_tensor.to(device)

            # 추론
            with torch.no_grad():
                outputs = model(img_tensor)

            # 좌표 추출
            outputs_coords = outputs['coordinates'].cpu().numpy()[0]  # [N]

            # 역정규화 (112x112 픽셀 기준 좌표로 변환)
            # 모델의 학습 데이터는 112x112 픽셀 기준으로 정규화됨
            pixel_coords_112 = denormalize_coordinates(outputs_coords)

            # 포인트 딕셔너리 생성 (원본 이미지 기준 좌표)
            points_dict = {}
            for idx, point_name in enumerate(target_points):
                x_112 = pixel_coords_112[idx * 2]
                y_112 = pixel_coords_112[idx * 2 + 1]

                # 112x112 기준 좌표를 crop 이미지 크기로 스케일링
                # 112x112 -> width x height
                x_crop = x_112 * (width / 112.0)
                y_crop = y_112 * (height / 112.0)

                # 원본 이미지 기준 좌표로 변환
                x_original = x_crop + left
                y_original = y_crop + top

                points_dict[point_name] = (x_original, y_original)

            # 결과 이미지에 포인트 그리기 (누적)
            img_result = draw_points_on_image(img_result, points_dict, bbox)

            print(f"  객체 {object_id}: 추론 완료")

            # 좌표 출력
            for point_name, (x, y) in points_dict.items():
                print(f"    {point_name}: ({x:.1f}, {y:.1f})")

        # 범례 추가
        img_result = add_legend(img_result, target_points)

        # 결과 저장 (하나의 이미지에 모든 객체)
        result_filename = f"{image_file.stem}_result.jpg"
        result_path = result_test_dir / result_filename
        cv2.imwrite(str(result_path), img_result)

        print(f"\n  최종 결과 저장: {result_filename}")

    print("\n" + "=" * 60)
    print(f"모든 처리 완료! 결과 저장 위치: {result_test_dir}")
    print("=" * 60)


def main():
    """메인 함수"""
    print("=" * 60)
    print("테스트 데이터 추론 스크립트")
    print("=" * 60)

    # 경로 설정
    base_dir = Path(__file__).parent
    config_path = base_dir / 'config.yml'
    test_data_dir = base_dir.parent / 'data' / 'test'
    result_dir = base_dir.parent / 'result'
    model_weight_path = base_dir.parent / 'model' / 'mpm_lightweight_optim_96_best.pth'

    # 디바이스 설정
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        print(f"\nGPU 사용: {torch.cuda.get_device_name(device)}")
    else:
        print("\nCPU 사용")

    # 모델 로드
    print("\n모델 로딩 중...")
    model, detector, target_points, features_config = load_model_and_detector(
        config_path, model_weight_path, device
    )
    print(f"모델 로드 완료")
    print(f"타겟 포인트: {', '.join(target_points)}")
    print(f"이미지 크기: {features_config['image_size']}")

    # 테스트 데이터 처리
    process_test_data(test_data_dir, result_dir, model, detector, target_points, features_config, device)


if __name__ == "__main__":
    main()
