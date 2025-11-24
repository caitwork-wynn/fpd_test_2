# -*- coding: utf-8 -*-
"""
PyTorch 모델 vs ONNX 모델 추론 결과 비교
- 동일한 이미지에 대해 두 모델의 출력이 일치하는지 확인
"""

import sys
import torch
import numpy as np
import cv2
import yaml
import importlib.util
from pathlib import Path
import onnxruntime as ort

# 모듈 로드
sys.path.append(str(Path(__file__).parent))
from util.save_load_model import load_model


def preprocess_image_pytorch_style(image_path, image_size=96):
    """
    PyTorch 모델용 전처리 (DataSet과 동일)
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    # BGR → Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, (image_size, image_size))

    # [0, 255] → [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # 정규화
    mean = 0.449
    std = 0.226
    normalized = (normalized - mean) / std

    # Tensor 변환: [H, W] → [1, 1, H, W]
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)

    return tensor


def preprocess_image_onnx_style(image_path, image_size=96):
    """
    ONNX 모델용 전처리 (onnx_inference.py와 동일)
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    # BGR → Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, (image_size, image_size))

    # 정규화 [0, 255] → [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # 학습 시 사용한 정규화 적용
    mean = 0.449
    std = 0.226
    normalized = (normalized - mean) / std

    # 배치 차원 추가: [H, W] → [1, 1, H, W]
    input_tensor = normalized[np.newaxis, np.newaxis, :, :]

    return input_tensor


def denormalize_coords(norm_x, norm_y, coord_ranges):
    """
    정규화된 좌표를 픽셀 좌표로 변환
    """
    x_min, x_max = coord_ranges['x']
    y_min, y_max = coord_ranges['y']

    pred_x = norm_x * (x_max - x_min) + x_min
    pred_y = norm_y * (y_max - y_min) + y_min

    return pred_x, pred_y


def main():
    print("=" * 80)
    print("PyTorch 모델 vs ONNX 모델 추론 비교")
    print("=" * 80)

    # Config 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 모듈 로드
    model_source = config['learning_model']['source']
    model_path = Path(__file__).parent / model_source
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 모델 config 가져오기
    if hasattr(model_module, 'get_model_config'):
        model_config = model_module.get_model_config()
        save_file_name = model_config['save_file_name']
        features_config = model_config['features']
    else:
        # Fallback
        save_file_name = 'mpm_lightweight_floor_optim_96'
        features_config = {'image_size': [96, 96]}

    # 테스트 이미지
    base_dir = Path(__file__).parent.parent
    test_image_path = base_dir / "data" / "learning" / "000141.jpg"
    if not test_image_path.exists():
        print(f"[오류] 테스트 이미지를 찾을 수 없습니다: {test_image_path}")
        return

    print(f"\n[이미지] 테스트 이미지: {test_image_path}")

    # 모델 경로
    checkpoint_dir = base_dir / "model"
    pth_path = checkpoint_dir / f"{save_file_name}_best.pth"
    onnx_path = checkpoint_dir / f"{save_file_name}_best.onnx"

    if not pth_path.exists():
        print(f"[오류] PyTorch 모델을 찾을 수 없습니다: {pth_path}")
        return

    if not onnx_path.exists():
        print(f"[오류] ONNX 모델을 찾을 수 없습니다: {onnx_path}")
        return

    print(f"[확인] PyTorch 모델: {pth_path.name}")
    print(f"[확인] ONNX 모델: {onnx_path.name}")

    # 좌표 범위
    coord_ranges = {
        'x': (-112, 224),
        'y': (0, 224)
    }

    image_size = features_config['image_size'][0]

    # ========================================
    # 1. 전처리 비교
    # ========================================
    print("\n" + "=" * 80)
    print("[1] 전처리 결과 비교")
    print("=" * 80)

    pytorch_tensor = preprocess_image_pytorch_style(test_image_path, image_size)
    onnx_tensor = preprocess_image_onnx_style(test_image_path, image_size)

    # NumPy 배열로 변환하여 비교
    pytorch_array = pytorch_tensor.numpy()
    onnx_array = onnx_tensor

    diff = np.abs(pytorch_array - onnx_array)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"PyTorch 텐서 shape: {pytorch_array.shape}")
    print(f"ONNX 텐서 shape: {onnx_array.shape}")
    print(f"최대 차이: {max_diff:.10f}")
    print(f"평균 차이: {mean_diff:.10f}")

    if max_diff < 1e-6:
        print("[확인] 전처리 결과 일치!")
    else:
        print("[경고] 전처리 결과에 차이가 있습니다!")

    # ========================================
    # 2. PyTorch 모델 추론
    # ========================================
    print("\n" + "=" * 80)
    print("[2] PyTorch 모델 추론")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 모델 생성
    if hasattr(model_module, 'MPMMobileNetLightweightModel'):
        model_class = model_module.MPMMobileNetLightweightModel
        # 모델 config 준비
        model_init_config = {
            'features': features_config,
            'training': config.get('training', {}),
            'target_points': model_config.get('target_points', ['floor']),
            'use_fpd_architecture': model_config.get('use_fpd_architecture', False)
        }
        model = model_class(model_init_config)
    else:
        print("[오류] 모델 클래스를 찾을 수 없습니다")
        return

    model = model.to(device)

    # 체크포인트 로드
    checkpoint = torch.load(str(pth_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 추론
    with torch.no_grad():
        pytorch_input = pytorch_tensor.to(device)
        outputs = model(pytorch_input)

        # 좌표 추출 (정규화된 값 [0, 1])
        pytorch_coords_norm = outputs['coordinates'].cpu().numpy()[0]  # [2]

    # 역정규화
    pytorch_x, pytorch_y = denormalize_coords(
        pytorch_coords_norm[0],
        pytorch_coords_norm[1],
        coord_ranges
    )

    print(f"정규화된 좌표: [{pytorch_coords_norm[0]:.6f}, {pytorch_coords_norm[1]:.6f}]")
    print(f"픽셀 좌표: ({pytorch_x:.2f}, {pytorch_y:.2f})")

    # ========================================
    # 3. ONNX 모델 추론
    # ========================================
    print("\n" + "=" * 80)
    print("[3] ONNX 모델 추론")
    print("=" * 80)

    # ONNX 세션 생성
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    # 추론
    outputs = sess.run(None, {input_name: onnx_tensor})

    # 좌표 추출 (정규화된 값 [0, 1])
    onnx_coords_norm = outputs[0][0]  # [2]

    # 역정규화
    onnx_x, onnx_y = denormalize_coords(
        onnx_coords_norm[0],
        onnx_coords_norm[1],
        coord_ranges
    )

    print(f"정규화된 좌표: [{onnx_coords_norm[0]:.6f}, {onnx_coords_norm[1]:.6f}]")
    print(f"픽셀 좌표: ({onnx_x:.2f}, {onnx_y:.2f})")

    # ========================================
    # 4. 결과 비교
    # ========================================
    print("\n" + "=" * 80)
    print("[4] 결과 비교")
    print("=" * 80)

    # 정규화된 좌표 비교
    coord_diff = np.abs(pytorch_coords_norm - onnx_coords_norm)
    max_coord_diff = np.max(coord_diff)

    print(f"\n정규화된 좌표 차이:")
    print(f"  X: {coord_diff[0]:.10f}")
    print(f"  Y: {coord_diff[1]:.10f}")
    print(f"  최대: {max_coord_diff:.10f}")

    # 픽셀 좌표 비교
    pixel_diff_x = abs(pytorch_x - onnx_x)
    pixel_diff_y = abs(pytorch_y - onnx_y)
    pixel_distance = np.sqrt(pixel_diff_x**2 + pixel_diff_y**2)

    print(f"\n픽셀 좌표 차이:")
    print(f"  PyTorch: ({pytorch_x:.2f}, {pytorch_y:.2f})")
    print(f"  ONNX:    ({onnx_x:.2f}, {onnx_y:.2f})")
    print(f"  차이:    ({pixel_diff_x:.2f}, {pixel_diff_y:.2f})")
    print(f"  거리:    {pixel_distance:.2f} pixels")

    # 결론
    print("\n" + "=" * 80)
    print("[결론]")
    print("=" * 80)

    if max_coord_diff < 1e-4:
        print("[확인] PyTorch와 ONNX 모델의 추론 결과가 일치합니다!")
        print("   -> ONNX 변환에는 문제가 없습니다.")
    else:
        print("[경고] PyTorch와 ONNX 모델의 추론 결과가 다릅니다!")
        print(f"   -> 정규화된 좌표 최대 차이: {max_coord_diff:.10f}")
        print(f"   -> 픽셀 좌표 거리 차이: {pixel_distance:.2f} pixels")
        print("   -> ONNX 변환 과정에 문제가 있을 수 있습니다.")

    # 정답 좌표 확인
    print("\n[정답] 정답 좌표 확인 (labels.txt에서):")
    labels_file = base_dir / "data" / "learning" / "labels.txt"
    if labels_file.exists():
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '000141.jpg' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        label_x = float(parts[3])
                        label_y = float(parts[4])
                        print(f"  정답: ({label_x:.2f}, {label_y:.2f})")

                        # 정답과의 오차
                        pytorch_error = np.sqrt((pytorch_x - label_x)**2 + (pytorch_y - label_y)**2)
                        onnx_error = np.sqrt((onnx_x - label_x)**2 + (onnx_y - label_y)**2)

                        print(f"  PyTorch 오차: {pytorch_error:.2f} pixels")
                        print(f"  ONNX 오차: {onnx_error:.2f} pixels")
                    break

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
