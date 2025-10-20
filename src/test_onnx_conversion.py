#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 변환 테스트
"""

import sys
from pathlib import Path
import yaml
import importlib.util
import torch

# 설정 로드
config_path = Path(__file__).parent / 'config.yml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 모델 로드
model_source = config['learning_model']['source']
model_path = (Path(__file__).parent / model_source).resolve()

print("=" * 60)
print("ONNX 변환 테스트")
print("=" * 60)
print(f"\n모델 소스: {model_source}")

# 모듈 동적 import
spec = importlib.util.spec_from_file_location("model_module", str(model_path))
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# PointDetector import
PointDetector = model_module.PointDetector

print("모델 클래스 로드 완료\n")

# 모델 설정
model_config = model_module.get_model_config()
save_file_name = model_config['save_file_name']
target_points = model_config['target_points']
features_config = model_config['features']

print(f"모델 설정: {save_file_name}")
print(f"이미지 크기: {features_config['image_size']}")
print(f"타겟 포인트: {target_points}\n")

# 디바이스 설정
device = torch.device("cpu")  # ONNX 변환은 CPU에서 수행

# Detector 생성
detector_config = config['learning_model'].copy()
detector_config['features'] = features_config
detector_config['training'] = config['training']
detector_config['target_points'] = target_points
detector = PointDetector(detector_config, device)

print(f"모델 파라미터 수: {sum(p.numel() for p in detector.model.parameters()):,}\n")

# get_input_dim() 확인
if hasattr(model_module, 'get_input_dim'):
    input_dim = model_module.get_input_dim()
    print(f"get_input_dim() 반환값: {input_dim}")
    print(f"✓ Grayscale 1채널: {'PASS' if input_dim[0] == 1 else 'FAIL'}")
    print(f"✓ 96×96 크기: {'PASS' if input_dim[1] == 96 and input_dim[2] == 96 else 'FAIL'}\n")

# ONNX 래퍼 확인
if hasattr(model_module, 'MPMMobileNetLightweightModelONNX'):
    print("✓ MPMMobileNetLightweightModelONNX 래퍼 발견\n")

    # 래퍼 테스트
    onnx_wrapper = model_module.MPMMobileNetLightweightModelONNX(detector.model)
    test_input = torch.randn(1, 1, 96, 96)

    print("래퍼 forward 테스트:")
    print(f"입력 shape: {test_input.shape}")

    onnx_wrapper.eval()
    with torch.no_grad():
        output = onnx_wrapper(test_input)

    print(f"출력 shape: {output.shape}")
    print(f"출력 타입: {type(output)}")
    print(f"✓ 래퍼 테스트 성공!\n")
else:
    print("❌ MPMMobileNetLightweightModelONNX 래퍼를 찾을 수 없음\n")

# ONNX 변환 시뮬레이션
print("=" * 60)
print("ONNX 변환 시뮬레이션")
print("=" * 60)

# save_load_model.py import
sys.path.append(str(Path(__file__).parent))
from util.save_load_model import save_model_as_onnx

# 임시 체크포인트 경로
checkpoint_path = Path("/tmp/test_mobilenet_onnx.pth")

print(f"\nONNX 변환 시작...")
print(f"체크포인트 경로: {checkpoint_path}")
print(f"디바이스: {device}\n")

onnx_path = save_model_as_onnx(detector.model, checkpoint_path, device, print)

if onnx_path and onnx_path.exists():
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ ONNX 변환 성공!")
    print(f"ONNX 파일: {onnx_path}")
    print(f"파일 크기: {onnx_size_mb:.2f}MB")
else:
    print(f"\n❌ ONNX 변환 실패!")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
