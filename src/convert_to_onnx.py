#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.pth 파일을 ONNX 형식으로 변환하는 스크립트
"""

import torch
import sys
from pathlib import Path
import importlib.util

# 현재 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from util.save_load_model import save_model_as_onnx
import yaml


def convert_pth_to_onnx(pth_path: str):
    """
    .pth 파일을 ONNX로 변환

    Args:
        pth_path: .pth 파일 경로
    """
    pth_path = Path(pth_path)

    if not pth_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 - {pth_path}")
        return False

    print(f"변환 시작: {pth_path}")
    print("="*60)

    # config.yml 로드
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 소스 파일 경로
    model_source = config['learning_model']['source']
    model_path = Path(__file__).parent / model_source

    if not model_path.exists():
        print(f"오류: 모델 파일을 찾을 수 없습니다 - {model_path}")
        return False

    print(f"모델 파일: {model_path}")

    # 모델 모듈 동적 import
    spec = importlib.util.spec_from_file_location("model_module", str(model_path))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # PointDetector 클래스 가져오기
    if not hasattr(model_module, 'PointDetector'):
        print("오류: 모델 파일에 PointDetector 클래스가 없습니다")
        return False

    PointDetector = model_module.PointDetector

    # 모델 설정
    model_config = model_module.get_model_config()
    print(f"모델 설정: {model_config}")

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")

    # 모델 생성
    detector = PointDetector(model_config, device)
    model = detector.model

    # 체크포인트 로드
    print(f"\n체크포인트 로드 중...")
    checkpoint = torch.load(str(pth_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("체크포인트 로드 완료")

    # ONNX 변환
    print(f"\nONNX 변환 시작...")
    print("="*60)
    onnx_path = save_model_as_onnx(model, pth_path, device, print)

    if onnx_path and onnx_path.exists():
        onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print("="*60)
        print(f"✓ 변환 성공!")
        print(f"  ONNX 파일: {onnx_path}")
        print(f"  파일 크기: {onnx_size_mb:.2f}MB")
        return True
    else:
        print("="*60)
        print("✗ 변환 실패")
        return False


if __name__ == "__main__":
    # 기본값: best 모델 변환
    if len(sys.argv) > 1:
        pth_file = sys.argv[1]
    else:
        # config.yml에서 모델명 가져오기
        config_path = Path(__file__).parent / "config.yml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model_source = config['learning_model']['source']
        model_path = Path(__file__).parent / model_source

        # 모델 파일에서 SAVE_FILE_NAME 가져오기
        spec = importlib.util.spec_from_file_location("model_module", str(model_path))
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        model_config = model_module.get_model_config()
        save_file_name = model_config['save_file_name']

        pth_file = f"../model/{save_file_name}_best.pth"

    success = convert_pth_to_onnx(pth_file)
    sys.exit(0 if success else 1)
