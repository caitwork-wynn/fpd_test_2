# -*- coding: utf-8 -*-
"""
기존 best 모델의 ONNX를 confidence 포함하여 재생성

사용법:
    cd src
    python regenerate_onnx.py
"""

import torch
import yaml
from pathlib import Path
import importlib.util
import sys

def main():
    print("=" * 60)
    print("ONNX 재생성 스크립트 (Confidence 포함)")
    print("=" * 60)

    # 설정 로드
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 모델 소스 및 파일명
    model_source = config['learning_model']['source']
    base_dir = Path(__file__).parent
    model_path = (base_dir / model_source).resolve()

    print(f"\n모델 파일: {model_path}")

    # 모듈 동적 import
    spec = importlib.util.spec_from_file_location("model_module", str(model_path))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 모델 설정 가져오기
    if hasattr(model_module, 'get_model_config'):
        model_config = model_module.get_model_config()
        save_file_name = model_config['save_file_name']
        target_points = model_config['target_points']
        features_config = model_config['features']
        use_fpd_architecture = model_config['use_fpd_architecture']

        print(f"모델명: {save_file_name}")
        print(f"타겟 포인트: {target_points}")
    else:
        print("오류: get_model_config() 함수를 찾을 수 없습니다.")
        sys.exit(1)

    # 디바이스
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"디바이스: {device}")

    # PointDetector 생성
    PointDetector = model_module.PointDetector
    detector_config = config['learning_model'].copy()
    detector_config['features'] = features_config
    detector_config['training'] = config['training']
    detector_config['target_points'] = target_points
    detector_config['use_fpd_architecture'] = use_fpd_architecture

    print("\n모델 생성 중...")
    detector = PointDetector(detector_config, device)
    model = detector.model

    # Best 모델 가중치 로드
    checkpoint_dir_str = config['learning_model']['checkpointing']['save_dir']
    checkpoint_dir = Path(checkpoint_dir_str)

    # 상대 경로인 경우 base_dir 기준으로 변환
    if not checkpoint_dir.is_absolute():
        # base_dir는 src 폴더이므로, parent가 fpd_test_2가 됨
        project_root = base_dir.parent
        checkpoint_dir = (project_root / checkpoint_dir_str.lstrip('../')).resolve()

    checkpoint_path = checkpoint_dir / f'{save_file_name}_best.pth'

    print(f"체크포인트 디렉토리: {checkpoint_dir}")
    print(f"체크포인트 파일: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"\n오류: Best 모델을 찾을 수 없습니다: {checkpoint_path}")
        print(f"디렉토리 내용 확인:")
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pth"):
                print(f"  - {f.name}")
        sys.exit(1)

    print(f"\nBest 모델 로드 중: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', -1)
        print(f"모델 로드 완료 (epoch: {epoch})")
    else:
        model.load_state_dict(checkpoint)
        print("모델 로드 완료")

    # ONNX 변환
    print("\n" + "=" * 60)
    print("ONNX 변환 시작 (Confidence 포함)")
    print("=" * 60)

    # util 모듈 import
    sys.path.append(str(base_dir))
    from util.save_load_model import save_model_as_onnx

    onnx_path = save_model_as_onnx(
        model,
        checkpoint_path,
        device,
        log_func=print
    )

    print("\n" + "=" * 60)
    if onnx_path and onnx_path.exists():
        onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print("ONNX 재생성 완료!")
        print(f"  파일: {onnx_path}")
        print(f"  크기: {onnx_size_mb:.2f}MB")
        print("\n검증을 위해 다음 명령을 실행하세요:")
        print("  python check_onnx_outputs.py")
    else:
        print("ONNX 생성 실패")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
