# -*- coding: utf-8 -*-
"""
MPM Cross Self Attention VPI 모델 테스트
- Feature extraction 검증
- Model forward pass 검증
- ONNX conversion 검증
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path

# 모델 임포트
sys.path.append(str(Path(__file__).parent))
from model_defs.mpm_cross_self_attention_vpi import (
    MPMFeatureExtractor,
    MPMAttentionModel,
    PointDetector,
    get_model_config
)

def test_feature_extraction():
    """특징 추출 테스트"""
    print("=" * 60)
    print("1. Feature Extraction Test")
    print("=" * 60)

    # 인코더 경로 확인
    encoder_path = Path(__file__).parent.parent / "model" / "autoencoder_16x16_best.pth"
    if not encoder_path.exists():
        print(f"⚠️  인코더 파일 없음: {encoder_path}")
        print("   기본 경로로 시도...")
        encoder_path = "../model/autoencoder_16x16_best.pth"

    # Feature extractor 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    extractor = MPMFeatureExtractor(
        encoder_path=str(encoder_path),
        image_size=(64, 64),
        latent_dim=32,
        device=device
    )

    # 더미 이미지 생성 (64x64 RGB)
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # 특징 추출
    print("\n특징 추출 중...")
    features = extractor.extract_features(dummy_image)

    # 결과 검증
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Expected: (202,)")
    print(f"✓ Latent: {features[:32].shape} (should be 32)")
    print(f"✓ ORB: {features[32:].shape} (should be 170 = 5×34)")

    assert features.shape == (202,), f"Expected (202,), got {features.shape}"
    print("\n✅ Feature extraction test passed!")

    return extractor, features


def test_model_forward(features):
    """모델 forward pass 테스트"""
    print("\n" + "=" * 60)
    print("2. Model Forward Pass Test")
    print("=" * 60)

    # 모델 설정 가져오기
    config = get_model_config()

    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPMAttentionModel(config).to(device)
    model.eval()

    print(f"\nModel device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 배치 생성
    batch_features = torch.FloatTensor(features).unsqueeze(0).to(device)
    print(f"\nInput shape: {batch_features.shape}")

    # Forward pass
    print("\nForward pass 중...")
    with torch.no_grad():
        output = model(batch_features)

    # 결과 검증
    print(f"\n✓ Output type: {type(output)}")
    print(f"✓ Keys: {output.keys()}")

    coords = output['coordinates']
    print(f"✓ Coordinates shape: {coords.shape}")
    print(f"✓ Expected: (1, 8)")
    print(f"\nPredicted coordinates:")
    print(f"  Center: ({coords[0, 0]:.2f}, {coords[0, 1]:.2f})")
    print(f"  Floor:  ({coords[0, 2]:.2f}, {coords[0, 3]:.2f})")
    print(f"  Front:  ({coords[0, 4]:.2f}, {coords[0, 5]:.2f})")
    print(f"  Side:   ({coords[0, 6]:.2f}, {coords[0, 7]:.2f})")

    assert coords.shape == (1, 8), f"Expected (1, 8), got {coords.shape}"
    print("\n✅ Model forward pass test passed!")

    return model


def test_onnx_conversion(model):
    """ONNX 변환 테스트"""
    print("\n" + "=" * 60)
    print("3. ONNX Conversion Test")
    print("=" * 60)

    # 모델을 CPU로 이동
    model = model.cpu()
    model.eval()

    # 더미 입력 생성
    dummy_input = torch.randn(1, 202)
    print(f"\nDummy input shape: {dummy_input.shape}")

    # ONNX 변환
    onnx_path = "/tmp/mpm_test.onnx"
    print(f"\nONNX 변환 중... -> {onnx_path}")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['coordinates'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'coordinates': {0: 'batch_size'}
            }
        )

        # 파일 크기 확인
        onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"\n✓ ONNX 파일 생성됨: {onnx_path}")
        print(f"✓ 파일 크기: {onnx_size:.2f} MB")

        print("\n✅ ONNX conversion test passed!")
        return True

    except Exception as e:
        print(f"\n❌ ONNX conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_point_detector():
    """PointDetector 래퍼 테스트"""
    print("\n" + "=" * 60)
    print("4. PointDetector Wrapper Test")
    print("=" * 60)

    # 설정
    config = get_model_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PointDetector 생성
    print("\nPointDetector 생성 중...")
    detector = PointDetector(config, device=device)

    print(f"✓ feature_extractor: {type(detector.feature_extractor).__name__}")
    print(f"✓ model: {type(detector.model).__name__}")
    print(f"✓ device: {detector.device}")

    # 더미 이미지로 예측
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    print("\n예측 수행 중...")
    predictions = detector.predict(dummy_image)

    print(f"\n✓ Predictions shape: {predictions.shape}")
    print(f"✓ Expected: (8,)")
    print(f"\nPredicted coordinates:")
    print(f"  Center: ({predictions[0]:.2f}, {predictions[1]:.2f})")
    print(f"  Floor:  ({predictions[2]:.2f}, {predictions[3]:.2f})")
    print(f"  Front:  ({predictions[4]:.2f}, {predictions[5]:.2f})")
    print(f"  Side:   ({predictions[6]:.2f}, {predictions[7]:.2f})")

    assert predictions.shape == (8,), f"Expected (8,), got {predictions.shape}"
    print("\n✅ PointDetector wrapper test passed!")


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 60)
    print("MPM Cross Self Attention VPI Model Test Suite")
    print("=" * 60)

    try:
        # 1. Feature extraction test
        extractor, features = test_feature_extraction()

        # 2. Model forward pass test
        model = test_model_forward(features)

        # 3. ONNX conversion test
        onnx_success = test_onnx_conversion(model)

        # 4. PointDetector wrapper test
        test_point_detector()

        # 최종 결과
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print("✅ Feature Extraction: PASSED")
        print("✅ Model Forward Pass: PASSED")
        print(f"{'✅' if onnx_success else '❌'} ONNX Conversion: {'PASSED' if onnx_success else 'FAILED'}")
        print("✅ PointDetector Wrapper: PASSED")
        print("\n" + "=" * 60)
        print("🎉 All critical tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
