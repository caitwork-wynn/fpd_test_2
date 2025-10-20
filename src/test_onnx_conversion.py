#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 변환 테스트 - 120.mpm_mobilenet_lightweight_mh.py 검증

목적:
1. 커스텀 MultiheadAttention이 ONNX 변환 시 _native_multi_head_attention 오류 없이 동작하는지 확인
2. 변환된 ONNX 모델이 PyTorch 모델과 동일한 출력을 생성하는지 검증
3. ONNX Runtime 추론 속도 측정
"""

import sys
import importlib.util
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

# 120 모델 동적 로드
print("=" * 80)
print("ONNX 변환 테스트: 120.mpm_mobilenet_lightweight_mh.py")
print("=" * 80)

model_path = Path(__file__).parent / "model_defs/120.mpm_mobilenet_lightweight_mh.py"
spec = importlib.util.spec_from_file_location("model_120", str(model_path))
model_120 = importlib.util.module_from_spec(spec)
sys.modules["model_120"] = model_120
spec.loader.exec_module(model_120)

# 1. 모델 생성
print("\n[1/7] 모델 생성 중...")
config = {
    'coord_bins': 32,
    'gaussian_sigma': 1.5,
    'embed_dim': 128,
    'num_heads': 2,
    'ffn_expansion': 2,
    'loss_alpha': 0.7
}

model = model_120.MPMMobileNetLightweightModel(config)
model.eval()

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ 총 파라미터: {total_params:,} ({total_params / 1e6:.2f}M)")
print(f"✓ 학습 가능 파라미터: {trainable_params:,}")

# 2. 더미 입력 생성
print("\n[2/7] 더미 입력 생성 중...")
batch_size = 1
dummy_input = torch.randn(batch_size, 1, 96, 96)
print(f"✓ 입력 shape: {dummy_input.shape}")

# 3. PyTorch 추론 (기준)
print("\n[3/7] PyTorch 모델 추론 중...")
with torch.no_grad():
    pytorch_output = model(dummy_input)
    pytorch_coords = pytorch_output['coordinates']

print(f"✓ PyTorch 출력 shape: {pytorch_coords.shape}")
print(f"✓ PyTorch 좌표 (정규화): {pytorch_coords.cpu().numpy()}")

# 4. ONNX 변환
print("\n[4/7] ONNX 변환 중...")
onnx_path = "/tmp/test_120_mobilenet_lightweight_mh.onnx"

try:
    # ONNX 래퍼 사용 (좌표만 반환)
    onnx_wrapper = model_120.MPMMobileNetLightweightModelONNX(model)
    onnx_wrapper.eval()

    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✓ ONNX 변환 성공: {onnx_path}")

    # ONNX 모델 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX 모델 검증 완료")

    # ONNX 그래프 정보
    print(f"✓ ONNX Opset 버전: {onnx_model.opset_import[0].version}")
    print(f"✓ ONNX 그래프 노드 수: {len(onnx_model.graph.node)}")

    # _native_multi_head_attention 연산자 검색
    native_mha_found = False
    for node in onnx_model.graph.node:
        if '_native_multi_head_attention' in node.op_type:
            native_mha_found = True
            print(f"⚠ 경고: _native_multi_head_attention 연산자 발견!")
            break

    if not native_mha_found:
        print("✓ _native_multi_head_attention 연산자 없음 (정상)")

except Exception as e:
    print(f"✗ ONNX 변환 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. ONNX Runtime 추론
print("\n[5/7] ONNX Runtime 추론 중...")
try:
    ort_session = ort.InferenceSession(onnx_path)

    # 입력 준비
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

    # 추론
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_coords = ort_outputs[0]

    print(f"✓ ONNX Runtime 출력 shape: {onnx_coords.shape}")
    print(f"✓ ONNX Runtime 좌표 (정규화): {onnx_coords}")

except Exception as e:
    print(f"✗ ONNX Runtime 추론 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 출력 비교
print("\n[6/7] PyTorch vs ONNX 출력 비교...")
pytorch_coords_np = pytorch_coords.cpu().numpy()
diff = np.abs(pytorch_coords_np - onnx_coords)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"✓ 최대 차이: {max_diff:.6f}")
print(f"✓ 평균 차이: {mean_diff:.6f}")

tolerance = 1e-5
if max_diff < tolerance:
    print(f"✓ 출력 일치 (허용 오차: {tolerance})")
else:
    print(f"⚠ 출력 불일치 (허용 오차 초과: {max_diff} > {tolerance})")
    print("\nPyTorch 좌표:")
    print(pytorch_coords_np)
    print("\nONNX 좌표:")
    print(onnx_coords)
    print("\n차이:")
    print(diff)

# 7. 추론 속도 벤치마크
print("\n[7/7] 추론 속도 벤치마크 (100회 반복)...")

# PyTorch
torch_times = []
for _ in range(100):
    start = time.time()
    with torch.no_grad():
        _ = onnx_wrapper(dummy_input)
    torch_times.append(time.time() - start)

# ONNX Runtime
onnx_times = []
for _ in range(100):
    start = time.time()
    _ = ort_session.run(None, ort_inputs)
    onnx_times.append(time.time() - start)

print(f"✓ PyTorch 평균 추론 시간: {np.mean(torch_times) * 1000:.2f} ms")
print(f"✓ ONNX Runtime 평균 추론 시간: {np.mean(onnx_times) * 1000:.2f} ms")
print(f"✓ 속도 향상: {np.mean(torch_times) / np.mean(onnx_times):.2f}x")

print("\n" + "=" * 80)
print("테스트 완료!")
print("=" * 80)
print(f"\n✓ ONNX 모델이 성공적으로 변환되었으며, _native_multi_head_attention 오류가 발생하지 않습니다.")
print(f"✓ ONNX 모델 저장 위치: {onnx_path}")
print(f"✓ 파일 크기: {Path(onnx_path).stat().st_size / (1024 * 1024):.2f} MB")
