# -*- coding: utf-8 -*-
"""ONNX 모델의 입출력 확인"""

import onnx
import sys
from pathlib import Path

# ONNX 모델 경로
model_path = Path(__file__).parent.parent / "model" / "mpm_lightweight_floor_optim_96_best.onnx"

print("=" * 60)
print("ONNX 모델 입출력 분석")
print("=" * 60)
print(f"모델 경로: {model_path}\n")

# ONNX 모델 로드
model = onnx.load(str(model_path))

# 입력 정보
print("=" * 60)
print("입력 (Inputs)")
print("=" * 60)
for input_tensor in model.graph.input:
    print(f"이름: {input_tensor.name}")
    print(f"타입: {input_tensor.type.tensor_type.elem_type}")
    shape = input_tensor.type.tensor_type.shape
    dims = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})"
            for dim in shape.dim]
    print(f"Shape: {dims}")
    print()

# 출력 정보
print("=" * 60)
print("출력 (Outputs)")
print("=" * 60)
for output_tensor in model.graph.output:
    print(f"이름: {output_tensor.name}")
    print(f"타입: {output_tensor.type.tensor_type.elem_type}")
    shape = output_tensor.type.tensor_type.shape
    dims = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})"
            for dim in shape.dim]
    print(f"Shape: {dims}")
    print()

print("=" * 60)
print("결론")
print("=" * 60)

output_count = len(model.graph.output)
if output_count == 1:
    print("[X] 좌표만 출력")
elif output_count == 2:
    print("[OK] 좌표 + confidence 출력")
elif output_count == 3:
    print("[OK] 좌표 + confidence + entropy 출력 (완전)")
else:
    print(f"[!] 예상치 못한 출력 개수: {output_count}개")

print("=" * 60)
