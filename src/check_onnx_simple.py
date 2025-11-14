# -*- coding: utf-8 -*-
"""ONNX 모델의 입출력 확인"""

import onnx
from pathlib import Path

model_path = Path(__file__).parent.parent / "model" / "mpm_lightweight_floor_optim_96_best.onnx"
model = onnx.load(str(model_path))

print("Inputs:")
for inp in model.graph.input:
    print(f"  - {inp.name}")

print("\nOutputs:")
for out in model.graph.output:
    print(f"  - {out.name}")

print(f"\nTotal outputs: {len(model.graph.output)}")
