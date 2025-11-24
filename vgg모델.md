src/model_def/200.floor_only_vgg.py를 131.mpm_mobilenet_lightweight_floor_optim.py의 정보에서 인공지능 모델을 다음과 같이 바꾸어서 만들어줘.
# VGG 기반 Floor 좌표 예측 모델 아키텍처

## 개요

96×96 Grayscale 이미지에서 Floor 좌표(x, y)를 Grid Classification 방식으로 예측하는 경량 모델

---

## 입출력 스펙

| 구분 | 형태 | 설명 |
|------|------|------|
| **입력** | `[B, 1, 96, 96]` | Grayscale 이미지 |
| **출력 - 좌표** | `[B, 2]` | 정규화된 floor_x, floor_y (0~1) |
| **출력 - Confidence** | `[B, 2]` | 각 좌표의 max probability |
| **출력 - Entropy** | `[B, 2]` | 각 좌표의 불확실성 지표 |

---

## 모델 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│                   [B, 1, 96, 96]                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 1: VGG BACKBONE                      │
│              (Feature Extraction)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Block 1: Conv Block                                  │   │
│  │   Conv2d(1→64, 3×3, pad=1) → BN → ReLU              │   │
│  │   Conv2d(64→64, 3×3, pad=1) → BN → ReLU             │   │
│  │   MaxPool2d(2×2, stride=2)                          │   │
│  │   Output: [B, 64, 48, 48]                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Block 2: Conv Block                                  │   │
│  │   Conv2d(64→128, 3×3, pad=1) → BN → ReLU            │   │
│  │   Conv2d(128→128, 3×3, pad=1) → BN → ReLU           │   │
│  │   MaxPool2d(2×2, stride=2)                          │   │
│  │   Output: [B, 128, 24, 24]                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Block 3: Conv Block                                  │   │
│  │   Conv2d(128→256, 3×3, pad=1) → BN → ReLU           │   │
│  │   Conv2d(256→256, 3×3, pad=1) → BN → ReLU           │   │
│  │   MaxPool2d(2×2, stride=2)                          │   │
│  │   Output: [B, 256, 12, 12]                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Block 4: Conv Block                                  │   │
│  │   Conv2d(256→512, 3×3, pad=1) → BN → ReLU           │   │
│  │   Conv2d(512→512, 3×3, pad=1) → BN → ReLU           │   │
│  │   MaxPool2d(2×2, stride=2)                          │   │
│  │   Output: [B, 512, 6, 6]                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Block 5: Conv Block                                  │   │
│  │   Conv2d(512→512, 3×3, pad=1) → BN → ReLU           │   │
│  │   Conv2d(512→512, 3×3, pad=1) → BN → ReLU           │   │
│  │   MaxPool2d(2×2, stride=2)                          │   │
│  │   Output: [B, 512, 3, 3]                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 2: GLOBAL POOLING                    │
├─────────────────────────────────────────────────────────────┤
│   AdaptiveAvgPool2d(1)                                      │
│   Flatten                                                   │
│   Output: [B, 512]                                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 3: FC HIDDEN LAYERS                      │
│                  (2 Hidden Layers)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Hidden Layer 1                                       │   │
│  │   Linear(512 → 256)                                  │   │
│  │   ReLU                                               │   │
│  │   Dropout(0.3)                                       │   │
│  │   Output: [B, 256]                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Hidden Layer 2                                       │   │
│  │   Linear(256 → 128)                                  │   │
│  │   ReLU                                               │   │
│  │   Dropout(0.2)                                       │   │
│  │   Output: [B, 128]                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│          STAGE 4: GRID CLASSIFICATION HEADS                 │
│              (X 좌표 / Y 좌표 분리)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│      ┌──────────────────┐    ┌──────────────────┐          │
│      │   X Coord Head   │    │   Y Coord Head   │          │
│      ├──────────────────┤    ├──────────────────┤          │
│      │ Linear(128→64)   │    │ Linear(128→64)   │          │
│      │ ReLU             │    │ ReLU             │          │
│      │ Linear(64→32)    │    │ Linear(64→32)    │          │
│      │ (32 bins)        │    │ (32 bins)        │          │
│      └────────┬─────────┘    └────────┬─────────┘          │
│               │                       │                     │
│               ▼                       ▼                     │
│        logits_x [B,32]         logits_y [B,32]             │
│               │                       │                     │
│               ▼                       ▼                     │
│          Softmax                  Softmax                   │
│               │                       │                     │
│               ▼                       ▼                     │
│         probs_x [B,32]          probs_y [B,32]             │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 5: OUTPUT COMPUTATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Coordinate Computation (Soft Argmax)                 │   │
│  │   bin_centers = linspace(coord_min, coord_max, 32)   │   │
│  │   coord = Σ(probs × bin_centers)                     │   │
│  │   Output: floor_x, floor_y                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Confidence Computation                               │   │
│  │   confidence = max(probs, dim=-1)                    │   │
│  │   Output: conf_x, conf_y  (0~1, 높을수록 확실)        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Entropy Computation                                  │   │
│  │   entropy = -Σ(probs × log(probs))                   │   │
│  │   Output: ent_x, ent_y  (낮을수록 확실)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 레이어별 상세 스펙

### VGG Backbone (Grayscale 수정)

| Block | 입력 채널 | 출력 채널 | 출력 크기 | Conv 수 |
|-------|----------|----------|----------|---------|
| Block 1 | 1 | 64 | 48×48 | 2 |
| Block 2 | 64 | 128 | 24×24 | 2 |
| Block 3 | 128 | 256 | 12×12 | 2 |
| Block 4 | 256 | 512 | 6×6 | 2 |
| Block 5 | 512 | 512 | 3×3 | 2 |

### FC Hidden Layers

| Layer | 입력 | 출력 | Activation | Dropout |
|-------|------|------|------------|---------|
| Hidden 1 | 512 | 256 | ReLU | 0.3 |
| Hidden 2 | 256 | 128 | ReLU | 0.2 |

### Classification Heads

| Head | 입력 | Hidden | 출력 (bins) | 좌표 범위 |
|------|------|--------|-------------|----------|
| X Coord | 128 | 64 | 32 | [-112, 224] |
| Y Coord | 128 | 64 | 32 | [0, 224] |

---

## Grid Classification 방식

### 개념

좌표를 직접 회귀(regression)하는 대신, 좌표 범위를 N개의 bin으로 나누어 분류(classification) 문제로 변환

```
X 좌표 범위: [-112, 224] → 32개 bin
Y 좌표 범위: [0, 224] → 32개 bin

bin_width_x = (224 - (-112)) / 32 = 10.5
bin_width_y = (224 - 0) / 32 = 7.0
```

### Soft Label (Gaussian)

학습 시 hard label 대신 Gaussian soft label 사용:

```
target_bin = (target_coord - coord_min) / (coord_max - coord_min) × num_bins
soft_label[i] = exp(-(i - target_bin)² / (2σ²))
soft_label = normalize(soft_label)  # sum = 1
```

σ = 1.5 권장

### 좌표 추론 (Soft Argmax)

```
probs = softmax(logits)
bin_centers = linspace(coord_min, coord_max, num_bins)
predicted_coord = Σ(probs × bin_centers)
```

---

## 손실 함수

### Hybrid Loss (KL + MSE)

```
Loss = α × KL_Loss + (1 - α) × MSE_Loss

KL_Loss = -Σ(soft_label × log(probs))
MSE_Loss = ||predicted_coord - target_coord||²

α = 0.7 (권장)
```

---

## 파라미터 추정

| 구성 요소 | 파라미터 수 (추정) |
|----------|-------------------|
| VGG Backbone | ~7.5M |
| Global Pooling | 0 |
| FC Hidden Layers | ~200K |
| Classification Heads | ~12K |
| **총합** | **~7.7M** |

### 경량화 옵션

채널 수를 절반으로 줄인 VGG-Light 버전:

| Block | 채널 (원본) | 채널 (경량) |
|-------|------------|------------|
| Block 1 | 64 | 32 |
| Block 2 | 128 | 64 |
| Block 3 | 256 | 128 |
| Block 4 | 512 | 256 |
| Block 5 | 512 | 256 |

경량 버전 파라미터: ~1.9M

---

## ONNX 출력 구조

```python
# 옵션 1: 좌표만
output = coordinates  # [B, 2]

# 옵션 2: 좌표 + Confidence
output = (coordinates, confidence)  # ([B, 2], [B, 2])

# 옵션 3: 전체
output = (coordinates, confidence, entropy)  # ([B, 2], [B, 2], [B, 2])
```

---

## 학습 설정 권장값

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 32~64 |
| Scheduler | CosineAnnealing |
| Epochs | 100~200 |
| Gaussian σ | 1.5 |
| Loss α | 0.7 |

---

## 데이터 전처리

```
1. BGR → Grayscale
2. Resize → 96×96
3. Normalize: (pixel / 255 - 0.449) / 0.226
4. 좌표 정규화: x → [0, 1], y → [0, 1]
```

---

## 모델 요약

```
VGGFloorModel(
  backbone: VGGBackbone(1 → 512, 5 blocks)
  global_pool: AdaptiveAvgPool2d(1)
  fc_layers: Sequential(
    Linear(512, 256) + ReLU + Dropout(0.3)
    Linear(256, 128) + ReLU + Dropout(0.2)
  )
  x_head: CoordClassificationHead(128 → 32 bins)
  y_head: CoordClassificationHead(128 → 32 bins)
)

Input:  [B, 1, 96, 96]
Output: coordinates [B, 2], confidence [B, 2], entropy [B, 2]
```