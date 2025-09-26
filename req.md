# FPD (Feature Point Detection) Model Requirements

## 프로젝트 개요
이미지에서 4개의 주요 특징점(center, floor, front, side)을 검출하는 딥러닝 모델 프로젝트

## 시스템 구성

### 1. Autoencoder 모델 (16x16 Image Reconstruction)

#### 1.1 목적
- 16x16 크기의 이미지 패치를 압축하고 복원하는 autoencoder
- 학습된 encoder는 이미지 특징 추출기로 활용 가능

#### 1.2 아키텍처
**ResidualEncoder16x16**
- 입력: 16x16x1 그레이스케일 이미지
- 출력: 16차원 latent vector
- 구조:
  - Conv2d(1→32, 3x3, stride=1) + BatchNorm + ReLU → 16x16x32
  - Conv2d(32→64, 3x3, stride=2) + BatchNorm + ReLU → 8x8x64
  - Conv2d(64→128, 3x3, stride=2) + BatchNorm + ReLU → 4x4x128
  - Conv2d(128→256, 3x3, stride=2) + BatchNorm + ReLU → 2x2x256
  - AdaptiveAvgPool2d(1) → 1x1x256
  - Linear(256→16) + ReLU → 16차원 latent

**ResidualDecoder16x16**
- 입력: 16차원 latent vector + residual features
- 출력: 16x16x1 복원된 이미지
- Residual connection을 통한 디테일 보존
- 3개의 스킵 커넥션 (res1: 32ch, res2: 64ch, res3: 128ch)

#### 1.3 데이터 처리
- 원본 이미지: data/base/not-labeled 폴더의 이미지들
- 전처리 과정:
  1. 이미지를 112x112로 리사이즈
  2. 112x112 이미지를 7x7 그리드로 나눔 (각 패치 16x16 크기)
  3. 각 16x16 패치가 하나의 학습 샘플
  4. 총 49개의 16x16 패치 생성 (7×7=49)
- 데이터 증강:
  - Random crop (0.8~1.0 비율)
  - Horizontal flip
- 메모리 기반 학습 (전체 데이터를 메모리에 로드)

#### 1.4 학습 설정 (ae_16x16_config.yml)
```yaml
model:
  name: "ae_16x16"
  latent_dim: 16

training:
  batch_size: 128
  epochs: 10000
  learning_rate: 0.001
  scheduler: reduce_on_plateau
  early_stopping:
    patience: 500
    min_delta: 0.000001
  save_interval: 10
  viz_interval: 10
```

#### 1.5 출력 파일
- 모델 저장: `model/`
  - `ae_16x16_best.pth` - 최고 성능 모델
  - `ae_16x16_current.pth` - 현재 모델 (best와 동일)
  - `ae_16x16_encoder.pth` - encoder만 분리 저장
  - `ae_16x16_epoch_XXXX.pth` - 체크포인트별 저장
- 결과 저장: `result/ae_16x16/`
  - 시각화 이미지 (epoch별)
  - 학습 로그 CSV

### 2. Multi-Point Detector 모델

#### 2.1 목적
이미지에서 4개의 주요 포인트 좌표를 예측

#### 2.2 아키텍처
**ConfigurableMLPModel**
- 입력: 특징 벡터 (16x16 패치 기반)
- 은닉층: [384, 256] 뉴런
- 출력: 8차원 (4포인트 × 2좌표)
- 특징:
  - Dropout: [0.2, 0.15]
  - BatchNorm 사용
  - ReLU 활성화 함수
  - He normal 초기화

#### 2.3 특징 추출 (16x16 이미지 기반)
**데이터 전처리**
- 112x112 이미지를 7x7 그리드로 나눔 (각 패치 16x16)
- 총 49개의 16x16 패치에서 특징 추출

**16x16 이미지별 특징**
- 픽셀 밝기 통계: 평균, 표준편차
- 엣지 특징: Canny edge 통계
- 색상 특징: RGB/HSV 채널별 평균
- ORB 키포인트 특징
- 텍스처 정보

**전체 특징 구성**
- 기본 정보: 3차원 (이미지 크기, 종횡비)
- 49개 × 16x16 이미지 특징
- Pre-trained encoder 사용 시: 49 × 16차원 = 784차원 latent features

#### 2.4 좌표 변환
1. 모델 출력: 정규화된 좌표 (0~1)
2. 112x112 좌표계 변환:
   - x: -112 ~ 224 범위 (이미지 왼쪽/오른쪽 외부까지 예측 가능)
   - y: 0 ~ 224 범위 (이미지 위/아래 외부까지 예측 가능)
   - **Note**: 예측된 특징점 좌표는 112x112 입력 이미지의 경계를 벗어날 수 있음. 이는 객체의 일부가 이미지 외부에 있거나, 특징점이 보이지 않는 영역을 추정하는 경우를 처리하기 위함
3. 원본 이미지 크기로 스케일링

#### 2.5 Enhanced Multi-Point Detector with Latent Embeddings

**LatentEmbeddingModel 아키텍처**
- 입력: 112x112 이미지
- 처리 과정:
  1. 112x112를 7x7 그리드로 나눔 (49개의 16x16 패치)
  2. Pre-trained encoder로 각 16x16 패치 → 16차원 latent vector
  3. Positional embedding 추가 (16차원)
  4. Feature fusion network 통과
  5. 8차원 좌표 출력 (4 포인트)

**네트워크 구성**
```
Input Processing:
- Image (112x112) → 49 patches (16x16 each)
- Encoder: 16x16 → 16-dim latent

Embedding Layer:
- Position Embeddings: 49 positions × 16 dimensions (7x7 grid)
- Combination: latent + positional (element-wise add)

Feature Processing:
- Option 1: Flatten → MLP [784 → 512 → 256 → 128 → 8]
- Option 2: Transformer blocks (2-4 layers)
  - Multi-head attention (8 heads)
  - Feed-forward network
  - Layer normalization
- Option 3: CNN-based fusion
  - Reshape to 7×7×16 feature map
  - Conv layers with residual connections
  - Global pooling → FC → 8 outputs

Output:
- 8 values (4 points × 2 coordinates)
- Sigmoid activation for normalized coords
```

**학습 전략**
- Frozen encoder (pre-trained autoencoder weights)
- Only train positional embeddings and fusion network
- Loss function: MSE or Smooth L1 loss
- Learning rate: 1e-4 for embeddings, 1e-3 for fusion network
- Batch size: 64-128 (메모리 효율적)

**데이터 증강 with Positional Consistency**
- Random crop 시 position embedding 재조정
- Horizontal flip 시 x 좌표 embedding 대칭 변환
- Rotation 증강 시 position embedding 회전 변환

### 3. 데이터 흐름

#### 3.1 Autoencoder 학습 플로우
```
원본 이미지 → 112x112 리사이즈 → 7x7 그리드로 나눔 (49개의 16x16 패치)
→ 각 16x16 패치로 Autoencoder 학습
→ Encoder (16x16 → 16차원) + Decoder (16차원 → 16x16)
```

#### 3.2 Multi-Point Detector 플로우
```
원본 이미지 → 112x112 리사이즈 → 7x7 그리드로 나눔 (49개의 16x16 패치)
→ 49개의 16x16 이미지에서 특징 추출
→ 특징 벡터 → MLP 모델 → 8차원 좌표
→ 원본 크기로 변환
```

#### 3.3 Latent Embedding Multi-Point Detector 플로우
```
원본 이미지 → 112x112 리사이즈 → 7x7 그리드로 나눔 (49개의 16x16 패치)
→ Pre-trained Encoder: 각 16x16 패치 → 16차원 latent vector
→ Positional Embedding 추가 (7x7 grid 좌표 기반)
→ Feature Fusion Network (MLP/Transformer/CNN)
→ 8차원 좌표 출력 → 원본 크기로 변환
```

### 4. 파일 구조
```
fpd_only_model/
├── src/
│   ├── 710.make_16x16_autoencoder.py  # 16x16 Autoencoder 학습 (완료)
│   ├── 700.make_7x7_autoencoder.py    # 7x7 Autoencoder (legacy)
│   ├── 200.learning_pytorch.py        # Baseline Multi-Point Detector 학습 (config.yml 사용)
│   ├── 201.learning_rvc.py            # RvC 및 고급 모델 학습 (생성 예정, rvc_config.yml 사용)
│   ├── 100.merge_learning_data.py     # 데이터 전처리
│   ├── ae_16x16_config.yml            # Autoencoder 설정
│   ├── config.yml                     # Baseline 모델 설정 (기존 파일)
│   ├── rvc_config.yml                 # RvC 및 고급 모델 설정 (생성 예정)
│   └── model_defs/
│       ├── autoencoder_16x16.py       # 16x16 Autoencoder 모델 정의
│       ├── autoencoder_7x7.py         # 7x7 Autoencoder (legacy)
│       ├── multi_point_model_pytorch.py  # Baseline Multi-point 모델
│       └── fpd_mix_ae_position_embedding.py  # 고급 모델 통합 파일 (생성 예정)
├── model/                              # 학습된 모델 저장
│   ├── ae_16x16_*.pth                 # Autoencoder 모델
│   └── mpd_*.pth                      # Multi-Point Detector 모델
├── result/
│   ├── ae_16x16/                      # Autoencoder 결과
│   └── mpd/                           # Multi-Point Detector 결과
├── data/
│   └── base/not-labeled/              # 학습 데이터
└── CLAUDE.md                           # AI 모델 저장 경로 명시

```

### 5. 향후 통합 가능성

#### 5.1 Encoder 특징 활용
- 학습된 autoencoder의 encoder를 feature extractor로 사용
- 49개의 16x16 패치 각각을 encoder에 통과 → 49 × 16 = 784차원 특징
- Hand-crafted 특징과 결합하여 더 풍부한 표현 가능

#### 5.2 통합 장점
- Encoder는 frozen 상태로 유지 (재학습 안함)
- 16x16 패치는 더 많은 디테일 보존
- 학습된 표현(encoder)과 hand-crafted 특징의 결합

#### 5.3 Latent Vector with Learned Positional Embedding 통합

**아키텍처 개요**
- 112x112 이미지를 7x7 그리드로 나눔 (총 49개의 16x16 패치)
- 학습된 autoencoder encoder를 통해 각 16x16 패치를 16차원 latent vector로 변환
- 각 latent vector에 해당 패치의 7x7 grid 좌표 기반 learned positional embedding 추가

**Positional Embedding 설계**
```
- Grid 좌표 시스템: (x, y) where x, y ∈ {0, 1, 2, ..., 6}
- Learned Embedding 차원: 16차원 (latent vector와 동일)
- 총 49개의 고유한 위치별 embedding 학습
- Position encoding: position_id = y * 7 + x (0~48)
```

**특징 결합 방식 (구현된 방식)**
**Additive 방식 (Element-wise addition)**:
- combined_feature = latent_vector + positional_embedding
- 차원 유지: 49 × 16 = 784차원
- 장점: 파라미터 효율적, 학습 안정적

*Note: Concatenative 방식과 Multi-Head Attention 방식은 메모리 효율과 계산 복잡도를 고려하여 현재 구현에서 제외됨*

**구현 상세**
```python
# Positional Embedding Layer
self.position_embeddings = nn.Embedding(49, embedding_dim=16)

# 각 16x16 패치 처리
for y in range(7):
    for x in range(7):
        patch_idx = y * 7 + x
        patch = image[y*16:(y+1)*16, x*16:(x+1)*16]  # 16x16 직접 추출

        # Encoder로 latent vector 추출
        latent = encoder(patch)  # 16차원

        # Positional embedding 추가
        pos_embed = position_embeddings(torch.tensor(patch_idx))

        # 결합
        combined = latent + pos_embed  # or concatenate
```

**장점**
- 공간적 위치 정보를 명시적으로 학습
- 16x16 패치의 풍부한 정보와 위치 정보의 시너지
- End-to-end 학습 가능한 구조
- 더 적은 파라미터로 효율적인 특징 추출

### 6. Hybrid Model with Latent Embedding (최신 구현)

#### 6.1 하이브리드 접근법 개요
기존 hand-crafted 특징과 autoencoder의 latent vector를 결합한 향상된 모델

**주요 구성요소**
- **기존 특징**: 903차원 hand-crafted features
  - 패치별 특징: 49 patches × 18 features = 882차원
  - 전역 특징: 21차원
    - 이미지 전체 통계 (3차원): 너비, 높이, 종횡비
    - 전체 색상 히스토그램 (12차원): RGB 각 4개 빈
    - 전체 엣지 밀도 (3차원): Canny, Sobel, Laplacian
    - 전체 텍스처 특징 (3차원): 대비, 에너지, 균질성
  - 총합: 882 + 21 = 903차원
- **Latent 특징**: 784차원 (49개 16x16 패치 × 16차원 latent)
- **Positional Embedding**: 49개 위치 × 16차원
- **총 입력 차원**: 1687차원 (903 + 784)

#### 6.2 모델 아키텍처 상세

##### 6.2.1 LatentEmbeddingModel 구조 (fpd_mix_ae_position_embedding.py)
```python
class LatentEmbeddingModel(nn.Module):
    def __init__(self, config):
        # Components:
        # 1. Pre-trained Encoder (frozen)
        self.encoder = ResidualEncoder16x16()  # 16x16 → 16-dim

        # 2. Positional Embeddings
        self.position_embeddings = nn.Embedding(49, 16)  # 7x7 grid

        # 3. Feature Fusion Network
        self.fusion_network = nn.Sequential(
            nn.Linear(1687, 768),  # Combined features
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 8)  # 4 points × 2 coords
        )
```

##### 6.2.2 특징 추출 파이프라인
```
1. 입력 이미지 (원본 크기)
   ↓
2. 112×112 리사이즈
   ↓
3. 특징 추출 (병렬 처리)
   ├─ Hand-crafted Features (903차원)
   │  ├─ 픽셀 통계 (98차원)
   │  ├─ 엣지 특징 (98차원)
   │  ├─ 색상 히스토그램 (12차원)
   │  ├─ HSV 특징 (153차원)
   │  ├─ ORB 키포인트 (392차원)
   │  └─ 색상 그리드 (147차원)
   │
   └─ Latent Features (784차원)
      ├─ 7×7 grid 분할 → 49개 16×16 패치
      ├─ 각 패치 → Encoder → 16차원 latent
      └─ Positional Embedding 추가
   ↓
4. Feature Concatenation (1687차원)
   ↓
5. Fusion Network → 8차원 출력
```

##### 6.2.3 Positional Embedding 설계
```python
# Grid 좌표 → Position ID 매핑
def get_position_id(x, y):
    return y * 7 + x  # 0 ~ 48

# 각 16x16 패치 처리
for y in range(7):
    for x in range(7):
        patch_idx = get_position_id(x, y)
        patch = image[y*16:(y+1)*16, x*16:(x+1)*16]

        # Latent 추출 + Positional Embedding
        latent = encoder(patch)  # 16-dim
        pos_embed = position_embeddings(patch_idx)  # 16-dim
        combined = latent + pos_embed  # Element-wise addition
```

#### 6.3 구현 파일 구조
```
src/
├── model_defs/
│   ├── multi_point_model_pytorch.py     # Baseline 모델
│   ├── autoencoder_16x16.py            # Autoencoder 정의
│   └── fpd_mix_ae_position_embedding.py # 고급 모델 통합 파일
├── 201.learning_rvc.py                 # RvC 및 고급 모델 학습 스크립트
└── rvc_config.yml                      # RvC 및 고급 모델 설정 파일
```

#### 6.4 설정 예시
Hybrid Model 설정은 `rvc_config.yml`의 `hybrid_model` 섹션 참조 (섹션 12.6)

**주의**: Hybrid Model은 `201.learning_rvc.py`로 학습 (Baseline은 `200.learning_pytorch.py` 사용)

#### 6.5 학습 전략
```yaml
training_strategy:
  # Phase 1: Encoder 학습 (완료)
  phase1:
    script: '710.make_16x16_autoencoder.py'
    status: 'completed'
    best_model: 'ae_16x16_encoder.pth'

  # Phase 2: Hybrid Model 학습
  phase2:
    script: '200.learning_pytorch.py'
    model_type: 'hybrid'
    encoder_frozen: true
    learning_rates:
      fusion_network: 0.001
      position_embeddings: 0.0001
    fine_tuning:
      enabled: false  # 나중에 encoder fine-tuning 가능
      start_epoch: 1000
      encoder_lr: 0.00001
```

#### 6.6 성능 향상 전략

##### 6.6.1 Multi-Scale Feature Fusion
```python
# 다양한 스케일의 특징 결합
features = {
    'global': hand_crafted_features,  # 전체 이미지 특징
    'local': latent_features,         # 16x16 패치 특징
    'spatial': positional_embeddings  # 공간 정보
}
```

##### 6.6.2 Attention Mechanism (선택적)
```python
# Self-attention을 통한 패치 간 관계 학습
class PatchAttention(nn.Module):
    def __init__(self, dim=16, num_heads=4):
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, patches):  # (49, batch, 16)
        attended, _ = self.attention(patches, patches, patches)
        return attended
```

##### 6.6.3 Regularization 기법
- Dropout: [0.2, 0.15, 0.1] (레이어별 차등 적용)
- Weight Decay: 0.01 (AdamW optimizer)
- Gradient Clipping: 5.0
- Early Stopping: patience 500

### 7. Advanced Hybrid Model with Dual Positional Embedding

#### 7.1 개선된 아키텍처 개요
Hand-crafted features와 Latent features 모두에 Positional Embedding을 적용한 향상된 모델

**핵심 개선점**
- Hand-crafted features에도 명시적 위치 정보 추가
- Grid 단위로 정렬된 multi-modal features
- Cross-attention mechanism을 통한 grid 간 관계 학습

#### 7.2 DualPositionalEmbeddingModel 아키텍처

##### 7.2.1 모델 구조 (fpd_mix_ae_position_embedding.py)
```python
class DualPositionalEmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 1. Pre-trained Encoder (frozen)
        self.encoder = ResidualEncoder16x16()  # 16x16 → 16-dim

        # 2. Dual Positional Embeddings
        # Hand-crafted features용 (더 큰 차원)
        self.handcrafted_pos_embed = nn.Embedding(49, 32)

        # Latent features용 (encoder 출력과 동일)
        self.latent_pos_embed = nn.Embedding(49, 16)

        # 3. Feature projection layers
        # Grid별 hand-crafted features를 32차원으로 projection
        self.handcrafted_proj = nn.Linear(18, 32)

        # 4. Cross-attention layer (제외)
        # self.cross_attention = nn.MultiheadAttention(
        #     embed_dim=48,  # 32(handcrafted) + 16(latent)
        #     num_heads=4
        # )

        # 5. Feature Fusion Network
        self.fusion_network = nn.Sequential(
            nn.Linear(1687, 768),  # 903 hand-crafted + 784 latent
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 8)  # 4 points × 2 coords
        )
```

##### 7.2.2 Grid-Aligned Feature Extraction
```python
def extract_grid_aligned_features(self, image):
    """각 grid cell에서 hand-crafted와 latent features를 동시 추출"""
    features_per_grid = []

    for y in range(7):
        for x in range(7):
            grid_idx = y * 7 + x

            # 16×16 패치 추출
            patch = image[y*16:(y+1)*16, x*16:(x+1)*16]

            # Hand-crafted features per patch (18차원)
            hc_features = self.extract_handcrafted_per_patch(patch)
            # - 픽셀 통계: mean, std (2차원)
            # - 엣지 밀도, 강도 (2차원)
            # - RGB 평균 (3차원)
            # - HSV 평균 (3차원)
            # - ORB 특징 (8차원)

            # Projection and positional embedding
            hc_projected = self.handcrafted_proj(hc_features)  # 18 → 32
            hc_with_pos = hc_projected + self.handcrafted_pos_embed(grid_idx)

            # Latent features with positional embedding
            latent = self.encoder(patch)  # 16차원
            latent_with_pos = latent + self.latent_pos_embed(grid_idx)

            # Concatenate at grid level
            grid_features = torch.cat([hc_with_pos, latent_with_pos])  # 48차원
            features_per_grid.append(grid_features)

    return torch.stack(features_per_grid)  # (49, 48)
```

##### 7.2.3 개선된 특징 처리 파이프라인
```
원본 이미지
    ↓
112×112 리사이즈
    ↓
7×7 Grid 분할 (49개 16×16 패치)
    ↓
각 Grid Cell에서 병렬 처리:
    ├─ Hand-crafted path:
    │   ├─ 패치별 특징 추출 (18차원)
    │   ├─ Projection: 18 → 32차원
    │   └─ Positional Embedding 추가 (32차원)
    │
    └─ Latent path:
        ├─ Encoder: 16×16 → 16차원
        └─ Positional Embedding 추가 (16차원)
    ↓
Grid-level Concatenation: [32 + 16] = 48차원/grid
    ↓
Global features 추가 (21차원)
    ↓
Total: 882 + 784 + 21 = 1687차원
    ↓
Fusion Network → 8차원 출력
```

##### 7.2.3.1 Fusion Network 상세 설명 (회귀 모델용)
```python
def build_fusion_network_regression(input_dim):
    """특징을 8차원 좌표로 변환하는 네트워크 (회귀)"""

    # input_dim 예시:
    # - Baseline: 903차원
    # - Hybrid: 1687차원
    # - Dual Positional: 2355차원

    return nn.Sequential(
        # 1단계: 고차원 특징을 중간 표현으로 압축
        nn.Linear(input_dim, 768),
        nn.BatchNorm1d(768),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),

        # 2단계: 추가 압축 및 비선형 변환
        nn.Linear(768, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.15),

        # 3단계: 좌표 예측을 위한 준비
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),

        # 4단계: 회귀 - 최종 8차원 출력 (4포인트 × 2좌표)
        nn.Linear(256, 8)
        # output[0:2] = center (x,y)
        # output[2:4] = floor (x,y)
        # output[4:6] = front (x,y)
        # output[6:8] = side (x,y)
    )

def forward(self, features):
    """Fusion Network 순전파"""
    # features: (batch_size, 2355)

    # Fusion Network 통과
    outputs = self.fusion_network(features)  # (batch_size, 8)

    # Sigmoid로 0~1 범위로 정규화 (옵션)
    outputs = torch.sigmoid(outputs)

    # 실제 좌표로 변환
    coords = self.denormalize_coordinates(outputs)

    return coords

def denormalize_coordinates(self, normalized):
    """정규화된 좌표를 실제 픽셀 좌표로 변환"""
    # normalized: (batch_size, 8) - 0~1 범위

    coords = torch.zeros_like(normalized)

    # X 좌표: -112 ~ 224 범위로 변환
    coords[:, 0::2] = normalized[:, 0::2] * 336 - 112

    # Y 좌표: 0 ~ 224 범위로 변환
    coords[:, 1::2] = normalized[:, 1::2] * 224

    return coords
```

##### 7.2.4 Cross-Attention Mechanism
```python
# Cross-attention은 계산 복잡도를 줄이기 위해 현재 구현에서 제외됨
# 필요시 다음 코드로 활성화 가능:
#
# def apply_grid_attention(self, grid_features):
#     """Grid 간 관계를 학습하는 attention mechanism"""
#     # grid_features: (batch, 49, 48)
#     grid_features = grid_features.transpose(0, 1)
#     attended, attention_weights = self.cross_attention(
#         grid_features,  # Query
#         grid_features,  # Key
#         grid_features   # Value
#     )
#     return attended.transpose(0, 1), attention_weights
```

#### 7.3 구현 세부사항

##### 7.3.1 Hand-crafted Features per Patch (18차원)
```python
def extract_handcrafted_per_patch(self, patch):
    """16×16 패치에서 간결한 hand-crafted features 추출"""
    features = []

    # 1. 픽셀 통계 (2차원)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    features.append(np.mean(gray) / 255.0)
    features.append(np.std(gray) / 255.0)

    # 2. 엣지 특징 (2차원)
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)  # 엣지 밀도
    features.append(np.std(edges) / 255.0)  # 엣지 강도 변화

    # 3. RGB 평균 (3차원)
    if len(patch.shape) == 3:
        for i in range(3):
            features.append(np.mean(patch[:, :, i]) / 255.0)
    else:
        features.extend([gray.mean() / 255.0] * 3)

    # 4. HSV 평균 (3차원)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV) if len(patch.shape) == 3 else patch
    for i in range(3):
        features.append(np.mean(hsv[:, :, i] if len(hsv.shape) == 3 else hsv) / 255.0)

    # 5. ORB 특징 (8차원)
    keypoints = cv2.ORB_create(nfeatures=5).detect(gray)
    if keypoints:
        features.append(len(keypoints) / 5.0)
        responses = [kp.response for kp in keypoints]
        features.append(np.mean(responses) / 100.0)
        # 키포인트 위치 통계
        x_coords = [kp.pt[0] / 16.0 for kp in keypoints]
        y_coords = [kp.pt[1] / 16.0 for kp in keypoints]
        features.extend([np.mean(x_coords), np.mean(y_coords),
                        np.std(x_coords), np.std(y_coords)])
        # 각도 통계
        angles = np.deg2rad([kp.angle for kp in keypoints])
        features.extend([np.mean(np.sin(angles)), np.mean(np.cos(angles))])
    else:
        features.extend([0] * 8)

    return np.array(features)  # 총 18차원
```

##### 7.3.2 설정 예시
Dual Positional Model 설정은 `rvc_config.yml`의 `dual_positional_model` 섹션 참조 (섹션 12.6)

**주의**: Dual Positional Model은 `201.learning_rvc.py`로 학습

#### 7.4 장점 및 기대 효과

##### 7.4.1 주요 장점
- **완전한 공간 인식**: 모든 특징이 명시적 위치 정보 포함
- **Feature Alignment**: 같은 위치의 서로 다른 특징이 정렬됨
- **Modular Design**: 각 컴포넌트 독립적 최적화 가능
- **간단한 구조**: 어텐션 제외로 학습 안정성 향상

##### 7.4.2 성능 향상 예측
- **기존 모델 (hand-crafted only)**: Baseline
- **Hybrid Model (섹션 6)**: 15-25% 오차 감소
- **Dual Positional Model**: 15-20% 오차 감소 예상 (Cross-attention 제외 버전)
- **특히 효과적인 경우**:
  - 객체가 특정 위치에 자주 나타나는 경우
  - 공간적 패턴이 중요한 경우
  - Grid 간 관계가 예측에 중요한 경우

##### 7.4.3 메모리 및 계산 효율성
```
기존 Hybrid Model:
- Input: 1687차원 (903 + 784)
- Parameters: ~1.5M

Dual Positional Model:
- Input: 2355차원 (2352 + 3)
- Parameters: ~1.9M (attention 제외)
- 장점: Grid별 병렬 처리 가능, 구조 단순화
- 단점: 메모리 사용량 증가 (~30%)
```

### 8. 실행 방법

#### 8.1 Autoencoder 학습 (완료)
```bash
cd src
python 710.make_16x16_autoencoder.py
# 이미 학습 완료: ae_16x16_encoder.pth 생성됨
```

#### 8.2 모델별 학습 실행
```bash
cd src

# 1. 기존 모델 (Baseline) - config.yml 사용
python 200.learning_pytorch.py

# 2. Hybrid Model (섹션 6) - rvc_config.yml 사용
python 201.learning_rvc.py --model hybrid

# 3. Dual Positional Model (섹션 7) - rvc_config.yml 사용
python 201.learning_rvc.py --model dual_positional

# 4. RvC Model (섹션 12) - rvc_config.yml 사용
python 201.learning_rvc.py --model rvc
```

#### 8.3 성능 비교 (선택적)
```bash
cd src
# compare_models.py 생성 후 사용 (필수 아님)
python compare_models.py --models baseline,hybrid,dual_positional,rvc
```

#### 8.4 설정 파일 분리
- `ae_16x16_config.yml`: Autoencoder 학습 파라미터
- `config.yml`: Baseline 모델 설정 (200.learning_pytorch.py 사용)
- `rvc_config.yml`: 고급 모델 설정 - Hybrid, Dual Positional, RvC (201.learning_rvc.py 사용)
- 모델 저장 경로: `@src\model_defs\`

### 9. 모델 구현 로드맵 및 우선순위

#### 9.1 구현 단계
**1단계 (완료)**: 16x16 Autoencoder
- 710.make_16x16_autoencoder.py로 특징 추출기 학습
- ae_16x16_encoder.pth 생성 완료

**2단계 (현재)**: Baseline Model
- Hand-crafted features only (903차원)
- 200.learning_pytorch.py --model baseline

**3단계 (계획)**: Hybrid Model
- Hand-crafted (903) + Latent features (784) = 1687차원
- Pre-trained encoder 활용
- 201.learning_rvc.py --model hybrid

**4단계 (향후)**: RvC Model
- Classification 기반 회귀 접근법
- 201.learning_rvc.py (구현 예정)

**Note**: Dual Positional Model은 Hybrid Model의 변형으로, 필요시 구현 가능

**생성될 파일**: 섹션 11 참조

### 10. 성능 지표 및 기대 효과

#### 10.1 성능 측정 지표
- **Autoencoder**: MSE Loss, SSIM (구조적 유사도)
- **Multi-Point Detection**: 픽셀 단위 MAE, 유클리드 거리
- **Early stopping**: patience 500, min_delta 0.000001
- **Attention Weights**: Grid 간 관계 시각화 (Dual Positional Model)

#### 10.2 모델별 기대 효과

##### 10.2.1 Baseline Model (Hand-crafted only)
- **특징 차원**: 903차원
- **성능**: Reference baseline

##### 10.2.2 Hybrid Model (섹션 6)
- **특징 차원**: 1687차원 (903 + 784)
- **개선점**: Latent features 추가
- **예상 성능 향상**: 15-25% 오차 감소

##### 10.2.3 Dual Positional Model (섹션 7)
- **특징 차원**: 2355차원 (2352 + 3)
- **개선점**: 모든 특징에 위치 정보 + Cross-attention
- **예상 성능 향상**: 25-35% 오차 감소

#### 10.3 주요 혁신 요소
- **Multi-modal Feature Fusion**: Hand-crafted + Learned features
- **Explicit Spatial Encoding**: Dual positional embeddings
- **Simplified Architecture**: 어텐션 제외로 더 안정적인 학습
- **Transfer Learning**: Pre-trained encoder 활용
- **Modular Architecture**: 각 컴포넌트 독립적 최적화 가능

### 11. 생성될 파일 목록 요약

#### 새로 생성될 파일들
1. **`src/rvc_config.yml`** - RvC 및 고급 모델 설정 파일 (Hybrid, Dual Positional, RvC 모델용)
2. **`src/model_defs/fpd_mix_ae_position_embedding.py`** - 통합 모델 파일 (Hybrid, Dual, RvC 모델 클래스 포함)
3. **`src/201.learning_rvc.py`** - RvC 및 고급 모델 학습 스크립트 (rvc_config.yml 사용)

#### 기존 파일 (수정 불필요)
1. **`src/200.learning_pytorch.py`** - 기존 config.yml 계속 사용 (Baseline 모델용)

### 12. Regression via Classification (RvC) Model

#### 12.1 개요
4개 포인트의 x,y 좌표를 회귀 문제가 아닌 분류 문제로 접근하는 혁신적인 방법. 각 좌표를 이산화된 빈(bin)으로 분류하여 예측.

**핵심 차이점: 회귀 vs RvC**
```
회귀 모델 (Regression):
- Fusion Network → Linear(256 → 8)
- 출력: 8개 연속값 (직접 좌표)
- 손실: MSE/L1 Loss

RvC 모델 (Classification):
- Fusion Network → 8개 분류 헤드
- 출력: 8개 확률 분포 (각 168 또는 112개 클래스)
- 손실: Cross-Entropy Loss
- 최종: Softmax → 가중평균 → 연속 좌표
```

**핵심 문제 해결**
- x 좌표 범위: -width ~ 2×width (-112 ~ 224, 총 336픽셀)
- y 좌표 범위: 0 ~ 2×height (0 ~ 224, 총 224픽셀)
- 넓은 좌표 범위를 효과적으로 처리하기 위한 적응적 빈 분할

#### 12.2 RvC 아키텍처

##### 12.2.1 좌표 이산화 전략
```python
class CoordinateDiscretization:
    """좌표 공간을 빈으로 분할"""

    def __init__(self, config):
        # X축: 336픽셀 범위를 168개 빈으로 분할 (각 빈당 2픽셀)
        self.num_x_bins = 168
        self.x_range = [-112, 224]

        # Y축: 224픽셀 범위를 112개 빈으로 분할 (각 빈당 2픽셀)
        self.num_y_bins = 112
        self.y_range = [0, 224]

        # 적응적 빈 분할 옵션
        if config['bin_strategy'] == 'adaptive':
            self.x_bins = self._create_adaptive_bins('x')
            self.y_bins = self._create_adaptive_bins('y')
        else:
            self.x_bins = np.linspace(*self.x_range, self.num_x_bins + 1)
            self.y_bins = np.linspace(*self.y_range, self.num_y_bins + 1)

    def _create_adaptive_bins(self, axis):
        """중심부는 세밀하게, 외곽은 넓게"""
        if axis == 'x':
            # 중심부 (-50~150): 100개 빈
            center = np.linspace(-50, 150, 100)
            # 좌측 외곽 (-112~-50): 20개 빈
            left = np.linspace(-112, -50, 20)
            # 우측 외곽 (150~224): 20개 빈
            right = np.linspace(150, 224, 20)
            return np.concatenate([left[:-1], center, right[1:]])
```

##### 12.2.2 RvCModel 구조 (fpd_mix_ae_position_embedding.py)
```python
class RvCModel(nn.Module):
    """Regression via Classification Model"""

    def __init__(self, config):
        super().__init__()

        self.num_x_bins = 168  # X축 빈 개수
        self.num_y_bins = 112  # Y축 빈 개수

        # 특징 추출 네트워크 (Fusion Network의 마지막 층 제외)
        self.feature_extractor = nn.Sequential(
            # 1단계: 고차원 특징을 중간 표현으로 압축
            nn.Linear(config['input_dim'], 768),  # input_dim: 903/1687/2355
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # 2단계: 추가 압축 및 비선형 변환
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            # 3단계: 분류를 위한 특징 준비
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            # 여기서 멈춤! 8차원으로 가지 않음
        )

        # RvC: 8개 분류 헤드 (회귀가 아닌 분류!)
        # X 좌표 분류기 (4개) - 각각 168개 클래스
        self.x_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.num_x_bins)  # 168개 빈으로 분류
            ) for _ in range(4)
        ])

        # Y 좌표 분류기 (4개) - 각각 112개 클래스
        self.y_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.num_y_bins)  # 112개 빈으로 분류
            ) for _ in range(4)
        ])

        # 불확실성 예측 헤드 (선택적)
        if config['predict_uncertainty']:
            self.uncertainty_head = nn.Linear(256, 8)

    def forward(self, x):
        # 특징 추출
        features = self.feature_extractor(x)  # (batch, 256)

        # 분류 (회귀 X)
        x_logits = [clf(features) for clf in self.x_classifiers]  # 4 × (batch, 168)
        y_logits = [clf(features) for clf in self.y_classifiers]  # 4 × (batch, 112)

        return {'x_logits': x_logits, 'y_logits': y_logits}
```

#### 12.3 손실 함수 설계

##### 12.3.1 Soft Label Generation
```python
def create_soft_labels(value, bins, sigma=2.0):
    """Gaussian 분포로 soft labels 생성"""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    distances = np.abs(bin_centers - value)

    # Gaussian 분포
    soft_labels = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    return soft_labels / soft_labels.sum()
```

##### 12.3.2 Hybrid Loss Function (fpd_mix_ae_position_embedding.py)
```python
class RvCLoss(nn.Module):
    """RvC를 위한 복합 손실 함수"""

    def __init__(self, config):
        self.use_soft_labels = config['use_soft_labels']
        self.ordinal_weight = config['ordinal_weight']

    def forward(self, predictions, targets):
        total_loss = 0

        # Classification loss
        if self.use_soft_labels:
            # KL divergence with soft labels
            loss = F.kl_div(log_softmax(predictions), soft_targets)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(predictions, target_classes)

        # Ordinal penalty (순서 관계 고려)
        if self.ordinal_weight > 0:
            pred_classes = predictions.argmax(dim=-1)
            distance = torch.abs(pred_classes - target_classes)
            loss += distance.mean() * self.ordinal_weight

        return loss
```

#### 12.4 추론 방법

##### 12.4.1 기본 추론 (Argmax)
```python
def decode_argmax(logits, bins):
    """가장 확률 높은 빈의 중심값 반환"""
    pred_class = torch.argmax(logits, dim=-1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers[pred_class]
```

##### 12.4.2 Softmax 기반 연속값 복원
```python
def decode_weighted(logits, bins, temperature=1.0):
    """Softmax 확률을 통한 원래 연속값 복원"""
    # Softmax로 각 빈의 확률 계산
    probs = F.softmax(logits / temperature, dim=-1)

    # 각 빈의 중심값
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 확률 가중 평균으로 연속값 복원
    # 예: x축 168개 빈 → softmax → 168개 확률 → 가중평균 → 원래 x 좌표
    continuous_value = torch.sum(probs * bin_centers, dim=-1)

    return continuous_value, probs  # 원래값과 확률 분포 반환

def decode_with_visualization(logits, bins):
    """Softmax 분포와 함께 원래값 복원 (시각화용)"""
    # Softmax 확률 계산
    probs = F.softmax(logits, dim=-1)

    # 빈 중심값
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 연속값 복원
    restored_value = torch.sum(probs * bin_centers, dim=-1)

    # Top-k 확률이 높은 빈들 (분포 확인용)
    top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
    top_values = bin_centers[top_indices]

    return {
        'value': restored_value,           # 복원된 연속값
        'distribution': probs,              # 전체 확률 분포
        'top_bins': top_indices,           # 상위 빈 인덱스
        'top_probs': top_probs,            # 상위 빈 확률
        'top_values': top_values           # 상위 빈의 실제 좌표값
    }
```

##### 12.4.3 불확실성 기반 추론
```python
def decode_with_uncertainty(logits, bins):
    """예측 불확실성도 함께 반환"""
    probs = F.softmax(logits, dim=-1)

    # 엔트로피 기반 불확실성
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

    # Top-k 확률 합 기반 신뢰도
    top_k_probs, _ = torch.topk(probs, k=3, dim=-1)
    confidence = top_k_probs.sum(dim=-1)

    # 예측값
    prediction = decode_weighted(logits, bins)

    return prediction, entropy, confidence
```

#### 12.5 구현 파일 구조

```
src/
├── 201.learning_rvc.py              # RvC 학습 스크립트
├── rvc_config.yml                   # 통합 설정 파일 (모든 모델 설정 포함)
└── model_defs/
    └── fpd_mix_ae_position_embedding.py  # 통합 모델 파일 (Hybrid, Dual, RvC 포함)
```

#### 12.6 RvC 및 고급 모델 설정 파일 (rvc_config.yml)

```yaml
# RvC 및 고급 모델 설정 파일
# Hybrid, Dual Positional, RvC 모델 설정 포함
# (Baseline 모델은 기존 config.yml 사용)

# ========================================
# 1. Hybrid Model 설정 (섹션 6)
# ========================================
hybrid_model:
  model_type: 'hybrid'
  use_latent: true
  use_handcrafted: true
  encoder_path: '../model/ae_16x16_encoder.pth'
  encoder_frozen: true  # Encoder 가중치 고정

  # 차원 설정
  handcrafted_dim: 903
  latent_dim: 784  # 49 patches × 16 dim
  input_dim: 1687  # 903 + 784

  # Positional Embedding
  positional_embedding:
    enabled: true
    embedding_dim: 16
    grid_size: 7
    combination: 'add'  # 'add' or 'concat'

  # Fusion Network
  hidden_dims: [768, 512, 256]
  dropout_rates: [0.2, 0.15, 0.1]
  activation: 'relu'
  use_batch_norm: true
  weight_init: 'he_normal'

# ========================================
# 2. Dual Positional Model 설정 (선택적, 섹션 7)
# ========================================
dual_positional_model:
  model_type: 'dual_positional'
  encoder_path: '../model/ae_16x16_encoder.pth'
  encoder_frozen: true

  # Dual Positional Embeddings
  positional_embedding:
    handcrafted:
      enabled: true
      embedding_dim: 32
      features_per_patch: 18
      projection_dim: 32
    latent:
      enabled: true
      embedding_dim: 16

  # Feature dimensions
  hand_crafted_dim: 903
  latent_dim: 784
  input_dim: 1687

  # Fusion Network
  hidden_dims: [768, 512, 256]
  dropout_rates: [0.2, 0.15, 0.1]
  activation: 'relu'
  use_batch_norm: true

# ========================================
# 3. RvC Model 설정 (섹션 12)
# ========================================
rvc_model:

  # 좌표 범위 설정
  x_range: [-112, 224]  # -width ~ 2*width
  y_range: [0, 224]     # 0 ~ 2*height

  # 빈 설정
  num_x_bins: 168  # x축 클래스 수 (336픽셀 / 168빈 = 각 빈당 2픽셀)
  num_y_bins: 112  # y축 클래스 수 (224픽셀 / 112빈 = 각 빈당 2픽셀)

  # 빈 분할 전략
  bin_strategy: 'uniform'  # 'uniform', 'adaptive', 'quantile'
  adaptive_config:
    center_range: [-50, 150]  # 중심부 범위
    center_bins: 100          # 중심부 빈 개수
    outer_bins: 20           # 외곽 빈 개수

  # 손실 함수 설정
  loss_type: 'ce_with_soft'  # 'ce', 'ce_with_soft', 'ordinal'
  use_soft_labels: true
  soft_label_sigma: 2.0  # Gaussian soft label 표준편차
  ordinal_weight: 0.1    # 순서 관계 패널티 가중치

  # 추론 설정
  inference_method: 'weighted_average'  # 'argmax', 'weighted_average'
  temperature: 1.0  # Softmax temperature

  # 불확실성 예측
  predict_uncertainty: true
  uncertainty_threshold: 0.5

# 모델 아키텍처
architecture:
  model_type: 'rvc'
  input_dim: 903  # 또는 1687 (hybrid), 2355 (dual_pos)

  # 특징 추출 네트워크
  feature_network:
    hidden_dims: [768, 512, 256]
    dropout_rates: [0.2, 0.15, 0.1]
    activation: 'relu'
    use_batch_norm: true

  # 분류 헤드
  classifier_head:
    hidden_dim: 128
    dropout: 0.1

# ========================================
# 4. 공통 학습 설정
# ========================================
training:
  batch_size: 64
  epochs: 100000
  learning_rate: 0.001
  weight_decay: 0.01
  optimizer: 'adamw'

  # 스케줄러
  scheduler:
    type: 'cosine_with_warmup'
    warmup_epochs: 100
    min_lr: 0.00001

  # Early stopping
  early_stopping:
    patience: 1000
    min_delta: 0.0001
    monitor: 'val_top1_accuracy'

# ========================================
# 5. 평가 지표
# ========================================
metrics:
  # 분류 지표
  top_k_accuracy: [1, 3, 5]

  # 회귀 지표 (변환 후)
  pixel_mae: true
  euclidean_distance: true

  # 불확실성 지표
  calibration_error: true
  entropy_analysis: true
```

#### 12.7 장점 및 기대 효과

##### 12.7.1 주요 장점
- **다중 모드 예측**: 불확실한 경우 여러 위치 후보 표현 가능
- **불확실성 정량화**: 분류 확률로 예측 신뢰도 직접 측정
- **학습 안정성**: Cross-entropy가 MSE보다 안정적 수렴
- **넓은 좌표 범위 처리**: -width~2×width 범위 효과적 처리

##### 12.7.2 성능 예측
- **기존 회귀 모델**: Baseline
- **RvC (uniform bins)**: 유사하거나 약간 개선
- **RvC (adaptive bins)**: 10-15% 오차 감소 예상
- **RvC with uncertainty**: 신뢰도 기반 후처리로 추가 개선

##### 12.7.3 메모리 및 계산 효율성
```
RvC Model:
- Input: 903차원 (또는 1687, 2355)
- Parameters: ~1.8M (8개 분류 헤드 포함)
- 장점: 병렬 분류 가능, GPU 효율적
- 단점: 분류 헤드로 인한 메모리 증가
```

#### 12.8 전체 추론 파이프라인

##### 12.8.1 완전한 좌표 복원 과정
```python
class RvCInference:
    """RvC 모델의 완전한 추론 파이프라인"""

    def __init__(self, model, config):
        self.model = model
        self.x_bins = self._create_bins(config['x_range'], config['num_x_bins'])
        self.y_bins = self._create_bins(config['y_range'], config['num_y_bins'])

    def predict(self, image):
        """이미지 → 특징 → 분류 → Softmax → 원래 좌표"""

        # 1. 특징 추출
        features = self.extract_features(image)  # (batch, 903)

        # 2. 모델 추론 - 8개 분류 헤드 출력
        outputs = self.model(features)
        # outputs['x_logits']: 4개 x좌표 logits [(batch, 168), ...]
        # outputs['y_logits']: 4개 y좌표 logits [(batch, 112), ...]

        # 3. Softmax를 통한 확률 변환 및 원래값 복원
        points = []
        for i in range(4):
            # X 좌표 복원
            x_probs = F.softmax(outputs['x_logits'][i], dim=-1)  # (batch, 168)
            x_bin_centers = (self.x_bins[:-1] + self.x_bins[1:]) / 2
            x_coord = torch.sum(x_probs * x_bin_centers, dim=-1)  # 가중평균

            # Y 좌표 복원
            y_probs = F.softmax(outputs['y_logits'][i], dim=-1)  # (batch, 112)
            y_bin_centers = (self.y_bins[:-1] + self.y_bins[1:]) / 2
            y_coord = torch.sum(y_probs * y_bin_centers, dim=-1)  # 가중평균

            points.append({
                'x': x_coord,          # 복원된 x 좌표 (-112 ~ 224)
                'y': y_coord,          # 복원된 y 좌표 (0 ~ 224)
                'x_probs': x_probs,    # x축 확률 분포
                'y_probs': y_probs     # y축 확률 분포
            })

        return points

    def visualize_prediction(self, point_data):
        """예측 결과와 확률 분포 시각화"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for i, point in enumerate(point_data):
            # X 좌표 확률 분포
            ax_x = axes[0, i]
            ax_x.bar(range(168), point['x_probs'].cpu().numpy())
            ax_x.axvline(point['x'], color='r', label=f'x={point["x"]:.1f}')
            ax_x.set_title(f'Point {i+1} X Distribution')

            # Y 좌표 확률 분포
            ax_y = axes[1, i]
            ax_y.bar(range(112), point['y_probs'].cpu().numpy())
            ax_y.axvline(point['y'], color='r', label=f'y={point["y"]:.1f}')
            ax_y.set_title(f'Point {i+1} Y Distribution')

        plt.tight_layout()
        return fig
```

##### 12.8.2 정확도 평가 with Softmax (200.learning_pytorch.py 스타일)
```python
def evaluate_with_softmax(model, dataloader, config, device):
    """Softmax 기반 복원으로 정확도 평가 - 평균과 표준편차 계산"""

    model.eval()
    all_errors = []  # 모든 오차 저장
    point_errors = [[] for _ in range(4)]  # 포인트별 오차
    coord_errors = {'x': [], 'y': []}  # 좌표축별 오차

    # 빈 중심값 미리 계산
    x_bin_centers = torch.tensor(
        (config.x_bins[:-1] + config.x_bins[1:]) / 2
    ).to(device)
    y_bin_centers = torch.tensor(
        (config.y_bins[:-1] + config.y_bins[1:]) / 2
    ).to(device)

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)  # (batch, 8)

            # 모델 추론 - 8개 분류 헤드 출력
            outputs = model(features)

            # Softmax를 통한 좌표 복원
            predictions = []
            for i in range(4):
                # X 좌표 복원
                x_probs = F.softmax(outputs['x_logits'][i], dim=-1)
                x_pred = torch.sum(x_probs * x_bin_centers, dim=-1)

                # Y 좌표 복원
                y_probs = F.softmax(outputs['y_logits'][i], dim=-1)
                y_pred = torch.sum(y_probs * y_bin_centers, dim=-1)

                predictions.extend([x_pred, y_pred])

                # 포인트별 오차 계산
                x_error = (x_pred - targets[:, i*2]).cpu().numpy()
                y_error = (y_pred - targets[:, i*2+1]).cpu().numpy()

                # 유클리드 거리
                euclidean_error = np.sqrt(x_error**2 + y_error**2)
                point_errors[i].extend(euclidean_error)

                # 좌표축별 오차
                coord_errors['x'].extend(np.abs(x_error))
                coord_errors['y'].extend(np.abs(y_error))

            predictions = torch.stack(predictions, dim=1)  # (batch, 8)

            # 전체 오차 계산
            errors = (predictions - targets).cpu().numpy()
            all_errors.extend(errors.flatten())

    # 통계 계산
    all_errors = np.array(all_errors)

    # 전체 통계
    total_mae = np.mean(np.abs(all_errors))
    total_std = np.std(all_errors)
    total_rmse = np.sqrt(np.mean(all_errors**2))

    # 포인트별 통계
    point_stats = []
    point_names = ['Center', 'Floor', 'Front', 'Side']
    for i, name in enumerate(point_names):
        point_err = np.array(point_errors[i])
        point_stats.append({
            'name': name,
            'mean': np.mean(point_err),
            'std': np.std(point_err),
            'max': np.max(point_err),
            'min': np.min(point_err)
        })

    # 좌표축별 통계
    x_errors = np.array(coord_errors['x'])
    y_errors = np.array(coord_errors['y'])

    # 결과 출력 - 포인트별 유클리드 거리만 표시
    print("\n" + "="*60)
    print("RvC 모델 평가 결과 (Softmax 기반 복원)")
    print("="*60)

    print(f"\n[포인트별 유클리드 거리 오차]")
    for stat in point_stats:
        print(f"  {stat['name']:6s}: {stat['mean']:.4f} ± {stat['std']:.4f} pixels "
              f"(최대: {stat['max']:.4f}, 최소: {stat['min']:.4f}, 표준편차: {stat['std']:.4f})")

    print("="*60 + "\n")

    # 딕셔너리로 반환
    return {
        'mae': total_mae,
        'std': total_std,
        'rmse': total_rmse,
        'point_stats': point_stats,
        'x_mae': np.mean(x_errors),
        'y_mae': np.mean(y_errors),
        'x_std': np.std(x_errors),
        'y_std': np.std(y_errors),
        'all_errors': all_errors
    }
```

##### 12.8.3 학습 중 실시간 모니터링
```python
def train_with_monitoring(model, train_loader, val_loader, config, device):
    """학습 중 평균과 표준편차 실시간 모니터링"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        # 학습
        model.train()
        train_losses = []

        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)

            # Forward pass
            outputs = model(features)

            # 손실 계산 (RvC Loss)
            loss = compute_rvc_loss(outputs, targets, config)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 검증 (매 N epoch마다)
        if epoch % config['eval_interval'] == 0:
            val_stats = evaluate_with_softmax(model, val_loader, config, device)

            # 로깅
            print(f"\nEpoch {epoch}/{config['epochs']}")
            print(f"  Train Loss: {np.mean(train_losses):.4f}")
            print(f"  Val MAE: {val_stats['mae']:.4f} ± {val_stats['std']:.4f}")
            print(f"  Val X-axis: {val_stats['x_mae']:.4f} ± {val_stats['x_std']:.4f}")
            print(f"  Val Y-axis: {val_stats['y_mae']:.4f} ± {val_stats['y_std']:.4f}")

            # 포인트별 간략 출력
            for stat in val_stats['point_stats']:
                print(f"    {stat['name']}: {stat['mean']:.2f} ± {stat['std']:.2f}")
```

#### 12.9 실행 방법

```bash
cd src

# 1. RvC 모델 학습 (rvc_config.yml 사용)
python 201.learning_rvc.py --model rvc

# 2. 빈 전략 비교 (선택적)
python 201.learning_rvc.py --model rvc --bin_strategy uniform
python 201.learning_rvc.py --model rvc --bin_strategy adaptive

# 3. 추론 방법 비교 (선택적)
python 201.learning_rvc.py --model rvc --inference argmax
python 201.learning_rvc.py --model rvc --inference weighted
```

#### 12.10 하이브리드 접근법 (Advanced)

##### 12.10.1 Coarse-to-Fine 예측
```python
class CoarseToFineRvC(nn.Module):
    """2단계 예측: 대략적 위치 → 세부 위치"""

    def __init__(self):
        # 1단계: 10×10 그리드로 대략적 위치
        self.coarse_classifier = nn.Linear(256, 100)  # 10×10 grid

        # 2단계: 그리드 내 세부 위치 (회귀 또는 세밀 분류)
        self.fine_regressor = nn.Linear(256 + 100, 8)  # 잔차 예측
```

##### 12.10.2 Ensemble 방법
- **Multi-scale bins**: 여러 해상도의 빈으로 앙상블
- **Regression + Classification**: 두 방법의 가중 평균
- **Temperature scaling**: 다양한 온도로 추론 후 앙상블