# 학습 스크립트 표준화 계획

## 현황 분석

### 대상 파일
- `src/200.learning_pytorch.py` - PyTorch 기본 학습 스크립트
- `src/201.learning_kornia.py` - Kornia 기반 학습 스크립트
- `src/202.learning_fpd_all.py` - FPD 통합 학습 스크립트

## 파이프라인 요소 비교

### 1. 공통 요소 (모든 파일에서 동일하거나 유사)

| 구분 | 공통 요소 | 세부 내용 |
|------|----------|-----------|
| **기본 구조** | PyTorch Dataset/DataLoader | Dataset 클래스 상속, DataLoader 사용 |
| | Train/Val/Test 분할 | ID 끝자리 기반 test 분리, random seed 고정 val 분리 |
| | GPU/CPU 자동 선택 | CUDA 가용성 체크 및 device 설정 |
| **데이터 처리** | labels.txt 파싱 | CSV 형식 레이블 파일 읽기 |
| | 좌표 정규화 | Min-Max normalization (설정 가능한 범위) |
| | 4개 포인트 처리 | center, floor, front, side 좌표 |
| **학습 루프** | Epoch 기반 학습 | train_epoch() → validate() 구조 |
| | Gradient Clipping | 설정 가능한 gradient clipping |
| | Early Stopping | patience와 min_delta 기반 |
| **최적화** | Optimizer 선택 | Adam, AdamW, SGD 지원 |
| | Scheduler 지원 | ReduceLROnPlateau, CosineAnnealingLR |
| | Learning Rate | 설정 가능한 초기 학습률 |
| **로깅** | DualLogger 클래스 | 화면과 파일 동시 출력 |
| | CSV 로그 저장 | 학습 이력 CSV 저장 |
| | 모델 정보 JSON | 모델 메타데이터 저장 |
| **체크포인트** | Best 모델 저장 | 검증 손실 기준 최적 모델 |
| | 주기적 저장 | save_frequency 설정 |
| | 학습 재개 | checkpoint에서 재시작 가능 |
| **평가** | 픽셀 오차 계산 | MAE, 거리 오차 |
| | 포인트별 분석 | X, Y, 거리 오차 통계 |
| **설정** | YAML config | 계층적 설정 구조 |
| **증강** | 크롭 증강 | 랜덤 스케일/시프트 크롭 |

### 2. 고유 요소 (파일별 차이점)

| 구분 | 200.learning_pytorch.py | 201.learning_kornia.py | 202.learning_fpd_all.py |
|------|-------------------------|------------------------|-------------------------|
| **Dataset 클래스** | MultiPointDataset | MultiPointDataset (Kornia용) | FPDDataset |
| **특징 추출** | OpenCV 기반 903차원 특징 | Kornia 이미지 처리 | 3종류 특징 (global, patch_cv, patch_latent) |
| **입력 형태** | 특징 벡터 (903D) | 이미지 텐서 (3x112x112) | 다중 특징 딕셔너리 |
| **모델 구조** | 단순 FC 네트워크 | Kornia + FPD 분류 모델 | FPDCoordinateRegression (하이브리드) |
| **손실 함수** | MSE Loss | FPD 분류 기반 회귀 손실 | 분류 + 회귀 가중 조합 |
| **출력 형태** | 직접 좌표 (8차원) | 분류 후 좌표 변환 | 분류 확률 + 회귀 좌표 |
| **특징 정규화** | Mean/Std 정규화 | 없음 (이미지 직접 처리) | 특징별 개별 정규화 |
| **Config 파일** | config.yml | config_kornia.yml | config_fpd.yml |
| **ONNX 변환** | 기본 ONNX export | Kornia 호환 ONNX | 없음 |
| **메모리 전략** | 특징 사전 추출 및 저장 | 이미지 텐서 메모리 저장 | 특징 사전 추출 및 저장 |
| **증강 방식** | 특징 레벨 노이즈 | 이미지 레벨 크롭 | 다중 레벨 노이즈 |
| **세션 관리** | 단일 로그 파일 | 단일 로그 파일 | 타임스탬프 세션 디렉토리 |
| **후처리** | 자동 예측 및 시각화 | 없음 | 상세 오차 분석 |
| **오차 분석** | 샘플별 상세 추적 | 기본 통계 | 백분위수 포함 상세 통계 |
| **특별 기능** | Bad sample 분석 | Position encoding (7x7) | Raw 오차 데이터 export |

## 표준화 계획

### 목표
- 코드 중복 제거
- 유지보수성 향상
- 모듈화를 통한 재사용성 증대
- 일관된 인터페이스 제공

### 표준화 구조

#### 1. 공통 모듈 생성 (`src/common/`)

##### `base_dataset.py` - 기본 Dataset 클래스
```python
class BaseMultiPointDataset(Dataset):
    """공통 데이터셋 기능"""
    - load_data(): 레이블 파일 로드 및 train/val/test 분할
    - normalize_coordinates(): 좌표 정규화
    - denormalize_coordinates(): 좌표 역정규화
    - apply_crop_augmentation(): 크롭 증강
    - 추상 메서드: extract_features(), precompute_data()
```

##### `trainer.py` - 통합 학습 관리자
```python
class UnifiedTrainer:
    """표준 학습 파이프라인"""
    - train_epoch(): 1 에폭 학습
    - validate(): 검증/테스트 평가
    - fit(): 전체 학습 루프
    - save_checkpoint(): 체크포인트 저장
    - load_checkpoint(): 체크포인트 로드
    - early_stopping(): 조기 종료 관리
```

##### `utils.py` - 유틸리티 함수
```python
- DualLogger: 화면/파일 동시 로깅
- calculate_errors(): 오차 계산
- save_training_log(): CSV 로그 저장
- save_model_info(): JSON 메타데이터 저장
- setup_device(): GPU/CPU 설정
- count_parameters(): 파라미터 수 계산
```

##### `evaluator.py` - 평가 메트릭
```python
class Evaluator:
    """통합 평가 시스템"""
    - calculate_pixel_errors(): 픽셀 오차
    - calculate_point_errors(): 포인트별 오차
    - generate_statistics(): 통계 생성
    - export_results(): 결과 export
    - visualize_errors(): 오차 시각화
```

##### `checkpoint_manager.py` - 체크포인트 관리
```python
class CheckpointManager:
    """체크포인트 저장/로드 관리"""
    - save(): 체크포인트 저장
    - load(): 체크포인트 로드
    - save_best(): 최고 모델 저장
    - cleanup_old(): 오래된 체크포인트 정리
```

#### 2. 설정 파일 통합 (`src/configs/`)

##### `base_config.yml` - 공통 설정
```yaml
# 모든 모델 공통 설정
data_split:
  test_id_suffix:
  validation_ratio:
  random_seed:

training:
  batch_size:
  epochs:
  learning_rate:
  optimizer:
  scheduler:
  early_stopping:
  gradient_clip:

logging:
  log_dir:
  save_csv:
  progress_interval:

checkpointing:
  save_dir:
  save_frequency:
  save_best_only:
```

##### 모델별 특화 설정
- `config_pytorch.yml`: PyTorch 모델 특화 설정 (특징 추출 방식 등)
- `config_kornia.yml`: Kornia 모델 특화 설정 (이미지 처리 파라미터 등)
- `config_fpd.yml`: FPD 모델 특화 설정 (다중 특징, 손실 가중치 등)

#### 3. 리팩토링된 학습 스크립트 구조

##### 공통 템플릿
```python
# 200/201/202_learning_refactored.py

from common.base_dataset import BaseMultiPointDataset
from common.trainer import UnifiedTrainer
from common.utils import DualLogger, setup_device
from common.evaluator import Evaluator

# 1. 모델별 Dataset 구현 (BaseMultiPointDataset 상속)
class ModelSpecificDataset(BaseMultiPointDataset):
    def extract_features(self, ...):
        # 모델별 특징 추출

# 2. 모델 생성 함수
def create_model(config):
    # 모델별 아키텍처 생성

# 3. 손실 함수 정의
def create_loss_fn(config):
    # 모델별 손실 함수

# 4. 메인 함수
def main():
    config = load_config()
    device = setup_device(config)

    # Dataset/DataLoader
    dataset = ModelSpecificDataset(config)

    # Model/Optimizer
    model = create_model(config)
    loss_fn = create_loss_fn(config)

    # Trainer
    trainer = UnifiedTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config
    )

    # Training
    trainer.fit(train_loader, val_loader)

    # Evaluation
    evaluator = Evaluator()
    results = evaluator.evaluate(model, test_loader)
```

### 구현 단계

#### Phase 1: 공통 모듈 생성 (1-2일)
1. `src/common/` 디렉토리 생성
2. `base_dataset.py` 구현
3. `utils.py` 구현 (기존 코드에서 추출)
4. `evaluator.py` 구현

#### Phase 2: Trainer 통합 (1-2일)
1. `trainer.py` 구현
2. `checkpoint_manager.py` 구현
3. 학습 루프 표준화

#### Phase 3: 설정 통합 (1일)
1. `base_config.yml` 생성
2. 모델별 설정 파일 정리
3. 설정 로더 구현

#### Phase 4: 스크립트 리팩토링 (2-3일)
1. 200.learning_pytorch.py 리팩토링
2. 201.learning_kornia.py 리팩토링
3. 202.learning_fpd_all.py 리팩토링

#### Phase 5: 테스트 및 검증 (1-2일)
1. 각 모델 학습 테스트
2. 체크포인트 호환성 확인
3. 성능 비교 (리팩토링 전후)
4. 문서화

### 기대 효과

1. **코드 중복 감소**: 약 60-70% 코드 중복 제거 예상
2. **유지보수성 향상**: 공통 로직 한 곳에서 관리
3. **확장성**: 새로운 모델 추가 시 최소한의 코드만 작성
4. **일관성**: 모든 모델에 동일한 학습/평가 파이프라인 적용
5. **디버깅 용이**: 모듈화로 문제 추적 용이

### 주의 사항

1. **호환성 유지**: 기존 체크포인트와 호환성 유지
2. **성능 보장**: 리팩토링 후에도 동일한 학습 성능 보장
3. **점진적 마이그레이션**: 한 번에 모든 것을 변경하지 않고 단계적 적용
4. **테스트 우선**: 각 단계마다 충분한 테스트 수행

### 추가 고려사항

1. **플러그인 시스템**: 새로운 증강 방법, 손실 함수 등을 쉽게 추가할 수 있는 구조
2. **설정 검증**: 설정 파일 스키마 검증 기능
3. **CLI 인터페이스**: 명령줄에서 다양한 옵션 제공
4. **모니터링**: TensorBoard, WandB 등 통합 지원 고려

## 즉시 적용 사항

### 1. max_train_images 설정 표준화

#### 목표
모든 학습 config 파일에 `max_train_images` 설정을 기본으로 추가하여 학습 데이터 수량을 일관되게 제어

#### 현재 상태
- **config_fpd.yml**: 이미 구현됨 (12번째 줄)
  ```yaml
  max_train_images: 200  # 학습 이미지 수 제한 (0이면 모두 사용)
  ```
- **config.yml**: 없음
- **config_kornia.yml**: 없음

#### 수정 계획

##### 1. Config 파일 수정
모든 config 파일의 `data_split` 섹션에 추가:
```yaml
data_split:
  test_id_suffix: '1'
  validation_ratio: 0.2
  random_seed: 42
  max_train_images: 0  # 학습 이미지 수 제한 (0: 모두 사용, 양수: 해당 개수만 사용)
```

**수정 대상:**
- `config.yml`: 8번째 줄과 75번째 줄 (pytorch_model 내부)
- `config_kornia.yml`: 해당 data_split 섹션들

##### 2. Dataset 클래스 수정
각 Dataset 클래스의 `load_data()` 메서드에 다음 로직 추가:

```python
def load_data(self, labels_file: str):
    # ... 기존 데이터 로드 로직 ...

    # max_train_images 설정 적용
    max_train_images = self.config.get('data_split', {}).get('max_train_images', 0)

    if self.mode == 'train' and max_train_images > 0:
        # 학습 데이터 수 제한
        self.data = self.data[:max_train_images]
        print(f"학습 데이터 제한: {max_train_images}개만 사용 (전체: {len(self.data)}개)")
```

**수정 대상 스크립트:**
- `200.learning_pytorch.py`: MultiPointDataset 클래스
- `201.learning_kornia.py`: MultiPointDataset 클래스
- `202.learning_fpd_all.py`: 이미 구현되어 있음 (확인 필요)

##### 3. 기본값 설정
- 개발/테스트 시: `max_train_images: 200` (빠른 반복)
- 실제 학습 시: `max_train_images: 0` (모든 데이터 사용)

#### 장점
1. **일관성**: 모든 학습 스크립트에서 동일한 방식으로 데이터 제한
2. **편의성**: Config 파일에서 간단히 조정 가능
3. **개발 효율**: 빠른 프로토타이핑과 테스트 가능
4. **명확성**: 설정 의도가 명확하게 문서화됨

### 2. 데이터 증강 설정 표준화

#### 목표
학습 데이터 생성 시 데이터 증강 알고리즘 적용 여부를 config 파일에서 쉽게 제어할 수 있도록 설정 표준화

#### 현재 상태
- **config.yml**: `augmentation.enabled: true/false` 설정 있음
- **config_kornia.yml**: 동일한 augmentation 설정 구조
- **config_fpd.yml**: augmentation 설정 구조 확인 필요

#### 수정 계획

##### 1. 표준 증강 설정 구조
모든 config 파일에 통일된 증강 설정 구조 적용:
```yaml
augmentation:
  enabled: false  # 전체 증강 on/off (마스터 스위치)
  augment_count: 3  # 원본 데이터당 증강 샘플 수

  # 크롭 증강
  crop:
    enabled: true  # 개별 증강 방법 on/off
    min_ratio: 0.8  # 최소 크롭 비율 (80%)
    max_ratio: 1.0  # 최대 크롭 비율 (100%)
    max_shift: 0.15  # 최대 이동 비율 (15%)

  # 노이즈 증강
  noise:
    enabled: true
    std: 0.01  # 노이즈 표준편차
    type: 'gaussian'  # 노이즈 타입

  # 플립 증강 (향후 추가 가능)
  flip:
    enabled: false
    horizontal: true
    vertical: false

  # 회전 증강 (향후 추가 가능)
  rotation:
    enabled: false
    max_angle: 15  # 최대 회전 각도

  # 밝기/대비 조정 (향후 추가 가능)
  color:
    enabled: false
    brightness_range: [0.8, 1.2]
    contrast_range: [0.8, 1.2]
```

##### 2. Dataset 클래스 수정 방안
각 Dataset 클래스에 통일된 증강 제어 로직 추가:

```python
class MultiPointDataset(Dataset):
    def __init__(self, ..., mode='train', config=None):
        # 증강 설정 읽기
        aug_config = config.get('augmentation', {})

        # 마스터 스위치: train 모드에서만 증강 가능
        self.augment_enabled = (
            aug_config.get('enabled', False) and
            mode == 'train'
        )

        # 증강이 활성화된 경우에만 증강 수 설정
        if self.augment_enabled:
            self.augment_count = aug_config.get('augment_count', 0)

            # 개별 증강 방법 설정
            self.crop_enabled = aug_config.get('crop', {}).get('enabled', False)
            self.noise_enabled = aug_config.get('noise', {}).get('enabled', False)
            self.flip_enabled = aug_config.get('flip', {}).get('enabled', False)

            # 증강 파라미터 저장
            self.crop_config = aug_config.get('crop', {})
            self.noise_config = aug_config.get('noise', {})
            self.flip_config = aug_config.get('flip', {})
        else:
            # 증강 비활성화 시 모든 개별 증강도 비활성화
            self.augment_count = 0
            self.crop_enabled = False
            self.noise_enabled = False
            self.flip_enabled = False

        print(f"[{mode.upper()}] 데이터 증강: {'활성화' if self.augment_enabled else '비활성화'}")
        if self.augment_enabled:
            print(f"  - 증강 샘플 수: {self.augment_count}개/원본")
            print(f"  - 크롭 증강: {'ON' if self.crop_enabled else 'OFF'}")
            print(f"  - 노이즈 증강: {'ON' if self.noise_enabled else 'OFF'}")
```

##### 3. 증강 제어 계층 구조
```
augmentation.enabled (마스터 스위치)
    ├── crop.enabled (개별 제어)
    ├── noise.enabled (개별 제어)
    ├── flip.enabled (개별 제어)
    └── rotation.enabled (개별 제어)
```

**제어 로직:**
1. `augmentation.enabled = false` → 모든 증강 비활성화
2. `augmentation.enabled = true` → 개별 증강 설정에 따라 적용
3. `mode != 'train'` → 모든 증강 자동 비활성화 (val/test 모드)

##### 4. 사용 시나리오별 설정 예시

**빠른 디버깅 (증강 없이):**
```yaml
augmentation:
  enabled: false  # 모든 증강 비활성화
```

**기본 학습 (모든 증강 활성화):**
```yaml
augmentation:
  enabled: true
  augment_count: 3
  crop:
    enabled: true
  noise:
    enabled: true
```

**선택적 증강 (크롭만 적용):**
```yaml
augmentation:
  enabled: true
  augment_count: 2
  crop:
    enabled: true
  noise:
    enabled: false
```

**최소 증강 (테스트용):**
```yaml
augmentation:
  enabled: true
  augment_count: 1
  crop:
    enabled: false
  noise:
    enabled: true
    std: 0.005  # 작은 노이즈만
```

#### 구현 우선순위
1. **Phase 1**: 마스터 스위치 구현 (`augmentation.enabled`)
2. **Phase 2**: 기존 증강 방법 개별 제어 (crop, noise)
3. **Phase 3**: 새로운 증강 방법 추가 (flip, rotation, color)

#### 장점
1. **유연성**: 증강 방법별 세밀한 제어 가능
2. **편의성**: 마스터 스위치로 전체 증강 간편 제어
3. **실험 용이**: 다양한 증강 조합 실험 가능
4. **성능 최적화**: 불필요한 증강 제거로 학습 속도 향상
5. **재현성**: 설정 파일로 증강 조건 완벽 재현