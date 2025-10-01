# AI 포인트 검출 학습 시스템

PyTorch 기반의 이미지 다중 포인트 검출 딥러닝 학습 시스템입니다.

## 시스템 개요

이 프로젝트는 이미지에서 다중 포인트(center, floor, front, side)를 검출하는 AI 모델을 학습하기 위한 범용 학습 시스템입니다.
현재는 객체의 위치를 분석하기 위한 바닥 중심점 (floor)를 추출하는 목적으로 사용합니다.
`config.yml` 파일을 통해 학습할 모델과 파라미터를 동적으로 설정할 수 있으며, 다양한 아키텍처를 지원합니다.

### 주요 특징

- **동적 모델 로딩**: `config.yml`의 `learning_model.source` 설정으로 모델 선택
- **타겟 포인트 선택**: 전체 포인트 또는 특정 포인트(예: floor만) 학습 가능
- **특징 사전 추출**: 학습 속도 10-20배 향상 옵션
- **데이터 증강**: 크롭 증강으로 모델 일반화 성능 향상
- **체크포인트 관리**: 자동 저장 및 재개, ONNX 변환
- **상세한 오차 분석**: 에폭별 픽셀 단위 오차 통계

## 프로젝트 구조

```
fpd_only_model/
├── src/
│   ├── 200.learning.py              # 메인 학습 스크립트
│   ├── config.yml                   # 학습 설정 파일
│   ├── model_defs/                  # 모델 정의 모듈
│   │   ├── floor_model_attention.py # Floor 전용 Attention 모델
│   │   └── autoencoder_*.py         # Autoencoder 모델들
│   └── util/                        # 유틸리티 모듈
│       ├── dual_logger.py           # 로깅 유틸
│       ├── save_load_model.py       # 모델 저장/로드
│       ├── error_analysis.py        # 오차 분석
│       └── data_augmentation.py     # 데이터 증강
├── data/
│   └── learning/                    # 학습 데이터 폴더
│       └── labels.txt               # 레이블 파일
├── model/                           # 저장된 모델 체크포인트
├── logs/                            # 학습 로그 파일
└── result/                          # 오차 분석 결과
```

## 설치 및 환경 설정

### 필수 패키지

```bash
pip install -r requirements.txt
```

### GPU 지원

CUDA가 설치된 환경에서는 자동으로 GPU를 사용합니다.

## 빠른 시작

### 1. 데이터 준비

학습 데이터는 `data/learning/` 폴더에 위치해야 하며, `labels.txt` 파일에 다음 형식으로 작성:

```
ID,SRC,FILE_NAME,CENTER_X,CENTER_Y,FLOOR_X,FLOOR_Y,FRONT_X,FRONT_Y,SIDE_X,SIDE_Y
001,source1,image001.jpg,56,112,56,100,45,80,70,90
002,source2,image002.jpg,60,115,60,105,50,85,75,95
...
```

### ⚠️ 중요: 데이터 분할 규칙

시스템은 **ID 끝자리**를 기준으로 자동으로 데이터를 분할합니다.

#### 1단계: Train/Val vs Test 분리

**ID 끝자리가 `test_id_suffix`(기본값: '1')로 끝나는 데이터 → Test 세트**

```
예시) test_id_suffix: '1' (기본값)

ID: 001 → Test     (끝자리 1)
ID: 002 → Train/Val (끝자리 2)
ID: 011 → Test     (끝자리 1)
ID: 021 → Test     (끝자리 1)
ID: 100 → Train/Val (끝자리 0)
ID: 1234561 → Test (끝자리 1)
```

#### 2단계: Train vs Validation 분리

**Test를 제외한 나머지 데이터를 `validation_ratio`(기본값: 0.2)로 분할**


```
예시) 전체 150개 데이터, validation_ratio: 0.2

1. Test 세트: ID 끝자리 1 → 15개
   (001, 011, 021, 031, ..., 141)

2. Train/Val 후보: 나머지 135개
   (002, 003, ..., 142, 143, ...)

3. 랜덤 섞기 후 분할:
   - Validation: 135 × 0.2 = 27개 (20%)
   - Train: 135 - 27 = 108개 (80%)
```

**💡 팁**:
- Test 세트 비율을 조정하려면 ID 끝자리 규칙을 변경하세요
  - `test_id_suffix: '1'` → 10% Test (001, 011, 021, ...)
  - `test_id_suffix: '0'` → 10% Test (010, 020, 030, ...)
  - ID를 '01', '11', '21'로 끝나게 → 10% Test
- Validation 비율은 `validation_ratio`로 조정
  - `0.1` → 10% Validation
  - `0.2` → 20% Validation (권장)
  - `0.3` → 30% Validation

### 2. 학습 설정

`src/config.yml` 파일을 편집하여 학습 설정을 변경합니다:

```yaml
learning_model:
  source: 'model_defs/floor_model_attention.py'  # 사용할 모델
  target_points: ['floor']  # 학습할 포인트 선택

training:
  batch_size: 512
  learning_rate: 0.001
  optimizer: 'adamw'  # adam, adamw, sgd
  extract_features: true  # 특징 사전 추출 (학습 속도 10-20배 향상)

  # 데이터 증강
  augmentation:
    enabled: true
    augment_count: 4  # 원본당 증강 샘플 수
```

### 3. 학습 실행

```bash
cd src
python 200.learning.py
```

학습은 자동으로 체크포인트에서 재개됩니다:
- 최신 epoch 파일 우선 로드
- 없으면 best 모델 로드
- 둘 다 없으면 새로운 학습 시작

## 주요 설정

### 모델 선택

```yaml
learning_model:
  source: 'model_defs/floor_model_attention.py'
  target_points: ['floor']  # ['center', 'floor', 'front', 'side']
```

### 학습 파라미터

```yaml
training:
  batch_size: 512              # 배치 크기
  epochs: 100000000            # 최대 에폭 수
  learning_rate: 0.001         # 학습률
  weight_decay: 0.01           # Weight decay
  optimizer: 'adamw'           # 옵티마이저
  gradient_clip: 5.0           # Gradient clipping
  extract_features: true       # 특징 사전 추출
```

### 학습률 스케줄러

```yaml
training:
  scheduler:
    enabled: false             # 스케줄러 활성화
    type: 'step'              # reduce_on_plateau, step, cosine, none

    # 시간 기반 학습률 감소
    time_based:
      enabled: true
      patience_hours: 0.3     # Best 갱신 없이 대기할 시간
      factor: 0.7             # 감소 비율

    min_lr: 0.0000001         # 최소 학습률
```

### 데이터 증강

```yaml
training:
  augmentation:
    enabled: true
    augment_count: 4          # 원본당 증강 샘플 수
    crop:
      enabled: true
      min_ratio: 0.8          # 최소 크롭 비율
      max_ratio: 1.0          # 최대 크롭 비율
      max_shift: 0.15         # 최대 이동 비율
```

### 오차 분석

```yaml
training:
  error_analysis:
    enabled: true             # 오차 분석 활성화
    interval: 1000            # 분석 간격 (에폭)
    save_raw_data: true       # 원천 데이터 저장
    results_dir: '../result'  # 결과 저장 경로
```

## 출력 파일

### 모델 파일

- `../model/{save_file_name}_best.pth`: 최고 성능 모델 (PyTorch)
- `../model/{save_file_name}_best.onnx`: ONNX 변환 모델
- `../model/{save_file_name}/{save_file_name}_epoch{N}.pth`: 주기적 체크포인트

### 로그 파일

- `../logs/{save_file_name}_{timestamp}.log`: 전체 학습 로그
- `../logs/{save_file_name}_best.log`: Best 모델 갱신 기록

### 결과 파일

- `../result/{save_file_name}/training_log.csv`: 에폭별 손실/오차 기록
- `../result/{save_file_name}/error_epoch_{N}.json`: 에폭별 상세 오차
- `../result/{save_file_name}/error_final.json`: 최종 테스트 오차
- `../result/{save_file_name}/model_info.json`: 모델 메타정보
- `../result/{save_file_name}/best_epoch_{N}.json`: Best 모델 정보

## 라이선스

이 프로젝트는 주식회사 캐이트워크에 귀속됩니다.
