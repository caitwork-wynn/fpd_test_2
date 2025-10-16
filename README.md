# AI 포인트 검출 학습 시스템

PyTorch 기반의 이미지 다중 포인트 검출 딥러닝 학습 시스템입니다.

## Git 저장소

```bash
# 프로젝트 클론
git clone https://github.com/caitwork-wynn/fpd_test_2.git
cd fpd_test_2
```

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
│   ├── 200.learning.py                        # 메인 학습 스크립트
│   ├── 299.inference_pretrained_model.py      # 학습된 모델 추론 스크립트
│   ├── config.yml                             # 학습 설정 파일
│   │
│   ├── model_defs/                            # 모델 정의 모듈
│   │   ├── floor_model_attention.py           # Floor 전용 Attention 모델
│   │   ├── multi_point_model_attention.py     # 다중 포인트 Attention 모델
│   │   ├── multi_point_model_ae.py            # Autoencoder 기반 모델
│   │   ├── multi_point_model_kornia.py        # Kornia 기반 모델
│   │   ├── multi_point_model_pytorch.py       # PyTorch 기본 모델
│   │   ├── fpd_feature_extractor.py           # FPD 특징 추출기
│   │   ├── fpd_coordinate_regression.py       # FPD 좌표 회귀 모델
│   │   ├── fpd_mix_ae_position_embedding.py   # FPD + AE + Position Embedding
│   │   ├── autoencoder_7x7.py                 # 7x7 Autoencoder
│   │   └── autoencoder_16x16.py               # 16x16 Autoencoder
│   │
│   ├── util/                                  # 유틸리티 모듈
│   │   ├── dual_logger.py                     # 로깅 유틸
│   │   ├── save_load_model.py                 # 모델 저장/로드
│   │   ├── error_analysis.py                  # 오차 분석
│   │   └── data_augmentation.py               # 데이터 증강
│   │
│   ├── 100.merge_learning_data.py             # 학습 데이터 병합
│   ├── 110.merge_learning_lc_data.py          # LC 데이터 병합
│   ├── 120.merge_learning_db_data.py          # DB 데이터 병합
│   ├── 180.check_labels.py                    # 레이블 검증
│   ├── 190.view_labeling_data.py              # 레이블 데이터 시각화
│   ├── 199.verify_learning_data_floor.py      # Floor 데이터 검증
│   ├── 700.make_7x7_autoencoder.py            # 7x7 Autoencoder 생성
│   ├── 710.make_16x16_autoencoder.py          # 16x16 Autoencoder 생성
│   ├── 901.test_fpd_feature_extractor.py      # FPD 특징 추출기 테스트
│   └── 902.test_fpd_coordinate_regression.py  # FPD 좌표 회귀 테스트
│
├── data/
│   └── learning/                              # 학습 데이터 폴더
│       └── labels.txt                         # 레이블 파일
├── model/                                     # 저장된 모델 체크포인트
├── logs/                                      # 학습 로그 파일
├── result/                                    # 오차 분석 결과
└── requirements.txt                           # Python 패키지 의존성
```

## 설치 및 환경 설정

### 1. 저장소 클론

```bash
git clone https://github.com/caitwork-wynn/fpd_test_2.git
cd fpd_test_2
```

### 2. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

주요 패키지:
- PyTorch >= 2.0.0 (딥러닝 프레임워크)
- torchvision >= 0.15.0 (컴퓨터 비전 라이브러리)
- Kornia >= 0.7.0 (미분 가능한 컴퓨터 비전 라이브러리)
- OpenCV >= 4.8.0 (이미지 처리)
- PyYAML >= 6.0 (설정 파일 파싱)
- ONNX >= 1.14.0 (모델 변환, 선택사항)

### 3. GPU 지원

CUDA가 설치된 환경에서는 자동으로 GPU를 사용합니다. CPU 전용 환경에서도 학습 가능합니다.

## 빠른 시작

### 1. 데이터 준비

학습 데이터는 `data/learning/` 폴더에 위치해야 하며, `labels.txt` 파일에 다음 형식으로 작성:

```
ID,SRC,FILE_NAME,CENTER_X,CENTER_Y,FLOOR_X,FLOOR_Y,FRONT_X,FRONT_Y,SIDE_X,SIDE_Y
001,source1,image001.jpg,56,112,56,100,45,80,70,90
002,source2,image002.jpg,60,115,60,105,50,85,75,95
...
```

**데이터 제한 옵션**: `config.yml`에서 `max_train_images` 설정으로 학습 데이터 수를 제한할 수 있습니다.
```yaml
data:
  max_train_images: 200  # 0이면 전체 사용, 200이면 200개만 사용
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

  # 아키텍처 설정
  architecture:
    use_fpd_architecture: true  # FPD 분류 기반 회귀 방식 사용
    features:
      image_size: [112, 112]
      grid_size: 7
      use_autoencoder: false  # Autoencoder 기반 특성 추출
      encoder_path: '../model/autoencoder_16x16_best.pth'
```

사용 가능한 모델:
- `floor_model_attention.py`: Floor 전용 Attention 모델
- `multi_point_model_attention.py`: 다중 포인트 Attention 모델
- `multi_point_model_ae.py`: Autoencoder 기반 모델
- `multi_point_model_kornia.py`: Kornia 기반 모델
- `multi_point_model_pytorch.py`: PyTorch 기본 모델

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

## 추가 유틸리티 스크립트

### 데이터 관리

- `100.merge_learning_data.py`: 여러 소스의 학습 데이터를 병합
- `110.merge_learning_lc_data.py`: LC 데이터를 학습 데이터로 병합
- `120.merge_learning_db_data.py`: DB 데이터를 학습 데이터로 병합
- `180.check_labels.py`: 레이블 파일 검증 및 오류 확인
- `190.view_labeling_data.py`: 레이블 데이터 시각화 도구
- `199.verify_learning_data_floor.py`: Floor 포인트 데이터 검증

### 모델 테스트 및 생성

- `299.inference_pretrained_model.py`: 학습된 모델로 추론 수행
- `700.make_7x7_autoencoder.py`: 7x7 그리드 Autoencoder 생성
- `710.make_16x16_autoencoder.py`: 16x16 그리드 Autoencoder 생성
- `901.test_fpd_feature_extractor.py`: FPD 특징 추출기 단위 테스트
- `902.test_fpd_coordinate_regression.py`: FPD 좌표 회귀 모델 단위 테스트

### 사용 예시

```bash
cd src

# 레이블 파일 검증
python 180.check_labels.py

# 레이블 데이터 시각화
python 190.view_labeling_data.py

# 학습된 모델로 추론
python 299.inference_pretrained_model.py
```

## 최근 업데이트

- **2024-10-02**: 학습 스크립트 및 설정 파일 정리
- **2024-10-01**: AI 포인트 검출 학습 시스템 문서화 완료
- **2024-10-01**: Best 모델 저장 시 상세 정보 파일 생성 기능 추가
- **2024-09-26**: ONNX 변환 문제 해결 - 딕셔너리 출력 모델 지원 추가

## 라이선스

이 프로젝트는 주식회사 캐이트워크에 귀속됩니다.
