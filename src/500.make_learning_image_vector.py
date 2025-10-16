#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
500.make_learning_image_vector.py
data/learning 폴더 이미지를 ae_16x16_best.pth로 인코딩하여 latent 벡터 생성

- 각 이미지를 112x112 그레이스케일로 변환
- 7x7 그리드로 16x16 패치 49개 생성
- ae_16x16_best.pth 인코더로 각 패치를 latent(16차원)로 변환
- 49개의 latent를 (49, 16) 배열로 저장 (이미지명.dat)
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List

# 모델 import
sys.path.append(str(Path(__file__).parent))
from model_defs.autoencoder_16x16 import SeparatedResidualAutoencoder16x16


class ImageVectorEncoder:
    """이미지를 latent vector로 변환하는 인코더"""

    def __init__(self, model_path: str, device: torch.device):
        """
        Args:
            model_path: 학습된 autoencoder 모델 경로
            device: 연산 디바이스
        """
        self.device = device
        self.model = self.load_model(model_path)

        # 이미지 처리 파라미터
        self.image_size = 112  # 112x112로 리사이즈
        self.patch_size = 16   # 각 패치 크기
        self.grid_size = 7     # 7x7 그리드

    def load_model(self, model_path: str) -> SeparatedResidualAutoencoder16x16:
        """모델 로드"""
        print(f"\n모델 로딩: {model_path}")

        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 설정 추출 (없으면 기본값 사용)
        if 'config' in checkpoint:
            config = checkpoint['config']
            input_channels = config['model']['input_channels']
            latent_dim = config['model']['latent_dim']
        else:
            # 기본값
            input_channels = 1
            latent_dim = 16

        print(f"  - Input channels: {input_channels}")
        print(f"  - Latent dim: {latent_dim}")

        # 모델 생성
        model = SeparatedResidualAutoencoder16x16(
            input_channels=input_channels,
            output_channels=input_channels,
            latent_dim=latent_dim
        ).to(self.device)

        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("모델 로드 완료")
        return model

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        이미지를 전처리하여 16x16 패치 배열로 변환

        Args:
            image_path: 이미지 파일 경로

        Returns:
            patches: (49, 16, 16) numpy 배열 (그레이스케일)
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")

        # 그레이스케일 변환
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 112x112로 리사이즈
        resized = cv2.resize(image, (self.image_size, self.image_size))

        # 7x7 그리드로 16x16 패치 추출
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y_start = i * self.patch_size
                x_start = j * self.patch_size

                patch = resized[y_start:y_start+self.patch_size,
                               x_start:x_start+self.patch_size]
                patches.append(patch)

        # (49, 16, 16) 배열로 변환
        patches = np.array(patches, dtype=np.float32)

        # 정규화 (0-1)
        patches = patches / 255.0

        return patches

    def encode_patches(self, patches: np.ndarray) -> np.ndarray:
        """
        패치들을 latent vector로 인코딩

        Args:
            patches: (49, 16, 16) numpy 배열

        Returns:
            latents: (49, 16) numpy 배열
        """
        # (49, 1, 16, 16) 형태로 변환 (채널 차원 추가)
        patches_with_channel = np.expand_dims(patches, axis=1)

        # torch tensor로 변환
        patches_tensor = torch.FloatTensor(patches_with_channel).to(self.device)

        # 인코딩
        with torch.no_grad():
            latents = self.model.get_latent_representation(patches_tensor)

        # numpy로 변환
        latents_np = latents.cpu().numpy()

        return latents_np

    def process_image(self, image_path: str) -> np.ndarray:
        """
        이미지를 처리하여 latent vector로 변환

        Args:
            image_path: 이미지 파일 경로

        Returns:
            latents: (49, 16) numpy 배열
        """
        # 전처리
        patches = self.preprocess_image(image_path)

        # 인코딩
        latents = self.encode_patches(patches)

        return latents


def main():
    """메인 함수"""
    print("=" * 60)
    print("Learning Image Vector 생성")
    print("data/learning 폴더 이미지를 latent vector로 변환")
    print("=" * 60)

    # 경로 설정
    src_dir = Path(__file__).parent
    base_dir = src_dir.parent

    learning_dir = base_dir / "data" / "learning"
    model_path = base_dir / "model" / "ae_16x16_best.pth"

    # 경로 확인
    if not learning_dir.exists():
        print(f"오류: {learning_dir} 폴더가 존재하지 않습니다")
        return

    if not model_path.exists():
        print(f"오류: {model_path} 모델 파일이 존재하지 않습니다")
        return

    print(f"\n입력 폴더: {learning_dir}")
    print(f"모델 파일: {model_path}")

    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"\n사용 디바이스: GPU - {torch.cuda.get_device_name(device)}")
    else:
        print(f"\n사용 디바이스: CPU")

    # 인코더 초기화
    encoder = ImageVectorEncoder(str(model_path), device)

    # 이미지 파일 목록
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(learning_dir.glob(ext)))

    # labels.txt 제외
    image_files = [f for f in image_files if f.stem != 'labels']

    print(f"\n처리할 이미지 수: {len(image_files)}")

    if len(image_files) == 0:
        print("처리할 이미지가 없습니다")
        return

    # 모델 검증 (첫 이미지로 테스트)
    print("\n=== 모델 검증 ===")
    test_image = image_files[0]
    print(f"테스트 이미지: {test_image.name}")

    # 원본 이미지 정보
    test_img = cv2.imread(str(test_image))
    print(f"원본 이미지 크기: {test_img.shape}")

    # 전처리 후 정보
    test_patches = encoder.preprocess_image(str(test_image))
    print(f"전처리 후 패치: {test_patches.shape} (49개의 16x16 패치)")
    print(f"패치 값 범위: [{test_patches.min():.3f}, {test_patches.max():.3f}]")

    # 인코딩 테스트
    test_latents = encoder.encode_patches(test_patches)
    print(f"인코딩 결과: {test_latents.shape} (49개의 16차원 latent)")
    print(f"Latent 값 범위: [{test_latents.min():.3f}, {test_latents.max():.3f}]")
    print("=" * 60)

    # 모든 이미지 처리
    print(f"\n이미지 인코딩 시작...")
    success_count = 0
    error_count = 0

    for image_path in tqdm(image_files, desc="Processing"):
        try:
            # 인코딩
            latents = encoder.process_image(str(image_path))

            # 저장 경로 (같은 폴더에 .dat 확장자로 저장)
            output_path = image_path.with_suffix('.dat')

            # NumPy 바이너리 형식으로 저장
            np.save(str(output_path), latents)

            success_count += 1

        except Exception as e:
            print(f"\n오류 발생: {image_path.name} - {str(e)}")
            error_count += 1
            continue

    # 결과 출력
    print("\n" + "=" * 60)
    print("처리 완료!")
    print(f"성공: {success_count}개")
    print(f"실패: {error_count}개")
    print(f"출력 형식: (49, 16) numpy 배열")
    print(f"저장 위치: {learning_dir}")
    print(f"파일 형식: 이미지명.dat (NumPy binary)")
    print("=" * 60)

    # 저장된 파일 예시
    if success_count > 0:
        sample_file = list(learning_dir.glob('*.dat'))[0]
        sample_data = np.load(str(sample_file))
        print(f"\n저장 예시: {sample_file.name}")
        print(f"  - Shape: {sample_data.shape}")
        print(f"  - Dtype: {sample_data.dtype}")
        print(f"  - Size: {sample_file.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
