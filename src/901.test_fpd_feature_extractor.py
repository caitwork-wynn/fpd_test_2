#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FPD Feature Extractor 테스트 스크립트
입력 이미지를 사용하여 특징 추출 테스트 (패치 기반 모드)
"""

import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_defs.fpd_feature_extractor import FPDFeatureExtractor


def test_fpd_extractor(image_path, config_path='config_fpd.yml'):
    """
    FPD Feature Extractor 테스트 (패치 기반 모드)
    
    Args:
        image_path: 테스트할 이미지 경로
        config_path: 설정 파일 경로
    """
    print("="*70)
    print("FPD Feature Extractor 테스트 (패치 기반)")
    print("="*70)
    
    # 1. 이미지 로드
    print(f"\n1. 이미지 로드: {image_path}")
    if not os.path.exists(image_path):
        print(f"ERROR: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: 이미지를 읽을 수 없습니다: {image_path}")
        return
    
    orig_h, orig_w = image.shape[:2]
    print(f"   - 원본 크기: {orig_w} x {orig_h}")
    print(f"   - 채널: {image.shape[2] if len(image.shape) == 3 else 1}")
    
    # 2. 이미지 리사이즈 (112x112)
    print(f"\n2. 이미지 리사이즈")
    image_resized = cv2.resize(image, (112, 112))
    print(f"   - 리사이즈 크기: 112 x 112")
    print(f"   - 원본 종횡비: {orig_w/orig_h:.3f}")
    
    # 3. Feature Extractor 생성
    print(f"\n3. Feature Extractor 초기화")
    extractor = FPDFeatureExtractor(config_path)
    
    # 4. 특징 추출
    print("\n" + "="*70)
    print("4. 특징 추출 실행")
    print("="*70)
    
    features = extractor.get_features(image_resized, orig_size=(orig_w, orig_h))
    
    print("\n추출된 특징:")
    
    # 전역 특징
    if 'global_features' in features:
        global_feat = features['global_features']
        print(f"\n   [1] 전역 특징 (전체 이미지 레벨):")
        print(f"       - Shape: {global_feat.shape}")
        print(f"       - 차원: {global_feat.shape[0]}")
        print(f"       - 구성: 기본정보(3) + 색상히스토(12) + HSV통계(6) = 21차원")
        print(f"       - Min: {global_feat.min():.6f}")
        print(f"       - Max: {global_feat.max():.6f}")
        print(f"       - Mean: {global_feat.mean():.6f}")
        print(f"       - Std: {global_feat.std():.6f}")
        
        # 세부 정보
        print(f"\n       세부 값:")
        print(f"       - 너비 정규화: {global_feat[0]:.3f}")
        print(f"       - 높이 정규화: {global_feat[1]:.3f}")
        print(f"       - 종횡비: {global_feat[2]:.3f}")
    
    # 패치별 OpenCV 특징
    if 'patch_opencv_features' in features:
        patch_opencv = features['patch_opencv_features']
        print(f"\n   [2] 패치별 OpenCV 특징 (7x7 그리드):")
        print(f"       - Shape: {patch_opencv.shape}")
        print(f"       - 49개 패치 x 21차원 특징")
        print(f"       - 각 패치 크기: 16x16 픽셀")
        print(f"       - 특징 구성: 픽셀통계(2) + 엣지(2) + RGB(3) + HSV(6) + ORB(8)")
        print(f"       - Min: {patch_opencv.min():.6f}")
        print(f"       - Max: {patch_opencv.max():.6f}")
        print(f"       - Mean: {patch_opencv.mean():.6f}")
        print(f"       - Std: {patch_opencv.std():.6f}")
        print(f"       - 0이 아닌 값: {np.count_nonzero(patch_opencv)}/{patch_opencv.size}")
    
    # 패치별 Latent 특징
    if 'patch_latent_features' in features:
        latent = features['patch_latent_features']
        print(f"\n   [3] 패치별 Latent 특징 (Autoencoder):")
        print(f"       - Shape: {latent.shape}")
        print(f"       - 49개 패치 x 16차원 latent")
        print(f"       - Min: {latent.min():.6f}")
        print(f"       - Max: {latent.max():.6f}")
        print(f"       - Mean: {latent.mean():.6f}")
        print(f"       - Std: {latent.std():.6f}")
        if latent.mean() == 0:
            print(f"       ⚠️  주의: Latent 값이 모두 0입니다. Encoder 가중치를 확인하세요.")
    
    # 5. 결합된 특징
    print("\n" + "="*70)
    print("5. 결합된 특징 벡터")
    print("="*70)
    
    combined = extractor.get_combined_features(image_resized, orig_size=(orig_w, orig_h))
    print(f"\n   총 특징 차원: {combined.shape[0]}")
    print(f"   구성:")
    print(f"   - 전역 특징: 21차원")
    print(f"   - 패치 OpenCV: 49 x 21 = 1029차원")
    print(f"   - 패치 Latent: 49 x 16 = 784차원")
    print(f"   - 합계: 21 + 1029 + 784 = 1834차원")
    print(f"\n   통계:")
    print(f"   - Min: {combined.min():.6f}")
    print(f"   - Max: {combined.max():.6f}")
    print(f"   - Mean: {combined.mean():.6f}")
    print(f"   - Std: {combined.std():.6f}")
    
    # 6. 패치 정보
    print("\n" + "="*70)
    print("6. 16x16 패치 분할 상세 정보")
    print("="*70)
    
    print(f"\n   112x112 이미지 → 7x7 그리드 분할:")
    print(f"   - 각 패치 크기: 16x16 픽셀")
    print(f"   - 총 패치 수: 49개")
    print(f"   - 패치 인덱스: [y,x] 형식 (0~6, 0~6)")
    
    # 일부 패치의 특징 예시
    if 'patch_opencv_features' in features:
        print(f"\n   패치별 특징 예시 (처음 5개):")
        for i in range(min(5, 49)):
            y, x = i // 7, i % 7
            opencv_mean = patch_opencv[i].mean()
            latent_mean = latent[i].mean() if 'patch_latent_features' in features else 0
            print(f"     패치[{y},{x}]: OpenCV mean={opencv_mean:.4f}, Latent mean={latent_mean:.4f}")
    
    # 7. 특징 분포 분석
    print("\n" + "="*70)
    print("7. 특징 분포 분석")
    print("="*70)
    
    # OpenCV 특징 분포
    if 'patch_opencv_features' in features:
        print(f"\n   OpenCV 특징 분포 (패치별 평균):")
        patch_means = [patch_opencv[i].mean() for i in range(49)]
        print(f"   - 최소 평균: {min(patch_means):.4f}")
        print(f"   - 최대 평균: {max(patch_means):.4f}")
        print(f"   - 전체 평균의 평균: {np.mean(patch_means):.4f}")
        
        # 가장 활성화된 패치
        max_idx = np.argmax(patch_means)
        max_y, max_x = max_idx // 7, max_idx % 7
        print(f"   - 가장 활성화된 패치: [{max_y},{max_x}] (평균: {patch_means[max_idx]:.4f})")
    
    print("\n" + "="*70)
    print("테스트 완료!")
    print("="*70)
    
    return {
        'features': features,
        'combined': combined,
        'image_size': (orig_w, orig_h)
    }


if __name__ == "__main__":
    # 테스트 이미지 경로
    test_image_path = "../data/learning/complete01-00015_005_210_alpha100_01.png"
    
    # 절대 경로로 변환
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(script_dir, test_image_path)
    test_image_path = os.path.normpath(test_image_path)
    
    print(f"테스트 이미지: {test_image_path}")
    
    # 테스트 실행
    results = test_fpd_extractor(test_image_path, config_path='config_fpd.yml')
    
    # 결과 저장 정보
    if results:
        print(f"\n💾 결과가 results 변수에 저장되었습니다:")
        print(f"   - results['features']: 추출된 모든 특징")
        print(f"   - results['combined']: 결합된 특징 벡터 (1834차원)")
        print(f"   - results['image_size']: 원본 이미지 크기")