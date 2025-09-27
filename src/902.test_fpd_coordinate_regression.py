#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FPD Coordinate Regression 테스트 스크립트
FPD 특징 추출 → 좌표 회귀 모델 테스트
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_defs.fpd_feature_extractor import FPDFeatureExtractor
from model_defs.fpd_coordinate_regression import FPDCoordinateRegression


def test_coordinate_regression(image_path, config_path='config_fpd.yml'):
    """
    FPD Coordinate Regression 테스트
    
    Args:
        image_path: 테스트할 이미지 경로
        config_path: 설정 파일 경로
    """
    print("="*70)
    print("FPD Coordinate Regression 테스트")
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
    
    # 2. 이미지 리사이즈
    print(f"\n2. 이미지 리사이즈: 112 x 112")
    image_resized = cv2.resize(image, (112, 112))
    
    # 3. Feature Extractor로 특징 추출
    print(f"\n3. FPD Feature Extractor로 특징 추출")
    extractor = FPDFeatureExtractor(config_path)
    features = extractor.get_features(image_resized, orig_size=(orig_w, orig_h))
    
    print("   추출된 특징:")
    print(f"   - 전역 특징: {features['global_features'].shape}")
    print(f"   - 패치 OpenCV: {features['patch_opencv_features'].shape}")
    print(f"   - 패치 Latent: {features['patch_latent_features'].shape}")
    
    # 4. Tensor로 변환
    print(f"\n4. PyTorch Tensor로 변환")
    features_tensor = {
        'global_features': torch.FloatTensor(features['global_features']).unsqueeze(0),
        'patch_opencv_features': torch.FloatTensor(features['patch_opencv_features']).unsqueeze(0),
        'patch_latent_features': torch.FloatTensor(features['patch_latent_features']).unsqueeze(0)
    }
    print(f"   - Batch 차원 추가: [1, ...]")
    
    # 5. Coordinate Regression 모델 생성
    print(f"\n5. FPD Coordinate Regression 모델 생성")
    model = FPDCoordinateRegression()
    model.eval()  # 평가 모드
    
    print(f"   모델 구조:")
    print(f"   - 입력: 1834차원 (21 + 1029 + 784)")
    print(f"   - 출력: 8개 좌표 (4쌍의 x,y)")
    print(f"   - x 범위: -112 ~ 228 (341 클래스)")
    print(f"   - y 범위: 0 ~ 228 (229 클래스)")
    
    # 6. Forward Pass
    print(f"\n" + "="*70)
    print("6. Forward Pass 실행")
    print("="*70)
    
    with torch.no_grad():
        output = model(features_tensor)
    
    print(f"\n출력 텐서:")
    print(f"   - logits: {output['logits'].shape}")
    print(f"   - coordinates: {output['coordinates'].shape}")
    print(f"   - coordinates_2d: {output['coordinates_2d'].shape}")
    
    # 7. 예측 좌표 출력
    coords = output['coordinates'].squeeze().numpy()  # [8]
    coords_2d = output['coordinates_2d'].squeeze().numpy()  # [4, 2]
    
    print(f"\n예측된 좌표 (8개 값):")
    for i in range(8):
        coord_type = 'x' if i % 2 == 0 else 'y'
        point_idx = i // 2 + 1
        print(f"   Point {point_idx} {coord_type}: {coords[i]:.2f}")
    
    print(f"\n예측된 2D 포인트 (4개):")
    for i in range(4):
        x, y = coords_2d[i]
        print(f"   Point {i+1}: ({x:.2f}, {y:.2f})")
    
    # 8. 손실 계산 테스트 (더미 타겟)
    print(f"\n" + "="*70)
    print("8. 손실 계산 테스트")
    print("="*70)
    
    # 더미 타겟 생성 (임의의 좌표)
    dummy_targets = torch.FloatTensor([
        50.5,   # x1
        100.3,  # y1
        -20.7,  # x2
        150.0,  # y2
        180.2,  # x3
        80.5,   # y3
        0.0,    # x4
        200.1   # y4
    ]).unsqueeze(0)  # [1, 8]
    
    print(f"\n더미 타겟 좌표:")
    for i in range(8):
        coord_type = 'x' if i % 2 == 0 else 'y'
        point_idx = i // 2 + 1
        print(f"   Point {point_idx} {coord_type}: {dummy_targets[0, i]:.2f}")
    
    # 손실 계산
    model.train()  # 학습 모드로 전환
    output = model(features_tensor)
    loss = model.compute_loss(output, dummy_targets, sigma=1.0)
    
    print(f"\n계산된 손실:")
    print(f"   - Cross-Entropy Loss: {loss.item():.6f}")
    
    # 9. Soft Label 분포 확인
    print(f"\n" + "="*70)
    print("9. Soft Label 분포 예시")
    print("="*70)
    
    # 첫 번째 x 좌표 헤드의 soft label
    head = model.coordinate_heads[0]
    soft_label = head.create_soft_label(dummy_targets[:, 0], sigma=1.0)
    
    print(f"\n타겟 x1 = {dummy_targets[0, 0]:.1f}에 대한 Soft Label:")
    
    # 가장 높은 확률을 가진 5개 클래스
    probs, indices = torch.topk(soft_label[0], k=5)
    for i, (prob, idx) in enumerate(zip(probs, indices)):
        class_value = head.class_centers[idx].item()
        print(f"   클래스 {idx:3d} (값={class_value:6.1f}): {prob:.4f}")
    
    # 10. 모델 파라미터 정보
    print(f"\n" + "="*70)
    print("10. 모델 파라미터 정보")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n파라미터 수:")
    print(f"   - 전체 파라미터: {total_params:,}")
    print(f"   - 학습 가능 파라미터: {trainable_params:,}")
    
    # 주요 모듈별 파라미터
    patch_params = sum(p.numel() for p in model.patch_encoder.parameters())
    fusion_params = sum(p.numel() for p in model.fusion_network.parameters())
    head_params = sum(p.numel() for p in model.coordinate_heads.parameters())
    
    print(f"\n모듈별 파라미터:")
    print(f"   - Patch Encoder: {patch_params:,}")
    print(f"   - Fusion Network: {fusion_params:,}")
    print(f"   - Coordinate Heads: {head_params:,}")
    
    # 11. 메모리 사용량
    print(f"\n" + "="*70)
    print("11. 메모리 사용량 추정")
    print("="*70)
    
    # 모델 크기 (MB)
    model_size = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"\n모델 크기: {model_size:.2f} MB")
    
    # 배치 크기별 메모리 추정
    for batch_size in [1, 8, 16, 32]:
        # 입력 특징
        input_size = batch_size * 1834 * 4 / (1024 * 1024)
        # 중간 활성화 (대략)
        activation_size = batch_size * 512 * 10 * 4 / (1024 * 1024)
        total_memory = model_size + input_size + activation_size
        print(f"   Batch {batch_size:2d}: ~{total_memory:.1f} MB")
    
    print(f"\n" + "="*70)
    print("테스트 완료!")
    print("="*70)
    
    return {
        'model': model,
        'features': features_tensor,
        'output': output,
        'loss': loss.item()
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
    results = test_coordinate_regression(test_image_path, config_path='config_fpd.yml')
    
    if results:
        print(f"\n[결과 저장]")
        print(f"   - results['model']: 회귀 모델")
        print(f"   - results['features']: 추출된 특징 (텐서)")
        print(f"   - results['output']: 모델 출력")
        print(f"   - results['loss']: 손실값")