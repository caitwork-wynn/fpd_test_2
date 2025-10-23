#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
199.verify_learning_data.py
학습 데이터 검증 도구

config.yml의 source_folder 설정을 사용하여
labels.txt를 읽고 랜덤하게 40개의 이미지를 선택하고 floor 위치를 시각화합니다.
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent

def load_config():
    """config.yml 파일을 로드합니다."""
    config_path = Path(__file__).parent / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_labels_data(config):
    """labels.txt 파일을 읽어 데이터를 로드합니다."""
    # config.yml에서 source_folder와 labels_file 읽기
    source_folder = config['data']['source_folder']
    labels_filename = config['data']['labels_file']

    # 상대 경로 처리
    base_dir = Path(__file__).parent
    data_path = (base_dir / source_folder).resolve()
    labels_file = data_path / labels_filename

    print(f"데이터 폴더: {data_path}")
    print(f"라벨 파일: {labels_file}")

    if not labels_file.exists():
        print(f"오류: {labels_file} 파일이 존재하지 않습니다.")
        sys.exit(1)

    data = []
    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # 헤더 건너뛰기
        header = lines[0].strip().split(',')
        print(f"헤더: {header}")

        # 데이터 파싱
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                entry = {
                    'id': parts[0],
                    'opacity': float(parts[1]),
                    'filename': parts[2],
                    'center_x': float(parts[3]),
                    'center_y': float(parts[4]),
                    'floor_x': float(parts[5]),
                    'floor_y': float(parts[6])
                }
                data.append(entry)

    return data, data_path

def verify_and_visualize_data(data, data_path, num_samples=40):
    """랜덤하게 샘플을 선택하고 floor 위치를 시각화합니다."""
    # 이미지가 존재하는 데이터만 필터링
    valid_data = []

    for entry in data:
        img_path = data_path / entry['filename']
        if img_path.exists():
            valid_data.append(entry)

    print(f"전체 데이터: {len(data)}개")
    print(f"유효한 이미지: {len(valid_data)}개")

    if len(valid_data) == 0:
        print("오류: 유효한 이미지가 없습니다.")
        sys.exit(1)

    # 랜덤 샘플링
    num_samples = min(num_samples, len(valid_data))
    sampled_data = random.sample(valid_data, num_samples)

    # 시각화 설정
    cols = 8
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
    fig.suptitle('Learning Data Verification - Floor Positions', fontsize=16)

    # axes를 2D 배열로 변환
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, entry in enumerate(sampled_data):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # 이미지 로드
        img_path = data_path / entry['filename']
        img = cv2.imread(str(img_path))

        if img is not None:
            # BGR to RGB 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 이미지 표시
            ax.imshow(img)

            # floor 위치에 원 그리기
            floor_x = entry['floor_x']
            floor_y = entry['floor_y']

            # 빨간색 원으로 floor 위치 표시
            circle = plt.Circle((floor_x, floor_y), 5, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)

            # 십자선 추가
            ax.plot([floor_x-10, floor_x+10], [floor_y, floor_y], 'r-', linewidth=1)
            ax.plot([floor_x, floor_x], [floor_y-10, floor_y+10], 'r-', linewidth=1)

            # 타이틀 설정 (ID만 표시)
            ax.set_title(entry['id'][:20], fontsize=8)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"Failed to load\n{entry['filename']}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    # 남은 서브플롯 숨기기
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    # 레이아웃 조정
    plt.tight_layout()

    # 결과 저장
    output_dir = ROOT_DIR / "result"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "learning_floor.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n결과 이미지 저장 완료: {output_path}")

    # 통계 출력
    print(f"\n=== Floor 위치 통계 ===")
    floor_xs = [e['floor_x'] for e in sampled_data]
    floor_ys = [e['floor_y'] for e in sampled_data]

    print(f"Floor X: 최소={min(floor_xs):.1f}, 최대={max(floor_xs):.1f}, 평균={np.mean(floor_xs):.1f}")
    print(f"Floor Y: 최소={min(floor_ys):.1f}, 최대={max(floor_ys):.1f}, 평균={np.mean(floor_ys):.1f}")

    plt.show()

def main():
    """메인 함수"""
    print("=== 학습 데이터 검증 도구 ===\n")

    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(42)

    # config.yml 로드
    print("config.yml 파일 로드 중...")
    config = load_config()

    # 데이터 로드
    print("labels.txt 파일 로드 중...")
    data, data_path = load_labels_data(config)

    # 시각화 및 검증
    verify_and_visualize_data(data, data_path, num_samples=40)

    print("\n프로그램 완료!")

if __name__ == "__main__":
    main()