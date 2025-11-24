# -*- coding: utf-8 -*-
"""
학습 진행 상태 그래프 생성 유틸리티
- matplotlib 기반 학습 진행 상태 시각화
- 동적 포인트 처리 (floor 등 현재 모델에서 사용하는 포인트만 표시)
"""

import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_training_progress(
    training_log: List[Dict],
    save_path: str,
    point_names: Optional[List[str]] = None,
    dpi: int = 150
) -> None:
    """
    학습 진행 상태 그래프 생성

    Parameters:
    -----------
    training_log : List[Dict]
        학습 로그 데이터 (각 에폭별 딕셔너리)
        예: [{'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.6, ...}, ...]
    save_path : str
        그래프 저장 경로 (.png)
    point_names : List[str], optional
        표시할 포인트 이름 리스트. None이면 자동 감지
        예: ['floor'], ['center', 'floor', 'front', 'side']
    dpi : int, optional
        그래프 해상도 (기본 150)
    """
    # 데이터 유효성 검사
    if not training_log:
        print("학습 로그가 비어있어 그래프를 생성할 수 없습니다.")
        return

    if len(training_log) < 2:
        print("그래프 생성에 충분한 데이터가 없습니다. (최소 2개 에폭 필요)")
        return

    # 데이터 추출
    epochs = [entry['epoch'] for entry in training_log]
    train_losses = [entry['train_loss'] for entry in training_log]
    val_losses = [entry['val_loss'] for entry in training_log]
    learning_rates = [entry['learning_rate'] for entry in training_log]
    avg_errors = [entry['avg_error'] for entry in training_log]

    # 포인트 이름 자동 감지
    if point_names is None:
        point_names = []
        for key in training_log[0].keys():
            if key.endswith('_error') and key != 'avg_error':
                point_name = key.replace('_error', '')
                point_names.append(point_name)

    # 0이 아닌 포인트만 필터링 (실제로 사용하는 포인트)
    active_points = []
    point_errors = {}
    for point_name in point_names:
        error_key = f'{point_name}_error'
        if error_key in training_log[0]:
            errors = [entry.get(error_key, 0) for entry in training_log]
            # 모든 값이 0이 아닌 경우만 활성 포인트로 간주
            if any(e > 0 for e in errors):
                active_points.append(point_name)
                point_errors[point_name] = errors

    # Figure 생성 (4개의 subplot)
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    fig.suptitle('학습 진행 상태', fontsize=16, fontweight='bold', y=0.995)

    # 1. Train/Val Loss 그래프
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'orange', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Train / Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 최소/최대 loss 표시
    min_val_loss = min(val_losses)
    min_val_epoch = epochs[val_losses.index(min_val_loss)]
    ax1.axhline(y=min_val_loss, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(epochs[-1] * 0.02, min_val_loss * 1.1,
             f'Best: {min_val_loss:.6f} (Epoch {min_val_epoch})',
             fontsize=9, color='red')

    # 2. Learning Rate 그래프
    ax2 = axes[1]
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_title('Learning Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 현재 LR 표시
    current_lr = learning_rates[-1]
    ax2.text(epochs[-1] * 0.98, current_lr * 1.1,
             f'Current: {current_lr:.2e}',
             fontsize=9, color='green', ha='right')

    # 3. 포인트별 Error 그래프
    ax3 = axes[2]
    if active_points:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for idx, point_name in enumerate(active_points):
            color = colors[idx % len(colors)]
            errors = point_errors[point_name]
            ax3.plot(epochs, errors, color=color, label=f'{point_name.capitalize()}',
                    linewidth=2, marker='o', markersize=2, alpha=0.7)

        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Error (pixels)', fontsize=11)
        ax3.set_title('Point-wise Error', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 최소 오차 표시 (첫 번째 포인트)
        if active_points:
            first_point_errors = point_errors[active_points[0]]
            min_error = min(first_point_errors)
            min_error_epoch = epochs[first_point_errors.index(min_error)]
            ax3.text(epochs[-1] * 0.02, min(first_point_errors) * 0.9,
                    f'Best {active_points[0]}: {min_error:.2f}px (Epoch {min_error_epoch})',
                    fontsize=9, color=colors[0])
    else:
        ax3.text(0.5, 0.5, 'No active points', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Error (pixels)', fontsize=11)
        ax3.set_title('Point-wise Error', fontsize=12, fontweight='bold')

    # 4. 평균 Error 그래프
    ax4 = axes[3]
    ax4.plot(epochs, avg_errors, 'purple', linewidth=2, marker='o', markersize=2)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Average Error (pixels)', fontsize=11)
    ax4.set_title('Average Error', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 최소/현재 평균 오차 표시
    min_avg_error = min(avg_errors)
    min_avg_epoch = epochs[avg_errors.index(min_avg_error)]
    current_avg_error = avg_errors[-1]

    ax4.axhline(y=min_avg_error, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.text(epochs[-1] * 0.02, min_avg_error * 0.9,
             f'Best: {min_avg_error:.2f}px (Epoch {min_avg_epoch})',
             fontsize=9, color='red')
    ax4.text(epochs[-1] * 0.98, current_avg_error * 1.1,
             f'Current: {current_avg_error:.2f}px',
             fontsize=9, color='purple', ha='right')

    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 저장
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_training_progress_simple(
    training_log: List[Dict],
    save_path: str,
    dpi: int = 100
) -> None:
    """
    간단한 버전 - Loss만 표시

    Parameters:
    -----------
    training_log : List[Dict]
        학습 로그 데이터
    save_path : str
        그래프 저장 경로
    dpi : int, optional
        그래프 해상도 (기본 100, 더 빠른 생성)
    """
    if not training_log or len(training_log) < 2:
        print("그래프 생성에 충분한 데이터가 없습니다.")
        return

    epochs = [entry['epoch'] for entry in training_log]
    train_losses = [entry['train_loss'] for entry in training_log]
    val_losses = [entry['val_loss'] for entry in training_log]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'orange', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # 테스트용 샘플 데이터
    sample_log = []
    for i in range(1, 101):
        sample_log.append({
            'epoch': i,
            'train_loss': 0.1 * np.exp(-i/30) + 0.001,
            'val_loss': 0.12 * np.exp(-i/30) + 0.0015,
            'learning_rate': 0.001 * (0.95 ** (i // 10)),
            'avg_error': 10 * np.exp(-i/40) + 1,
            'floor_error': 10 * np.exp(-i/40) + 1,
            'center_error': 0,  # 사용하지 않는 포인트
            'front_error': 0,
            'side_error': 0
        })

    # 테스트 그래프 생성
    plot_training_progress(
        training_log=sample_log,
        save_path='test_training_progress.png',
        point_names=['floor', 'center', 'front', 'side']
    )

    print("테스트 완료!")
