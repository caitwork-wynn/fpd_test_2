#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
710.make_16x16_autoencoder.py
16x16 이미지 Autoencoder 학습 스크립트

- data/base/not-labeled의 이미지를 사용
- 112x112로 랜덤 크롭 + 좌우 반전 증강
- 7x7 그리드로 분할하여 16x16 패치 생성 (리사이즈 없음)
- 메모리 기반 학습으로 속도 최적화
"""

import os
import sys
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import csv
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Optional

# 모델 import
sys.path.append(str(Path(__file__).parent))
from model_defs.autoencoder_16x16 import SeparatedResidualAutoencoder16x16


class PatchDatasetMemory(Dataset):
    """
    16x16 패치 데이터셋 (전체 데이터를 메모리에 로드)
    """

    def __init__(self, config: Dict, mode: str = 'train'):
        """
        Args:
            config: 설정 딕셔너리
            mode: 'train' 또는 'test'
        """
        self.config = config
        self.mode = mode

        # 설정 로드
        self.source_folder = Path(config['data']['source_folder'])
        if not self.source_folder.is_absolute():
            self.source_folder = (Path(__file__).parent / self.source_folder).resolve()

        self.image_size = config['data']['image_size']
        self.patch_size = config['data']['patch_size']
        self.grid_size = config['data']['grid_size']
        self.use_grayscale = config['data']['use_grayscale']
        self.max_images = config['data']['max_images']

        # 증강 설정
        self.random_crop = config['augmentation']['random_crop'] and mode == 'train'
        self.horizontal_flip = config['augmentation']['horizontal_flip'] and mode == 'train'
        self.crop_min_ratio = config['augmentation']['crop_min_ratio']
        self.crop_max_ratio = config['augmentation']['crop_max_ratio']

        # 데이터 로드
        self.patches = self.load_all_patches()

        # Train/Test 분할 (90/10)
        n_patches = len(self.patches)
        n_train = int(n_patches * 0.9)

        if mode == 'train':
            self.patches = self.patches[:n_train]
        else:
            self.patches = self.patches[n_train:]

        print(f"{mode.upper()} Dataset: {len(self.patches)} patches loaded")
        print(f"Memory usage: {self.patches.nbytes / (1024**2):.2f} MB")

    def load_all_patches(self) -> np.ndarray:
        """모든 이미지를 로드하고 패치로 변환"""
        print("\nLoading images and creating patches...")

        # 이미지 파일 목록
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(self.source_folder.glob(ext)))

        if self.max_images:
            image_files = image_files[:self.max_images]

        print(f"Number of images to process: {len(image_files)}")

        # 모든 패치를 저장할 리스트
        all_patches = []

        # 각 이미지 처리
        for img_path in tqdm(image_files, desc="Processing images"):
            # 이미지 로드
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # 그레이스케일 변환
            if self.use_grayscale and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif not self.use_grayscale and len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # 원본 이미지로 패치 생성
            patches_orig = self.extract_patches_from_image(image, augment=False)
            all_patches.extend(patches_orig)

            # 좌우 반전 증강
            if self.horizontal_flip:
                image_flipped = cv2.flip(image, 1)
                patches_flip = self.extract_patches_from_image(image_flipped, augment=False)
                all_patches.extend(patches_flip)

        # numpy 배열로 변환
        all_patches = np.array(all_patches, dtype=np.float32)

        # 채널 차원 추가 (그레이스케일의 경우)
        if self.use_grayscale and len(all_patches.shape) == 3:
            all_patches = np.expand_dims(all_patches, axis=1)  # (N, 1, 16, 16)
        elif not self.use_grayscale and len(all_patches.shape) == 4:
            all_patches = np.transpose(all_patches, (0, 3, 1, 2))  # (N, 3, 16, 16)

        # 정규화 (0-1)
        all_patches = all_patches / 255.0

        return all_patches

    def extract_patches_from_image(self, image: np.ndarray, augment: bool = True) -> List[np.ndarray]:
        """이미지에서 16x16 패치 추출"""
        h, w = image.shape[:2]

        # 랜덤 크롭 또는 리사이즈
        if augment and self.random_crop and min(h, w) > self.image_size:
            # 랜덤 크롭
            scale = random.uniform(self.crop_min_ratio, self.crop_max_ratio)
            crop_size = int(self.image_size / scale)
            crop_size = min(crop_size, min(h, w))

            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)

            cropped = image[y:y+crop_size, x:x+crop_size]
            resized = cv2.resize(cropped, (self.image_size, self.image_size))
        else:
            # 단순 리사이즈
            resized = cv2.resize(image, (self.image_size, self.image_size))

        # 패치 추출
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y_start = i * self.patch_size
                x_start = j * self.patch_size

                patch = resized[y_start:y_start+self.patch_size,
                               x_start:x_start+self.patch_size]

                # 16x16 패치를 그대로 사용 (리사이즈 없음)
                patches.append(patch)

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.patches[idx])


def train_epoch(model, dataloader, optimizer, criterion, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(device)

        # Forward
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def visualize_reconstruction(model, dataset, device, save_path, epoch, config):
    """복원 결과 시각화"""
    model.eval()

    # 랜덤 샘플 선택
    n_samples = config['training']['viz_samples']
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    # 샘플 추출 및 복원
    originals = []
    reconstructed = []

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx].unsqueeze(0).to(device)
            recon = model(sample)

            originals.append(sample.cpu().numpy()[0])
            reconstructed.append(recon.cpu().numpy()[0])

    # 그리드 생성 (4행 x 10열 = 40개)
    n_rows = 4
    n_cols = 10
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(n_rows * 2, n_cols, hspace=0.02, wspace=0.02)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(originals):
                break

            # 원본 이미지 (홀수 행)
            ax_orig = fig.add_subplot(gs[i*2, j])
            if config['data']['use_grayscale']:
                ax_orig.imshow(originals[idx][0], cmap='gray', vmin=0, vmax=1)
            else:
                img = np.transpose(originals[idx], (1, 2, 0))
                ax_orig.imshow(np.clip(img, 0, 1))
            ax_orig.axis('off')

            # 복원 이미지 (짝수 행)
            ax_recon = fig.add_subplot(gs[i*2+1, j])
            if config['data']['use_grayscale']:
                ax_recon.imshow(reconstructed[idx][0], cmap='gray', vmin=0, vmax=1)
            else:
                img = np.transpose(reconstructed[idx], (1, 2, 0))
                ax_recon.imshow(np.clip(img, 0, 1))
            ax_recon.axis('off')

    # 제목 추가
    fig.suptitle(f'Epoch {epoch} - Top: Original, Bottom: Reconstructed (16x16)', fontsize=16)

    # 저장
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved: {save_path}")


def save_training_log(log_data: List[Dict], filepath: str):
    """학습 로그 CSV 저장"""
    if not log_data:
        return

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
        writer.writeheader()
        writer.writerows(log_data)


def main():
    """메인 함수"""
    print("=" * 60)
    print("16x16 Autoencoder Training")
    print("=" * 60)

    # 설정 로드
    config_path = Path(__file__).parent / 'ae_16x16_config.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 경로 설정
    src_dir = Path(__file__).parent
    base_dir = src_dir.parent

    # 상대 경로를 src 폴더 기준으로 해석
    model_dir = (src_dir / config['paths']['model_dir']).resolve()
    result_base_dir = (src_dir / config['paths']['result_dir']).resolve()
    log_dir = (src_dir / config['paths']['log_dir']).resolve()

    model_dir.mkdir(exist_ok=True, parents=True)
    result_base_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    # 모델 이름
    model_name = config['model']['name']

    # 모델별 result 서브폴더 생성
    result_dir = result_base_dir / model_name
    result_dir.mkdir(exist_ok=True)

    # 디바이스 설정
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 데이터셋 생성
    print("\nCreating dataset...")
    train_dataset = PatchDatasetMemory(config, mode='train')
    test_dataset = PatchDatasetMemory(config, mode='test')

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # 모델 생성
    print("\nCreating model...")
    model = SeparatedResidualAutoencoder16x16(
        input_channels=config['model']['input_channels'],
        output_channels=config['model']['output_channels'],
        latent_dim=config['model']['latent_dim']
    ).to(device)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 옵티마이저 및 손실 함수
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    criterion = nn.MSELoss()

    # 스케줄러 설정
    scheduler_config = config['training']['scheduler']
    if scheduler_config['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            min_lr=scheduler_config['min_lr']
        )
    else:  # cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=scheduler_config['min_lr']
        )

    # 기존 모델 체크
    best_checkpoint_path = model_dir / f"{model_name}_best.pth"
    checkpoint_path = model_dir / f"{model_name}.pth"

    start_epoch = 0
    best_loss = float('inf')

    if best_checkpoint_path.exists():
        print(f"\nLoading existing model: {best_checkpoint_path}")
        checkpoint = torch.load(str(best_checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint['loss']
        print(f"Previous best loss: {best_loss:.6f}")

    # 학습 시작
    print(f"\n===== Training Start =====")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total epochs: {config['training']['epochs']}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Batch size: {config['training']['batch_size']}")
    print("=" * 50)

    # Early stopping 설정
    patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']
    patience_counter = 0

    # 학습 로그
    training_log = []

    # 학습 루프
    for epoch in range(start_epoch + 1, config['training']['epochs'] + 1):
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 검증
        test_loss = validate(model, test_loader, criterion, device)

        # 스케줄러 업데이트
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        # 로깅
        current_lr = optimizer.param_groups[0]['lr']
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'learning_rate': current_lr
        }
        training_log.append(log_entry)

        # 출력
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch:4d}/{config['training']['epochs']}] | "
                  f"LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f}")

        # 체크포인트 저장
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'config': config
            }
            torch.save(checkpoint_data, str(checkpoint_path))

            # Encoder/Decoder 분리 저장
            encoder_state = {
                'encoder_state_dict': model.encoder.state_dict(),
                'latent_dim': config['model']['latent_dim'],
                'input_channels': config['model']['input_channels']
            }
            encoder_path = model_dir / f"{model_name}_encoder.pth"
            torch.save(encoder_state, str(encoder_path))

        # Best 모델 저장
        if test_loss < best_loss - min_delta:
            best_loss = test_loss
            patience_counter = 0

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'config': config
            }
            torch.save(checkpoint_data, str(best_checkpoint_path))

            # 현재 모델을 ae_16x16_current.pth로도 저장
            current_model_path = model_dir / f"{model_name}_current.pth"
            torch.save(checkpoint_data, str(current_model_path))

            print(f"[BEST] Model saved! Loss: {test_loss:.6f}")
            print(f"  - {best_checkpoint_path}")
            print(f"  - {current_model_path}")
        else:
            patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # 시각화
        if epoch % config['training']['viz_interval'] == 0:
            viz_path = result_dir / f"{model_name}_epoch_{epoch:04d}.png"
            visualize_reconstruction(model, test_dataset, device, viz_path, epoch, config)

            # 시각화 시점에 현재 모델 저장
            viz_model_path = model_dir / f"{model_name}_epoch_{epoch:04d}.pth"
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'config': config
            }
            torch.save(checkpoint_data, str(viz_model_path))
            print(f"Model saved at visualization: {viz_model_path}")

    # 학습 로그 저장
    if config['logging']['save_csv']:
        log_csv_path = result_dir / f"{model_name}_training_log.csv"
        save_training_log(training_log, str(log_csv_path))
        print(f"\nTraining log saved: {log_csv_path}")

    # 최종 시각화
    final_viz_path = result_dir / f"{model_name}.png"
    visualize_reconstruction(model, test_dataset, device, final_viz_path, epoch, config)

    # 최종 모델 저장
    final_model_path = model_dir / f"{model_name}_final.pth"
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
        'config': config
    }
    torch.save(final_checkpoint, str(final_model_path))
    print(f"Final model saved: {final_model_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final Test Loss: {test_loss:.6f}")
    print(f"Best Test Loss: {best_loss:.6f}")
    print(f"\nSaved files:")
    print(f"  - Model: {checkpoint_path}")
    print(f"  - Best model: {best_checkpoint_path}")
    print(f"  - Encoder: {model_dir / f'{model_name}_encoder.pth'}")
    print(f"  - Visualization: {final_viz_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()