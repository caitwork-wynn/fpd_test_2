# -*- coding: utf-8 -*-
"""
RvC 및 고급 모델 학습 스크립트
- Hybrid Model, Dual Positional Model, RvC Model 지원
- rvc_config.yml에서 설정 읽기
"""

import os
import sys
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import csv
from typing import Dict, List, Tuple, Optional
import random
import time
import json
from tqdm import tqdm
import shutil
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import models
from model_defs.fpd_mix_ae_position_embedding import (
    create_model, RvCLoss, LatentEmbeddingModel, RvCModel
)
from model_defs.multi_point_model_pytorch import MultiPointDetectorPyTorch


class DualLogger:
    """화면과 파일에 동시 출력하는 로거 클래스"""

    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message: str):
        """메시지를 화면과 파일에 동시 출력"""
        print(message, end='')
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def close(self):
        """파일 핸들 닫기"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class AdvancedMultiPointDataset(Dataset):
    """고급 모델용 다중 포인트 검출 데이터셋"""

    def __init__(self,
                 source_folder: str,
                 labels_file: str,
                 detector: MultiPointDetectorPyTorch,
                 mode: str = 'train',
                 config: Dict = None,
                 model_type: str = 'hybrid',
                 augment: bool = True):
        """
        Args:
            source_folder: 이미지 폴더 경로
            labels_file: 레이블 파일명
            detector: 특징 추출용 검출기
            mode: 'train', 'val', 'test'
            config: 설정 딕셔너리
            model_type: 'hybrid', 'dual_positional', 'rvc'
            augment: 데이터 증강 여부
        """
        self.source_folder = source_folder
        self.detector = detector
        self.mode = mode
        self.config = config or {}
        self.model_type = model_type
        self.augment = augment and (mode == 'train')

        # 설정 읽기
        self.image_size = config['data']['input_size']
        self.grid_size = config['data']['grid_size']
        self.patch_size = config['data']['patch_size']

        # 데이터 증강 설정
        aug_config = config['data']['augmentation']
        self.use_random_crop = aug_config.get('random_crop', False) and self.augment
        self.crop_scale = aug_config.get('crop_scale', [0.8, 1.0])
        self.use_horizontal_flip = aug_config.get('horizontal_flip', False) and self.augment
        self.flip_prob = aug_config.get('flip_prob', 0.5)

        # 레이블 로드
        self.labels = self._load_labels(labels_file)
        self.image_paths = []
        self.targets = []

        # 데이터 분리 (train/val/test)
        self._split_data()

        print(f"Dataset initialized: {self.mode} mode, {len(self.image_paths)} samples")

    def _load_labels(self, labels_file: str) -> Dict:
        """레이블 파일 로드"""
        labels_path = os.path.join(self.source_folder, labels_file)

        if not os.path.exists(labels_path):
            # 레이블 파일이 없으면 빈 딕셔너리 반환
            print(f"Warning: Labels file not found: {labels_path}")
            return {}

        labels = {}
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 9:  # ID + 8 coordinates
                        image_id = parts[0]
                        coords = list(map(float, parts[1:9]))
                        labels[image_id] = coords

        return labels

    def _split_data(self):
        """데이터를 train/val/test로 분리"""
        all_files = []

        # 이미지 파일 찾기
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            all_files.extend(Path(self.source_folder).glob(ext))

        # 파일명과 레이블 매칭
        valid_files = []
        for file_path in all_files:
            file_id = file_path.stem
            if file_id in self.labels:
                valid_files.append((file_path, self.labels[file_id]))

        # 데이터가 없는 경우 처리
        if not valid_files:
            print(f"Warning: No valid data found for {self.mode} mode")
            self.image_paths = []
            self.targets = []
            return

        # 파일 정렬 (재현성을 위해)
        valid_files.sort(key=lambda x: str(x[0]))

        # Train/Val/Test 분리
        total = len(valid_files)
        val_split = self.config['training']['val_split']
        test_split = self.config['training']['test_split']

        val_size = int(total * val_split)
        test_size = int(total * test_split)
        train_size = total - val_size - test_size

        if self.mode == 'train':
            selected = valid_files[:train_size]
        elif self.mode == 'val':
            selected = valid_files[train_size:train_size+val_size]
        else:  # test
            selected = valid_files[train_size+val_size:]

        # 저장
        for file_path, coords in selected:
            self.image_paths.append(file_path)
            self.targets.append(coords)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """데이터 샘플 반환"""
        # 이미지 로드
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 타겟 좌표
        target = np.array(self.targets[idx], dtype=np.float32)

        # 리사이즈
        original_size = image.shape[:2]
        image_resized = cv2.resize(image, (self.image_size, self.image_size))

        # 좌표 스케일링
        scale_x = self.image_size / original_size[1]
        scale_y = self.image_size / original_size[0]
        target[0::2] *= scale_x
        target[1::2] *= scale_y

        # 데이터 증강
        if self.augment:
            image_resized, target = self._augment(image_resized, target)

        # 특징 추출
        features = self._extract_features(image_resized)

        # 이미지를 텐서로 변환 (Hybrid 모델용)
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:
            # RGB to grayscale for encoder
            image_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            image_tensor = image_tensor.unsqueeze(0)

        return {
            'features': torch.from_numpy(features).float(),
            'image': image_tensor,
            'target': torch.from_numpy(target).float(),
            'path': str(img_path)
        }

    def _extract_features(self, image):
        """특징 추출 (hand-crafted features)"""
        # 기존 detector 사용하여 특징 추출
        # 여기서는 간단히 구현 (실제로는 200.learning_pytorch.py의 방식 사용)
        features = self.detector.extract_features(image)
        return features

    def _augment(self, image, target):
        """데이터 증강"""
        h, w = image.shape[:2]

        # Horizontal flip
        if self.use_horizontal_flip and random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            # x 좌표 뒤집기
            target[0::2] = w - target[0::2]

        # Random crop
        if self.use_random_crop:
            scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
            new_h = int(h * scale)
            new_w = int(w * scale)

            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

            image = image[top:top+new_h, left:left+new_w]
            image = cv2.resize(image, (w, h))

            # 좌표 조정
            target[0::2] = (target[0::2] - left) * (w / new_w)
            target[1::2] = (target[1::2] - top) * (h / new_h)

        return image, target


class ModelTrainer:
    """모델 학습 관리 클래스"""

    def __init__(self, config: Dict, model_type: str = 'hybrid'):
        self.config = config
        self.model_type = model_type
        self.device = self._setup_device()

        # 로거 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path('../logs') / model_type / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = DualLogger(str(log_dir / 'training.log'))

        # 모델 생성
        self.model = self._create_model()
        self.model.to(self.device)

        # 손실 함수
        self.criterion = self._create_criterion()

        # 옵티마이저
        self.optimizer = self._create_optimizer()

        # 스케줄러
        self.scheduler = self._create_scheduler()

        # 결과 저장 경로
        self.checkpoint_dir = Path('../model')
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_loss = float('inf')
        self.patience_counter = 0

    def _setup_device(self):
        """디바이스 설정"""
        if self.config['device']['cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['device_id']}")
            self.logger.write(f"Using GPU: {torch.cuda.get_device_name(device)}\n")
        else:
            device = torch.device("cpu")
            self.logger.write("Using CPU\n")
        return device

    def _create_model(self):
        """모델 생성"""
        if self.model_type == 'hybrid':
            model_config = self.config['hybrid_model']
        elif self.model_type == 'dual_positional':
            model_config = self.config['dual_positional_model']
        elif self.model_type == 'rvc':
            model_config = self.config['rvc_model']
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model = create_model(self.model_type, model_config)

        # 파라미터 수 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.write(f"Total parameters: {total_params:,}\n")
        self.logger.write(f"Trainable parameters: {trainable_params:,}\n")

        return model

    def _create_criterion(self):
        """손실 함수 생성"""
        if self.model_type == 'rvc':
            criterion = RvCLoss(self.config['rvc_model'])
        else:
            # Hybrid와 Dual Positional은 MSE 사용
            criterion = nn.MSELoss()
        return criterion

    def _create_optimizer(self):
        """옵티마이저 생성"""
        train_config = self.config['training']

        if train_config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=train_config['learning_rate']
            )
        return optimizer

    def _create_scheduler(self):
        """학습률 스케줄러 생성"""
        scheduler_config = self.config['training']['scheduler']

        if scheduler_config['type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self, dataloader, epoch):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            features = batch['features'].to(self.device)
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.model_type == 'rvc':
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
            elif isinstance(self.model, LatentEmbeddingModel):
                # Hybrid model needs both features and images
                outputs = self.model(features, images)
                loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(features, images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / batch_count

    def validate(self, dataloader, epoch):
        """검증"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        all_errors = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                features = batch['features'].to(self.device)
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)

                # Forward pass
                if self.model_type == 'rvc':
                    outputs = self.model(features)
                    # Decode predictions for RvC
                    predictions = self.model.decode_predictions(
                        outputs,
                        method=self.config['rvc_model']['inference_method'],
                        temperature=self.config['rvc_model']['temperature']
                    )
                    loss = self.criterion(outputs, targets)
                elif isinstance(self.model, LatentEmbeddingModel):
                    predictions = self.model(features, images)
                    loss = self.criterion(predictions, targets)
                else:
                    predictions = self.model(features, images)
                    loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                batch_count += 1

                # Calculate errors
                if self.model_type != 'rvc' or hasattr(predictions, 'cpu'):
                    errors = (predictions - targets).cpu().numpy()
                    all_errors.extend(errors)

        avg_loss = total_loss / batch_count

        # Calculate metrics
        if all_errors:
            all_errors = np.array(all_errors)
            mae = np.mean(np.abs(all_errors))
            rmse = np.sqrt(np.mean(all_errors ** 2))

            self.logger.write(f"Validation - Loss: {avg_loss:.6f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}\n")
        else:
            self.logger.write(f"Validation - Loss: {avg_loss:.6f}\n")

        return avg_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'model_type': self.model_type
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_type}_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_type}_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.write(f"Best model saved: {best_path}\n")

    def train(self, train_loader, val_loader, num_epochs):
        """전체 학습 프로세스"""
        self.logger.write(f"\n{'='*60}\n")
        self.logger.write(f"Starting training for {self.model_type} model\n")
        self.logger.write(f"{'='*60}\n\n")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss = self.validate(val_loader, epoch)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Check for improvement
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1

            # Save checkpoint periodically
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, val_loss)

            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.write(f"Early stopping triggered at epoch {epoch}\n")
                break

            # Log progress
            self.logger.write(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                            f"Best Loss={self.best_loss:.6f}, Patience={self.patience_counter}\n")

        self.logger.write(f"\nTraining completed. Best validation loss: {self.best_loss:.6f}\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Train RvC and advanced models')
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['hybrid', 'dual_positional', 'rvc'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default='rvc_config.yml',
                       help='Config file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # 설정 파일 로드
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from {config_path}")

    # 기본 detector 생성 (특징 추출용)
    detector_config_path = Path(__file__).parent / 'config.yml'
    if detector_config_path.exists():
        with open(detector_config_path, 'r', encoding='utf-8') as f:
            detector_config = yaml.safe_load(f)
        detector = MultiPointDetectorPyTorch(detector_config)
    else:
        # 기본 설정 사용
        detector = MultiPointDetectorPyTorch({})

    # 데이터셋 생성
    data_config = config['data']
    data_path = Path(__file__).parent / data_config['data_path']

    train_dataset = AdvancedMultiPointDataset(
        source_folder=str(data_path),
        labels_file='labels.csv',
        detector=detector,
        mode='train',
        config=config,
        model_type=args.model,
        augment=True
    )

    val_dataset = AdvancedMultiPointDataset(
        source_folder=str(data_path),
        labels_file='labels.csv',
        detector=detector,
        mode='val',
        config=config,
        model_type=args.model,
        augment=False
    )

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory']
    )

    # 학습 시작
    trainer = ModelTrainer(config, args.model)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")
    else:
        start_epoch = 1

    # Train
    trainer.train(train_loader, val_loader, config['training']['epochs'])

    # Clean up
    trainer.logger.close()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()