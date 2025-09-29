# -*- coding: utf-8 -*-
"""
모델 저장 및 로드 유틸리티 함수들
"""

import torch
import torch.onnx
from pathlib import Path
import warnings
from typing import Optional, Dict, Any, Tuple
import glob
import re


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    model_name: str,
    is_best: bool,
    epoch: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Path:
    """
    PyTorch 모델과 옵티마이저 상태를 저장

    Args:
        model: PyTorch 모델 객체
        optimizer: 옵티마이저 객체
        save_dir: 저장 폴더 경로
        model_name: 모델명
        is_best: best 모델인지 여부
        epoch: 에폭 번호 (일반 모델일 때 필수)
        device: 디바이스 (ONNX 변환용, best일 때만 사용)

    Returns:
        Path: 저장된 체크포인트 경로

    Raises:
        ValueError: is_best=False일 때 epoch가 None인 경우
    """
    # 저장 디렉토리 생성
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 파일명 결정
    if is_best:
        # best 모델은 기존 경로 유지
        checkpoint_path = save_path / f"{model_name}_best.pth"
    else:
        if epoch is None:
            raise ValueError("일반 모델 저장 시 epoch 번호가 필요합니다.")
        # epoch 체크포인트는 하위 폴더에 저장
        epoch_dir = save_path / model_name
        epoch_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = epoch_dir / f"{model_name}_epoch{epoch}.pth"

    # 체크포인트 데이터 구성
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # 에폭 정보 추가 (일반 모델인 경우)
    if not is_best and epoch is not None:
        checkpoint_data['epoch'] = epoch

    # 체크포인트 저장
    torch.save(checkpoint_data, str(checkpoint_path))
    print(f"모델 저장: {checkpoint_path}")

    # Best 모델인 경우 ONNX도 저장
    if is_best and device is not None:
        onnx_path = save_model_as_onnx(model, checkpoint_path, device)
        if onnx_path and onnx_path.exists():
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"ONNX 모델 저장: {onnx_path.name} ({onnx_size_mb:.2f}MB)")

    return checkpoint_path


def save_model_as_onnx(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device
) -> Optional[Path]:
    """
    PyTorch 모델을 ONNX 형식으로 변환 및 저장

    Args:
        model: PyTorch 모델
        checkpoint_path: 원본 .pth 경로
        device: 현재 디바이스

    Returns:
        Path: ONNX 파일 경로 (실패 시 None)
    """
    try:
        # ONNX 파일 경로 생성 (.pth -> .onnx)
        onnx_path = Path(str(checkpoint_path).replace('.pth', '.onnx'))

        # 모델을 CPU로 이동하고 평가 모드로 설정
        original_device = device
        model = model.cpu()
        model.eval()

        # 모델로부터 입력 차원 동적으로 구하기
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # ConfigurableMLPModel의 경우 (PyTorch 모델)
            input_dim = model.layers[0].in_features
            dummy_input = torch.randn(1, input_dim)
            input_names = ['features']
            print(f"ONNX 변환: MLP 모델 감지, 입력 차원={input_dim}")
        elif hasattr(model, 'input_dim'):
            # input_dim 속성이 있는 경우
            input_dim = model.input_dim
            dummy_input = torch.randn(1, input_dim)
            input_names = ['features']
            print(f"ONNX 변환: 특징 기반 모델 감지, 입력 차원={input_dim}")
        elif hasattr(model, 'feature_extractor'):
            # Kornia 모델 등 이미지 입력 모델
            dummy_input = torch.randn(1, 3, 112, 112)
            input_names = ['image']
            print(f"ONNX 변환: 이미지 입력 모델 감지, shape={dummy_input.shape}")
        else:
            # 기본값 (이미지 입력) - 호환성 유지
            dummy_input = torch.randn(1, 3, 112, 112)
            input_names = ['image']
            print(f"ONNX 변환: 기본 이미지 입력 사용, shape={dummy_input.shape}")

        # ONNX로 변환
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=['coordinates'],
                    dynamic_axes={
                        input_names[0]: {0: 'batch_size'},
                        'coordinates': {0: 'batch_size'}
                    },
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                    verbose=False
                )

        # 모델을 원래 디바이스로 복원
        model = model.to(original_device)
        model.train()

        return onnx_path

    except Exception as e:
        print(f"ONNX 변환 실패: {e}")
        # 에러 발생 시에도 모델을 원래 디바이스로 복원
        if 'original_device' in locals():
            model = model.to(original_device)
            model.train()
        return None


def load_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    save_dir: str,
    model_name: str,
    load_best: bool = True,
    device: torch.device = torch.device('cpu')
) -> int:
    """
    저장된 모델 체크포인트 로드

    Args:
        model: PyTorch 모델 객체
        optimizer: 옵티마이저 객체 (Optional)
        save_dir: 저장 폴더 경로
        model_name: 모델명
        load_best: best 모델 로드 여부 (기본: True)
        device: 타겟 디바이스

    Returns:
        int: 로드된 에폭 번호 (best 모델인 경우 -1)

    Raises:
        FileNotFoundError: 체크포인트 파일을 찾을 수 없는 경우
    """
    save_path = Path(save_dir)

    if load_best:
        # Best 모델 로드
        checkpoint_path = save_path / f"{model_name}_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best 모델을 찾을 수 없습니다: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Best 모델 로드: {checkpoint_path}")
        return -1  # Best 모델은 -1 반환

    else:
        # 최신 epoch 모델 로드
        latest_checkpoint = find_latest_checkpoint(save_dir, model_name)
        if latest_checkpoint is None:
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {model_name}_epoch*.pth")

        checkpoint_path, epoch = latest_checkpoint
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Epoch {epoch} 모델 로드: {checkpoint_path}")
        return epoch


def find_latest_checkpoint(
    save_dir: str,
    model_name: str
) -> Optional[Tuple[Path, int]]:
    """
    가장 최신 epoch의 체크포인트 찾기

    Args:
        save_dir: 저장 폴더 경로
        model_name: 모델명

    Returns:
        Optional[Tuple[Path, int]]: (체크포인트 경로, epoch 번호) 또는 None
    """
    save_path = Path(save_dir)

    # epoch 체크포인트는 하위 폴더에서 찾기
    epoch_dir = save_path / model_name
    if not epoch_dir.exists():
        # 하위 폴더가 없으면 기존 경로에서도 찾아보기 (호환성)
        pattern = f"{model_name}_epoch*.pth"
        checkpoint_files = list(save_path.glob(pattern))
    else:
        # 하위 폴더에서 찾기
        pattern = f"{model_name}_epoch*.pth"
        checkpoint_files = list(epoch_dir.glob(pattern))

    if not checkpoint_files:
        return None

    # 파일명에서 epoch 번호 추출
    epoch_pattern = re.compile(rf"{model_name}_epoch(\d+)\.pth")

    max_epoch = -1
    latest_file = None

    for file in checkpoint_files:
        match = epoch_pattern.match(file.name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = file

    if latest_file:
        return latest_file, max_epoch

    return None