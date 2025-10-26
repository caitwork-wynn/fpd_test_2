# -*- coding: utf-8 -*-
"""
모델 저장 및 로드 유틸리티 함수들
"""

import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
import warnings
from typing import Optional, Dict, Any, Tuple, Callable
import glob
import re
import traceback
import yaml


def get_model_path_from_config() -> Optional[Path]:
    """
    config.yml에서 현재 사용 중인 모델 경로를 가져옴
    
    Returns:
        모델 파일의 Path 객체, 설정을 읽을 수 없으면 None
    """
    try:
        config_path = Path(__file__).parent.parent / "config.yml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            model_source = config.get('learning_model', {}).get('source')
            if model_source:
                return Path(__file__).parent.parent / model_source
    except Exception:
        pass
    return None


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    model_name: str,
    is_best: bool,
    epoch: Optional[int] = None,
    device: Optional[torch.device] = None,
    log_func: Optional[Callable[[str], None]] = None,
    is_curr: bool = False
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
        log_func: 로깅 함수 (선택사항)
        is_curr: 현재 진행 중인 모델(curr)인지 여부

    Returns:
        Path: 저장된 체크포인트 경로

    Raises:
        ValueError: is_best=False이고 is_curr=False일 때 epoch가 None인 경우
    """
    # 로깅 함수가 없으면 print 사용
    if log_func is None:
        log_func = print
    # 저장 디렉토리 생성
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 파일명 결정
    if is_best:
        # best 모델은 기존 경로 유지
        checkpoint_path = save_path / f"{model_name}_best.pth"
    elif is_curr:
        # curr 모델 (주기적 저장)
        checkpoint_path = save_path / f"{model_name}_curr.pth"
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

    # 에폭 정보 추가 (일반 모델 또는 curr 모델인 경우)
    if (not is_best or is_curr) and epoch is not None:
        checkpoint_data['epoch'] = epoch

    # 체크포인트 저장
    torch.save(checkpoint_data, str(checkpoint_path))
    log_func(f"모델 저장: {checkpoint_path}")

    # Best 모델인 경우 ONNX도 저장
    if is_best and device is not None:
        onnx_path = save_model_as_onnx(model, checkpoint_path, device, log_func)
        if onnx_path and onnx_path.exists():
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            log_func(f"ONNX 모델 저장: {onnx_path.name} ({onnx_size_mb:.2f}MB)")

    return checkpoint_path


def save_model_as_onnx(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    log_func: Optional[Callable[[str], None]] = None
) -> Optional[Path]:
    """
    PyTorch 모델을 ONNX 형식으로 변환 및 저장

    Args:
        model: PyTorch 모델
        checkpoint_path: 원본 .pth 경로
        device: 현재 디바이스
        log_func: 로깅 함수 (선택사항)

    Returns:
        Path: ONNX 파일 경로 (실패 시 None)
    """
    # 로깅 함수가 없으면 print 사용
    if log_func is None:
        log_func = print
    try:
        # ONNX 파일 경로 생성 (.pth -> .onnx)
        onnx_path = Path(str(checkpoint_path).replace('.pth', '.onnx'))

        # 모델을 CPU로 이동하고 평가 모드로 설정
        original_device = device
        model = model.cpu()
        model.eval()

        # 모델로부터 입력 차원 동적으로 구하기
        if hasattr(model, 'MODEL_NAME') and model.MODEL_NAME == "mpm_cross_self_attention_vpi":
            # MPM 모델: 모델 객체의 input_dim 속성 우선 사용 (가장 정확)
            if hasattr(model, 'input_dim'):
                input_dim = model.input_dim
                log_func(f"ONNX 변환: MPM 모델 감지, 입력 차원={input_dim} (모델 객체 속성)")
            else:
                # fallback: 모델 파일에서 입력 차원 동적으로 가져오기
                try:
                    import importlib.util
                    # 모델 소스 파일 경로 찾기 (여러 패턴 시도)
                    possible_paths = [
                        Path(__file__).parent.parent / "model_defs" / "mpm_cross_self_attention_vpi.py",
                        Path(__file__).parent.parent / "model_defs" / "001.mpm_cross_self_attention_vpi.py",
                        Path(__file__).parent.parent / "model_defs" / "002.mpm_001_ex_canny.py",
                    ]

                    input_dim = None
                    loaded_from = None
                    for model_path in possible_paths:
                        if model_path.exists():
                            try:
                                spec = importlib.util.spec_from_file_location("mpm_model_temp", str(model_path))
                                mpm_module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mpm_module)

                                if hasattr(mpm_module, 'get_input_dim'):
                                    input_dim = mpm_module.get_input_dim()
                                    loaded_from = model_path.name
                                    log_func(f"ONNX 변환: MPM 모델 입력 차원 로드 ({loaded_from}), 입력 차원={input_dim}")
                                    break
                            except Exception as e:
                                # 이 파일에서 로드 실패하면 다음 파일 시도
                                continue

                    if input_dim is None:
                        # 최종 fallback
                        input_dim = 202
                        log_func(f"ONNX 변환: MPM 모델 감지, 입력 차원={input_dim} (기본값)")
                except Exception as e:
                    log_func(f"ONNX 변환 경고: 입력 차원 자동 감지 실패 ({e}), 기본값 202 사용")
                    input_dim = 202

            dummy_input = torch.randn(1, input_dim)
            input_names = ['features']
        elif hasattr(model, 'layers') and len(model.layers) > 0:
            # ConfigurableMLPModel의 경우 (PyTorch 모델)
            input_dim = model.layers[0].in_features
            dummy_input = torch.randn(1, input_dim)
            input_names = ['features']
            log_func(f"ONNX 변환: MLP 모델 감지, 입력 차원={input_dim}")
        elif hasattr(model, 'input_dim'):
            # input_dim 속성이 있는 경우
            input_dim = model.input_dim
            dummy_input = torch.randn(1, input_dim)
            input_names = ['features']
            log_func(f"ONNX 변환: 특징 기반 모델 감지, 입력 차원={input_dim}")
        elif hasattr(model, 'MODEL_NAME') and 'mobilenet' in model.MODEL_NAME.lower():
            # MobileNet 모델: get_input_dim() 함수로부터 입력 차원 가져오기
            try:
                import importlib.util
                # 모델 소스 파일 경로 찾기 - config.yml에서 가져오기
                model_path = get_model_path_from_config()
                
                input_shape = None
                if model_path and model_path.exists():
                    try:
                        spec = importlib.util.spec_from_file_location("mobilenet_model_temp", str(model_path))
                        mobilenet_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mobilenet_module)

                        if hasattr(mobilenet_module, 'get_input_dim'):
                            input_shape = mobilenet_module.get_input_dim()
                            log_func(f"ONNX 변환: MobileNet 모델 감지 ({model_path.name}), 입력 shape={input_shape}")
                    except Exception:
                        pass

                if input_shape is None:
                    # fallback
                    input_shape = (1, 96, 96)
                    log_func(f"ONNX 변환: MobileNet 모델 감지 (기본값), 입력 shape={input_shape}")

                dummy_input = torch.randn(1, *input_shape)
                input_names = ['image']
            except Exception as e:
                log_func(f"ONNX 변환 경고: MobileNet 입력 차원 자동 감지 실패 ({e}), 기본값 사용")
                dummy_input = torch.randn(1, 1, 96, 96)
                input_names = ['image']
        elif hasattr(model, 'feature_extractor'):
            # Kornia 모델 등 이미지 입력 모델
            dummy_input = torch.randn(1, 3, 112, 112)
            input_names = ['image']
            log_func(f"ONNX 변환: 이미지 입력 모델 감지, shape={dummy_input.shape}")
        else:
            # 기본값 (이미지 입력) - 호환성 유지
            dummy_input = torch.randn(1, 3, 112, 112)
            input_names = ['image']
            log_func(f"ONNX 변환: 기본 이미지 입력 사용, shape={dummy_input.shape}")

        # 모델 출력 형태 확인을 위한 테스트 실행
        with torch.no_grad():
            test_output = model(dummy_input)

            # 출력이 딕셔너리인 경우 처리
            if isinstance(test_output, dict):
                log_func("ONNX 변환: 딕셔너리 출력 모델 감지")
                # 'coordinates' 키가 있으면 그것만 반환하는 래퍼 생성
                if 'coordinates' in test_output:
                    # MPM 모델인 경우 전용 래퍼 사용
                    if hasattr(model, 'MODEL_NAME') and model.MODEL_NAME == "mpm_cross_self_attention_vpi":
                        try:
                            # MPMAttentionModelONNX import (여러 파일 시도)
                            import importlib.util

                            possible_wrapper_paths = [
                                Path(__file__).parent.parent / "model_defs" / "001.mpm_cross_self_attention_vpi.py",
                                Path(__file__).parent.parent / "model_defs" / "002.mpm_001_ex_canny.py",
                                Path(__file__).parent.parent / "model_defs" / "mpm_cross_self_attention_vpi.py",
                            ]

                            wrapper_loaded = False
                            for wrapper_path in possible_wrapper_paths:
                                if wrapper_path.exists():
                                    try:
                                        spec = importlib.util.spec_from_file_location(
                                            "mpm_model_wrapper",
                                            str(wrapper_path)
                                        )
                                        mpm_module = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(mpm_module)

                                        if hasattr(mpm_module, 'MPMAttentionModelONNX'):
                                            MPMAttentionModelONNX = mpm_module.MPMAttentionModelONNX
                                            export_model = MPMAttentionModelONNX(model)
                                            log_func(f"ONNX 변환: MPMAttentionModelONNX 래퍼 사용 ({wrapper_path.name})")
                                            wrapper_loaded = True
                                            break
                                    except Exception:
                                        # 이 파일에서 로드 실패하면 다음 파일 시도
                                        continue

                            if not wrapper_loaded:
                                # 모든 파일에서 래퍼를 찾지 못한 경우 기본 래퍼 사용
                                raise ImportError("MPMAttentionModelONNX를 찾을 수 없음")

                        except Exception as e:
                            # 기본 래퍼 사용 (경고 없이 조용히 처리)
                            class ONNXWrapper(nn.Module):
                                def __init__(self, model):
                                    super().__init__()
                                    self.model = model

                                def forward(self, x):
                                    output_dict = self.model(x)
                                    return output_dict['coordinates']

                            export_model = ONNXWrapper(model)
                            log_func("ONNX 변환: 기본 ONNX 래퍼 사용")
                    elif hasattr(model, 'MODEL_NAME') and 'mobilenet' in model.MODEL_NAME.lower():
                        # MobileNet 모델 전용 래퍼 시도
                        try:
                            import importlib.util
                            # config.yml에서 현재 모델 경로 가져오기
                            wrapper_path = get_model_path_from_config()
                            
                            wrapper_loaded = False
                            if wrapper_path and wrapper_path.exists():
                                try:
                                    spec = importlib.util.spec_from_file_location(
                                        "mobilenet_model_wrapper",
                                        str(wrapper_path)
                                    )
                                    mobilenet_module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(mobilenet_module)

                                    if hasattr(mobilenet_module, 'MPMMobileNetLightweightModelONNX'):
                                        MPMMobileNetLightweightModelONNX = mobilenet_module.MPMMobileNetLightweightModelONNX
                                        export_model = MPMMobileNetLightweightModelONNX(model)
                                        log_func(f"ONNX 변환: MPMMobileNetLightweightModelONNX 래퍼 사용 ({wrapper_path.name})")
                                        wrapper_loaded = True
                                except Exception:
                                    pass

                            if not wrapper_loaded:
                                raise ImportError("MPMMobileNetLightweightModelONNX를 찾을 수 없음")

                        except Exception:
                            # 기본 래퍼 사용
                            class ONNXWrapper(nn.Module):
                                def __init__(self, model):
                                    super().__init__()
                                    self.model = model

                                def forward(self, x):
                                    output_dict = self.model(x)
                                    return output_dict['coordinates']

                            export_model = ONNXWrapper(model)
                            log_func("ONNX 변환: MobileNet 기본 ONNX 래퍼 사용")
                    else:
                        class ONNXWrapper(nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model

                            def forward(self, x):
                                output_dict = self.model(x)
                                return output_dict['coordinates']

                        export_model = ONNXWrapper(model)
                        log_func("ONNX 변환: coordinates 출력만 추출하는 래퍼 사용")
                else:
                    log_func("ONNX 변환 경고: 'coordinates' 키를 찾을 수 없음")
                    export_model = model
            else:
                export_model = model

        # ONNX로 변환
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    export_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=18,  # PyTorch 권장 버전, ReduceMean 호환성
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

        # external data 형식으로 저장된 경우, 단일 파일로 변환
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))

            # 모든 데이터를 하나의 파일에 저장
            onnx.save(onnx_model, str(onnx_path))
            log_func(f"ONNX 변환: 단일 파일로 통합 완료")
        except Exception as e:
            log_func(f"ONNX 단일 파일 변환 경고: {e}")

        # 모델을 원래 디바이스로 복원
        model = model.to(original_device)
        model.train()

        return onnx_path

    except Exception as e:
        log_func(f"ONNX 변환 실패: {e}")
        log_func(f"에러 타입: {type(e).__name__}")
        log_func("상세 스택 트레이스:")
        for line in traceback.format_exc().splitlines():
            log_func(line)
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
    저장된 모델 체크포인트 로드 (우선순위: curr → best → epoch)

    Args:
        model: PyTorch 모델 객체
        optimizer: 옵티마이저 객체 (Optional)
        save_dir: 저장 폴더 경로
        model_name: 모델명
        load_best: best 모델 로드 여부 (기본: True, 하지만 curr가 있으면 curr 우선)
        device: 타겟 디바이스

    Returns:
        int: 로드된 에폭 번호 (curr/best 모델인 경우 체크포인트의 epoch 값 또는 -1)

    Raises:
        FileNotFoundError: 모든 체크포인트 파일을 찾을 수 없는 경우
    """
    save_path = Path(save_dir)

    # 1순위: curr 모델 시도
    curr_path = save_path / f"{model_name}_curr.pth"
    if curr_path.exists():
        checkpoint = torch.load(str(curr_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', -1)
        print(f"Current 모델 로드 (epoch {epoch}): {curr_path}")
        return epoch

    # 2순위: best 모델 시도
    if load_best:
        best_path = save_path / f"{model_name}_best.pth"
        if best_path.exists():
            checkpoint = torch.load(str(best_path), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            epoch = checkpoint.get('epoch', -1)
            print(f"Best 모델 로드 (epoch {epoch}): {best_path}")
            return epoch

    # 3순위: 최신 epoch 모델 로드
    latest_checkpoint = find_latest_checkpoint(save_dir, model_name)
    if latest_checkpoint is None:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {model_name}_*.pth")

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