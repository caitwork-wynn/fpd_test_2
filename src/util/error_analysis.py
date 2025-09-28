import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional, Any


def display_error_analysis(
    errors: Dict[str, Dict[str, List[float]]],
    epoch: Union[int, str] = None,
    point_names: Optional[Dict[str, str]] = None,
    title: Optional[str] = None
) -> List[str]:
    """
    테스트 결과의 상세 오차 분석을 출력 형식으로 변환

    Parameters:
    -----------
    errors : Dict[str, Dict[str, List[float]]]
        오차 데이터. 각 포인트별로 'x', 'y', 'dist' 리스트를 포함
        예: {'center': {'x': [1.2, 2.3, ...], 'y': [...], 'dist': [...]}}

    epoch : Union[int, str], optional
        Epoch 번호 또는 'final' 같은 문자열

    point_names : Dict[str, str], optional
        포인트 키와 표시 이름 매핑
        예: {'point1': 'floor,center', 'point2': 'x,y'}

    title : str, optional
        커스텀 제목. 없으면 기본 제목 사용

    Returns:
    --------
    List[str]
        포맷팅된 출력 라인들의 리스트
    """
    output_lines = []

    # 제목 설정
    if title:
        output_lines.append(title)
    elif epoch is not None:
        output_lines.append(f"=== Epoch {epoch} 오차 분석 ===")
    else:
        output_lines.append("=== 오차 분석 ===")

    # 포인트 이름 매핑 준비
    if point_names is None:
        # 기본 이름 사용 (키를 대문자로)
        point_names = {key: key.upper() for key in errors.keys()}

    # 각 포인트별 통계 출력
    all_distances = []

    for key in errors.keys():
        if key in errors and errors[key].get('x') and errors[key].get('y'):
            # 통계 계산
            stats = calculate_statistics(errors[key])

            # 표시 이름 결정
            display_name = point_names.get(key, key.upper())

            # 포맷팅된 라인 생성
            line = format_single_coordinate(display_name, stats)
            output_lines.append(line)

            # 전체 거리 오차 수집
            if 'dist' in errors[key]:
                all_distances.extend(errors[key]['dist'])

    # 전체 평균 거리 오차 계산 및 출력
    if all_distances:
        overall_mean = np.mean(all_distances)
        overall_std = np.std(all_distances)
        output_lines.append(f"전체: 평균={overall_mean:.2f}±{overall_std:.2f} pixels")

    # 비고 추가
    output_lines.append("비고) 값:좌표 오차 평균±표준편차, Dist:유클리드 거리 오차")

    return output_lines


def calculate_statistics(error_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    오차 데이터의 통계 계산

    Parameters:
    -----------
    error_data : Dict[str, List[float]]
        'x', 'y', 'dist' 키를 가진 오차 데이터

    Returns:
    --------
    Dict[str, Dict[str, float]]
        각 축별 평균과 표준편차
    """
    stats = {}

    for axis in ['x', 'y', 'dist']:
        if axis in error_data and error_data[axis]:
            stats[axis] = {
                'mean': np.mean(error_data[axis]),
                'std': np.std(error_data[axis])
            }
        else:
            stats[axis] = {'mean': 0.0, 'std': 0.0}

    return stats


def format_single_coordinate(
    name: str,
    stats: Dict[str, Dict[str, float]]
) -> str:
    """
    단일 좌표의 통계를 포맷팅

    Parameters:
    -----------
    name : str
        좌표 이름 (예: 'CENTER', 'floor,center')
    stats : Dict[str, Dict[str, float]]
        통계 데이터

    Returns:
    --------
    str
        포맷팅된 문자열
    """
    # 이름 포맷팅 (최소 6자 너비)
    formatted_name = f"{name:6s}"

    # 각 축의 통계 포맷팅
    x_str = f"X={stats['x']['mean']:4.1f}±{stats['x']['std']:4.1f}"
    y_str = f"Y={stats['y']['mean']:4.1f}±{stats['y']['std']:4.1f}"
    dist_str = f"Dist={stats['dist']['mean']:4.1f}±{stats['dist']['std']:4.1f}"

    # 탭으로 정렬된 형식
    return f"{formatted_name}: {x_str},\t{y_str},\t{dist_str}"


def print_error_analysis(
    errors: Dict[str, Dict[str, List[float]]],
    epoch: Union[int, str] = None,
    point_names: Optional[Dict[str, str]] = None,
    title: Optional[str] = None
) -> None:
    """
    오차 분석을 직접 출력하는 헬퍼 함수

    display_error_analysis와 동일한 파라미터를 받아서 결과를 직접 출력
    """
    lines = display_error_analysis(errors, epoch, point_names, title)
    for line in lines:
        print(line)


def create_error_dict_from_arrays(
    x_errors: Union[List[float], np.ndarray],
    y_errors: Union[List[float], np.ndarray],
    point_name: str = 'point'
) -> Dict[str, Dict[str, List[float]]]:
    """
    X, Y 오차 배열로부터 오차 딕셔너리 생성

    Parameters:
    -----------
    x_errors : array-like
        X 좌표 오차 (절댓값)
    y_errors : array-like
        Y 좌표 오차 (절댓값)
    point_name : str
        포인트 이름

    Returns:
    --------
    Dict[str, Dict[str, List[float]]]
        display_error_analysis에서 사용할 수 있는 형식의 딕셔너리
    """
    x_errors = list(x_errors) if isinstance(x_errors, np.ndarray) else x_errors
    y_errors = list(y_errors) if isinstance(y_errors, np.ndarray) else y_errors

    # 유클리드 거리 계산
    dist_errors = [np.sqrt(x**2 + y**2) for x, y in zip(x_errors, y_errors)]

    return {
        point_name: {
            'x': x_errors,
            'y': y_errors,
            'dist': dist_errors
        }
    }


def merge_error_dicts(*error_dicts: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """
    여러 오차 딕셔너리를 하나로 병합

    Parameters:
    -----------
    *error_dicts : Dict[str, Dict[str, List[float]]]
        병합할 오차 딕셔너리들

    Returns:
    --------
    Dict[str, Dict[str, List[float]]]
        병합된 오차 딕셔너리
    """
    merged = {}

    for error_dict in error_dicts:
        for key, value in error_dict.items():
            if key not in merged:
                merged[key] = value
            else:
                # 키가 중복되는 경우 데이터 병합
                for axis in ['x', 'y', 'dist']:
                    if axis in value:
                        if axis not in merged[key]:
                            merged[key][axis] = []
                        merged[key][axis].extend(value[axis])

    return merged


def save_error_analysis_json(
    errors: Dict[str, Dict[str, List[float]]],
    epoch: Union[int, str],
    save_path: str,
    additional_info: Optional[Dict] = None
) -> None:
    """
    오차 분석 데이터를 JSON으로 저장

    Parameters:
    -----------
    errors : Dict[str, Dict[str, List[float]]]
        오차 데이터. 각 포인트별로 'x', 'y', 'dist' 리스트를 포함
    epoch : Union[int, str]
        에폭 번호 또는 'final' 같은 문자열
    save_path : str
        저장할 JSON 파일 경로
    additional_info : Dict, optional
        추가 정보 (loss, learning_rate 등)
    """
    # JSON 데이터 구성
    json_data = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'errors': {},  # 원본 오차 데이터 (리스트를 통계로 변환)
        'statistics': {},  # 통계 정보
        'formatted_output': [],  # display_error_analysis 출력
    }

    # 원본 데이터를 JSON 직렬화 가능한 형태로 변환 (리스트가 너무 길면 통계만 저장)
    for key in errors.keys():
        if key in errors:
            error_data = errors[key]
            # 데이터가 100개 이하면 원본 저장, 아니면 통계만 저장
            json_data['errors'][key] = {}
            for axis in ['x', 'y', 'dist']:
                if axis in error_data and error_data[axis]:
                    data = error_data[axis]
                    if len(data) <= 100:
                        # 원본 데이터 저장
                        json_data['errors'][key][axis] = [float(v) for v in data]
                    else:
                        # 통계만 저장
                        json_data['errors'][key][axis] = {
                            'count': len(data),
                            'mean': float(np.mean(data)),
                            'std': float(np.std(data)),
                            'min': float(np.min(data)),
                            'max': float(np.max(data)),
                            'percentile_25': float(np.percentile(data, 25)),
                            'percentile_50': float(np.percentile(data, 50)),
                            'percentile_75': float(np.percentile(data, 75))
                        }

    # 통계 계산 및 저장
    for key in errors.keys():
        if key in errors:
            stats = calculate_statistics(errors[key])
            # numpy float를 일반 float로 변환
            json_data['statistics'][key] = {
                axis: {
                    'mean': float(stats[axis]['mean']),
                    'std': float(stats[axis]['std'])
                } for axis in stats.keys()
            }

    # 포맷된 출력 저장
    json_data['formatted_output'] = display_error_analysis(errors, epoch)

    # 추가 정보 병합
    if additional_info:
        # numpy 타입을 일반 Python 타입으로 변환
        for key, value in additional_info.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                json_data[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                json_data[key] = int(value)
            else:
                json_data[key] = value

    # 디렉토리 생성
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # JSON 파일 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def load_error_analysis_json(json_path: str) -> Dict[str, Any]:
    """
    저장된 오차 분석 JSON 파일 로드

    Parameters:
    -----------
    json_path : str
        JSON 파일 경로

    Returns:
    --------
    Dict[str, Any]
        로드된 JSON 데이터
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_training_log_csv(log_data: List[Dict], filepath: str) -> None:
    """
    학습 로그를 CSV 파일로 저장

    Parameters:
    -----------
    log_data : List[Dict]
        학습 로그 데이터 (각 에폭별 딕셔너리 리스트)
    filepath : str
        저장할 CSV 파일 경로
    """
    import csv

    if not log_data:
        return

    # 디렉토리 생성
    save_dir = Path(filepath).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
        writer.writeheader()
        writer.writerows(log_data)

    print(f"학습 로그 저장: {filepath}")