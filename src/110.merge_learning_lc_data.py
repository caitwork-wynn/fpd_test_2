#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
300.merge_learning_data.py
학습 데이터 병합 도구

data/base의 여러 폴더에서 labels.txt와 이미지를 읽어
통합된 학습 데이터셋을 data/learning에 생성합니다.
"""

import os
import shutil
from pathlib import Path
import sys
import json
import cv2
import numpy as np

# 프로젝트 루트 디렉토리 설정 (src의 상위 디렉토리)
ROOT_DIR = Path(__file__).parent.parent

def list_and_select_folders():
    """data/base 디렉토리의 폴더 목록을 표시하고 다수 선택을 받습니다."""
    base_path = ROOT_DIR / "data" / "base"
    
    if not base_path.exists():
        print(f"오류: {base_path} 디렉토리가 존재하지 않습니다.")
        sys.exit(1)
    
    # 폴더 목록 가져오기
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"오류: {base_path}에 폴더가 없습니다.")
        sys.exit(1)
    
    # 폴더 목록 출력
    print("\n=== data/base 폴더 목록 ===")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder.name}")
    
    print("\n선택 방법:")
    print("  - 단일 선택: 번호 하나 입력 (예: 3)")
    print("  - 다중 선택: 쉼표로 구분 (예: 1,3,5)")
    print("  - 범위 선택: 하이픈 사용 (예: 2-5)")
    print("  - 혼합 선택: (예: 1,3-5,7)")
    print("  - 전체 선택: *")
    print("  - 종료: 0")
    
    # 사용자 선택
    while True:
        try:
            choice = input("\n병합할 폴더를 선택하세요: ").strip()
            
            if choice == "0":
                print("프로그램을 종료합니다.")
                sys.exit(0)
            
            if choice == "*":
                return folders
            
            selected_folders = []
            selected_indices = set()
            
            # 선택 항목 파싱
            parts = choice.split(',')
            for part in parts:
                part = part.strip()
                
                if '-' in part:
                    # 범위 선택
                    try:
                        start, end = part.split('-')
                        start_idx = int(start.strip()) - 1
                        end_idx = int(end.strip()) - 1
                        
                        if start_idx < 0 or end_idx >= len(folders):
                            print(f"범위가 올바르지 않습니다: {part}")
                            continue
                        
                        for idx in range(start_idx, end_idx + 1):
                            selected_indices.add(idx)
                    except ValueError:
                        print(f"잘못된 범위 형식입니다: {part}")
                        continue
                else:
                    # 단일 선택
                    try:
                        idx = int(part) - 1
                        if 0 <= idx < len(folders):
                            selected_indices.add(idx)
                        else:
                            print(f"잘못된 번호입니다: {part}")
                            continue
                    except ValueError:
                        print(f"숫자가 아닙니다: {part}")
                        continue
            
            if selected_indices:
                selected_folders = [folders[idx] for idx in sorted(selected_indices)]
                print(f"\n선택된 폴더 ({len(selected_folders)}개):")
                for folder in selected_folders:
                    print(f"  - {folder.name}")
                
                confirm = input("\n이대로 진행하시겠습니까? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected_folders
                else:
                    print("다시 선택해주세요.")
            else:
                print("선택된 폴더가 없습니다. 다시 선택해주세요.")
                
        except Exception as e:
            print(f"입력 오류: {e}")
            print("다시 입력해주세요.")

def detect_format(folder_path):
    """폴더의 데이터 형식을 자동으로 판별합니다."""
    labels_path = folder_path / "labels.txt"
    frames_json_path = folder_path / "frames.json"

    if labels_path.exists():
        # 기존 labels.txt 형식 감지
        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 빈 파일 체크
        if len(lines) < 2:
            print(f"  경고: {labels_path}가 비어있거나 데이터가 없습니다.")
            return None

        # 첫 번째 데이터 라인 파싱 (헤더 다음 라인)
        data_line = lines[1].strip()
        columns = data_line.split(',')
        num_columns = len(columns)

        if num_columns == 11:
            return 1  # 유형 1 (4 point)
        elif num_columns == 5:
            return 2  # 유형 2 (1 point)
        else:
            print(f"  경고: 알 수 없는 형식입니다. (컬럼 수: {num_columns})")
            return None
    elif frames_json_path.exists():
        # frames.json 파일이 있으면 유형 3
        return 3
    else:
        # 둘 다 없으면 None
        return None

def calculate_floor_coords(detected, bbox, target_size):
    """detected 좌표를 crop된 이미지의 좌표로 변환합니다."""
    floor_x = (detected['x'] - bbox['x']) / bbox['width'] * target_size
    floor_y = (detected['y'] - bbox['y']) / bbox['height'] * target_size
    return floor_x, floor_y

def crop_and_resize_image(image, bbox, target_size):
    """이미지에서 bbox 영역을 crop하고 리사이즈합니다."""
    x = max(0, int(bbox['x']))
    y = max(0, int(bbox['y']))
    w = int(bbox['width'])
    h = int(bbox['height'])

    # 이미지 경계 체크
    img_h, img_w = image.shape[:2]
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # Crop
    cropped = image[y:y2, x:x2]

    # 리사이즈
    if cropped.size > 0:
        resized = cv2.resize(cropped, (target_size, target_size))
        return resized
    return None

def parse_frames_json(folder_path, folder_name, target_size=224):
    """frames.json을 읽어 user.json 파일들을 파싱하고 이미지를 처리합니다."""
    data = []
    frames_json_path = folder_path / "frames.json"

    # frames.json 읽기
    try:
        with open(frames_json_path, 'r', encoding='utf-8') as f:
            frames_data = json.load(f)
    except Exception as e:
        print(f"    경고: frames.json 읽기 실패: {e}")
        return data

    frames = frames_data.get('frames', [])
    total_frames = len(frames)
    processed_images = 0
    skipped_images = 0

    print(f"  총 프레임 수: {total_frames}")

    # 각 프레임 처리
    for frame_idx, frame in enumerate(frames):
        timestamp = frame.get('timestamp', '').replace(':', '').replace('-', '')
        images = frame.get('images', {})

        # 각 CCTV 이미지 처리
        for cctv_no, image_rel_path in images.items():
            # 이미지 경로 구성
            image_path = folder_path / image_rel_path
            user_json_path = folder_path / f"{image_rel_path}.user.json"

            # user.json이 없으면 건너뛰기
            if not user_json_path.exists():
                skipped_images += 1
                continue

            # 이미지 파일이 없으면 건너뛰기
            if not image_path.exists():
                print(f"    경고: 이미지 파일을 찾을 수 없습니다: {image_path.name}")
                skipped_images += 1
                continue

            # user.json 읽기
            try:
                with open(user_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"    경고: JSON 파일 읽기 실패: {user_json_path.name} - {e}")
                skipped_images += 1
                continue

            # 이미지 로드
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"    경고: 이미지 로드 실패: {image_path.name}")
                skipped_images += 1
                continue

            processed_images += 1

            # 각 객체 처리
            for obj in json_data.get('objects', []):
                obj_id = obj.get('objectId', obj.get('id'))
                bbox = obj.get('bbox')
                detected = obj.get('detected')

                if not bbox or not detected:
                    continue

                # 이미지 crop 및 리사이즈
                cropped_image = crop_and_resize_image(image, bbox, target_size)
                if cropped_image is None:
                    continue

                # floor 좌표 계산
                floor_x, floor_y = calculate_floor_coords(detected, bbox, target_size)

                # 중심점은 crop된 이미지의 중앙
                center_x = center_y = target_size / 2

                # 파일명 생성
                filename = f"{folder_name}-{timestamp}-{cctv_no}-obj{obj_id}.jpg"

                record = {
                    'id': f"{folder_name}-{timestamp}-{cctv_no}-{obj_id}",
                    'transparency': str(obj.get('confidence', 1.0)),
                    'filename': filename,
                    'center_x': str(center_x),
                    'center_y': str(center_y),
                    'floor_x': str(floor_x),
                    'floor_y': str(floor_y),
                    'front_x': str(floor_x),  # 현재는 floor와 동일
                    'front_y': str(floor_y),
                    'side_x': str(floor_x),
                    'side_y': str(floor_y),
                    'cropped_image': cropped_image,
                    'object_type': obj.get('objectType', 'UNKNOWN')
                }

                data.append(record)

        # 진행 상황 표시
        if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
            print(f"    프레임 처리: {frame_idx + 1}/{total_frames}")

    print(f"  처리된 이미지: {processed_images}, 건너뛴 이미지: {skipped_images}")

    return data

def parse_labels(labels_path, format_type, folder_name):
    """labels.txt 파일을 파싱하여 통합 형식으로 변환합니다."""
    data = []
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 헤더 건너뛰기
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        columns = line.split(',')
        
        if format_type == 1:
            # 유형 1: 이미 4 point 형식
            # ID,투명도,파일명,center_x,center_y,floor_x,floor_y,front_x,front_y,side_x,side_y
            record = {
                'id': f"{folder_name}-{columns[0]}",
                'transparency': columns[1],
                'filename': f"{folder_name}-{columns[2]}",
                'center_x': columns[3],
                'center_y': columns[4],
                'floor_x': columns[5],
                'floor_y': columns[6],
                'front_x': columns[7],
                'front_y': columns[8],
                'side_x': columns[9],
                'side_y': columns[10],
                'original_filename': columns[2]
            }
        
        elif format_type == 2:
            # 유형 2: 1 point를 4 point로 변환
            # ID,class,파일명,x,y
            x, y = columns[3], columns[4]
            record = {
                'id': f"{folder_name}-{columns[0]}",
                'transparency': columns[1],  # class를 transparency 위치에
                'filename': f"{folder_name}-{columns[2]}",
                'center_x': x,
                'center_y': y,
                'floor_x': x,  # 같은 값으로 복제
                'floor_y': y,
                'front_x': x,
                'front_y': y,
                'side_x': x,
                'side_y': y,
                'original_filename': columns[2]
            }
        
        data.append(record)
    
    return data

def copy_images(data, source_folder, learning_path, format_type=None):
    """이미지 파일을 data/learning으로 복사합니다."""
    copied = 0
    failed = 0

    for record in data:
        dest_file = learning_path / record['filename']

        if format_type == 3 and 'cropped_image' in record:
            # 유형 3: crop된 이미지 저장
            try:
                cv2.imwrite(str(dest_file), record['cropped_image'])
                copied += 1
            except Exception as e:
                print(f"    경고: {record['filename']} 저장 실패: {e}")
                failed += 1
        else:
            # 유형 1, 2: 원본 이미지 복사
            source_file = source_folder / record.get('original_filename', record['filename'])

            if source_file.exists():
                try:
                    shutil.copy2(source_file, dest_file)
                    copied += 1
                except Exception as e:
                    print(f"    경고: {source_file.name} 복사 실패: {e}")
                    failed += 1
            else:
                # 경고 메시지 최소화 (많은 파일 처리 시)
                failed += 1

    return copied, failed

def update_merged_labels(data, learning_path):
    """통합 labels.txt 파일을 업데이트합니다."""
    labels_file = learning_path / "labels.txt"
    
    # 헤더 확인 및 작성
    write_header = not labels_file.exists() or labels_file.stat().st_size == 0
    
    with open(labels_file, 'a', encoding='utf-8') as f:
        if write_header:
            # 유형 1 형식의 헤더 작성
            f.write("ID,투명도,파일명,center_x,center_y,floor_x,floor_y,front_x,front_y,side_x,side_y\n")
        
        # 데이터 추가
        for record in data:
            line = f"{record['id']},{record['transparency']},{record['filename']},"
            line += f"{record['center_x']},{record['center_y']},"
            line += f"{record['floor_x']},{record['floor_y']},"
            line += f"{record['front_x']},{record['front_y']},"
            line += f"{record['side_x']},{record['side_y']}\n"
            f.write(line)
    
    return len(data)

def process_single_folder(folder, learning_path):
    """단일 폴더를 처리합니다."""
    folder_name = folder.name
    print(f"\n처리 중: {folder_name}")

    # 형식 자동 판별
    format_type = detect_format(folder)
    if format_type is None:
        print(f"  경고: {folder_name}에 처리할 수 있는 데이터가 없습니다. 건너뜁니다.")
        return None

    # 형식별 처리
    if format_type == 3:
        # 유형 3: frames.json 기반 처리
        format_name = "유형 3 (frames.json)"
        print(f"  형식: {format_name}")

        # frames.json 파싱 및 이미지 처리
        data = parse_frames_json(folder, folder_name)
        print(f"  파싱된 레코드: {len(data)}개")

        # crop된 이미지 저장
        copied, failed = copy_images(data, folder, learning_path, format_type)
        print(f"  이미지 저장: 성공 {copied}개, 실패 {failed}개")
    else:
        # 유형 1, 2: 기존 labels.txt 처리
        labels_path = folder / "labels.txt"
        format_name = "유형 1 (4 point)" if format_type == 1 else "유형 2 (1 point)"
        print(f"  형식: {format_name}")

        # labels.txt 파싱 및 변환
        data = parse_labels(labels_path, format_type, folder_name)
        print(f"  파싱된 레코드: {len(data)}개")

        # 이미지 파일 복사
        copied, failed = copy_images(data, folder, learning_path, format_type)
        print(f"  이미지 복사: 성공 {copied}개, 실패 {failed}개")

    # 통합 labels.txt 업데이트
    added = update_merged_labels(data, learning_path)

    return {
        'folder': folder_name,
        'format': format_name,
        'records': len(data),
        'copied': copied,
        'failed': failed
    }

def main():
    """메인 실행 함수"""
    print("=== 학습 데이터 병합 도구 ===")
    print("data/base의 폴더를 선택하여 data/learning으로 병합합니다.")
    
    # 1. 폴더 선택 (다중 선택 가능)
    selected_folders = list_and_select_folders()
    
    # 2. data/learning 디렉토리 생성
    learning_path = ROOT_DIR / "data" / "learning"
    learning_path.mkdir(parents=True, exist_ok=True)
    
    # 3. 기존 데이터 확인
    merged_labels = learning_path / "labels.txt"
    existing_records = 0
    if merged_labels.exists():
        with open(merged_labels, 'r', encoding='utf-8') as f:
            existing_records = sum(1 for _ in f) - 1  # 헤더 제외
        
        if existing_records > 0:
            print(f"\n기존 레코드가 {existing_records}개 있습니다.")
            action = input("어떻게 처리하시겠습니까? (a: 추가, r: 초기화, c: 취소): ").strip().lower()
            
            if action == 'c':
                print("작업을 취소합니다.")
                sys.exit(0)
            elif action == 'r':
                # 초기화
                merged_labels.unlink()
                # 이미지 파일도 삭제할지 확인
                delete_images = input("이미지 파일도 모두 삭제하시겠습니까? (y/n): ").strip().lower()
                if delete_images == 'y':
                    for img_file in learning_path.glob("*.png"):
                        img_file.unlink()
                    for img_file in learning_path.glob("*.jpg"):
                        img_file.unlink()
                    print("기존 데이터를 모두 삭제했습니다.")
                else:
                    print("labels.txt만 초기화했습니다.")
    
    # 4. 각 폴더 처리
    print(f"\n=== {len(selected_folders)}개 폴더 처리 시작 ===")
    
    results = []
    total_records = 0
    total_copied = 0
    total_failed = 0
    
    for folder in selected_folders:
        result = process_single_folder(folder, learning_path)
        if result:
            results.append(result)
            total_records += result['records']
            total_copied += result['copied']
            total_failed += result['failed']
    
    # 5. 최종 보고
    print(f"\n=== 병합 완료 ===")
    print(f"처리된 폴더: {len(results)}개 / {len(selected_folders)}개")
    print(f"총 레코드: {total_records}개")
    print(f"복사된 이미지: {total_copied}개")
    if total_failed > 0:
        print(f"실패한 이미지: {total_failed}개")
    
    # 폴더별 요약
    if len(results) > 0:
        print(f"\n=== 폴더별 요약 ===")
        for result in results:
            print(f"{result['folder']}:")
            print(f"  - 형식: {result['format']}")
            print(f"  - 레코드: {result['records']}개")
            if result['format'] == "유형 3 (frames.json)":
                print(f"  - 저장: {result['copied']}개")
            else:
                print(f"  - 복사: {result['copied']}개")
            if result['failed'] > 0:
                print(f"  - 실패: {result['failed']}개")
    
    # 현재 통합 파일 상태
    if merged_labels.exists():
        with open(merged_labels, 'r', encoding='utf-8') as f:
            final_total = sum(1 for _ in f) - 1  # 헤더 제외
        print(f"\n통합 labels.txt 전체 레코드 수: {final_total}개")

if __name__ == "__main__":
    main()