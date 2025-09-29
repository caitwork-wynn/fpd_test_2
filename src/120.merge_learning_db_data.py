#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
120.merge_learning_db_data.py
DB에서 직접 학습 데이터를 생성하는 도구

PostgreSQL DB의 lb_shooting_image_object 테이블에서 데이터를 조회하여
통합된 학습 데이터셋을 data/learning에 생성합니다.
"""

import os
import sys
import argparse
import json
import base64
from pathlib import Path
from urllib.parse import unquote
import urllib.request
import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

# 프로젝트 루트 디렉토리 설정 (src의 상위 디렉토리)
ROOT_DIR = Path(__file__).parent.parent


def parse_arguments():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='DB에서 학습 데이터 생성')

    # DB 연결 정보 - 참조 파일의 기본값 사용
    parser.add_argument('--host',
                       default='cait-data-base.caitwork.co.kr',
                       help='DB 호스트 (default: cait-data-base.caitwork.co.kr)')
    parser.add_argument('--database',
                       default='labeling',
                       help='DB 이름 (default: labeling)')
    parser.add_argument('--user',
                       default='labeling_user',
                       help='DB 사용자 (default: labeling_user)')
    parser.add_argument('--password',
                       default='1243@db',
                       help='DB 비밀번호 (default: 1243@db)')
    parser.add_argument('--port',
                       type=int,
                       default=2002,
                       help='DB 포트 (default: 2002)')

    # 데이터 처리 옵션
    parser.add_argument('--shooting-id',
                       type=int,
                       default=0,
                       help='특정 shooting_id (0=전체) (default: 0)')
    parser.add_argument('--row-limit',
                       type=int,
                       default=0,
                       help='처리할 행 수 제한 (0=전체) (default: 0)')

    # 출력 경로 및 설정
    parser.add_argument('--output-folder',
                       default='data/learning',
                       help='출력 폴더 (default: data/learning)')
    parser.add_argument('--image-size',
                       type=int,
                       default=224,
                       help='출력 이미지 크기 (default: 224)')

    # 처리 모드
    parser.add_argument('--append',
                       action='store_true',
                       help='기존 데이터에 추가 (기본값: 초기화 후 생성)')

    return parser.parse_args()


def connect_to_db(host, database, user, password, port):
    """PostgreSQL 데이터베이스에 연결합니다."""
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
            cursor_factory=RealDictCursor
        )
        print(f"✓ DB 연결 성공: {database}@{host}:{port}")
        return conn
    except Exception as e:
        print(f"✗ DB 연결 실패: {e}")
        sys.exit(1)


def decode_base64_image(data_url):
    """base64로 인코딩된 이미지를 OpenCV 이미지로 변환합니다."""
    try:
        if not data_url:
            return None

        if data_url.startswith('data:image'):
            # data URL에서 base64 부분만 추출
            base64_data = data_url.split(',')[1]
            # base64 디코딩
            image_data = base64.b64decode(base64_data)
            # 바이트 배열로 변환
            nparr = np.frombuffer(image_data, np.uint8)
            # OpenCV 이미지로 디코딩
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        else:
            # URL 다운로드 (필요시)
            resp = urllib.request.urlopen(data_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        # 디버깅용 상세 정보
        if 'data:image' in str(data_url)[:20]:
            print(f"이미지 디코드 실패 (base64): {e}")
        else:
            print(f"이미지 디코드 실패 (URL): {e}")
        return None


def crop_and_resize_image(image, bbox, target_size):
    """이미지에서 bbox 영역을 크롭하고 리사이즈합니다."""
    try:
        x1, y1, x2, y2 = bbox

        # 이미지 경계 체크
        img_h, img_w = image.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_w, int(x2))
        y2 = min(img_h, int(y2))

        # Crop
        cropped = image[y1:y2, x1:x2]

        # 리사이즈
        if cropped.size > 0:
            resized = cv2.resize(cropped, (target_size, target_size))
            return resized
        return None
    except Exception as e:
        print(f"이미지 크롭/리사이즈 실패: {e}")
        return None


def convert_floor_coordinates(db_floor_x, db_floor_y, bbox, target_size):
    """DB의 floor 좌표(center_x, center_y)를 크롭된 이미지 좌표로 변환합니다."""
    x1, y1, x2, y2 = bbox

    # bbox 크기
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    if bbox_width <= 0 or bbox_height <= 0:
        return 0, 0

    # 상대 좌표로 변환 후 target_size에 맞게 스케일링
    floor_x = (db_floor_x - x1) / bbox_width * target_size
    floor_y = (db_floor_y - y1) / bbox_height * target_size

    # 경계 체크
    floor_x = max(0, min(target_size, floor_x))
    floor_y = max(0, min(target_size, floor_y))

    return floor_x, floor_y


def fetch_shooting_data(conn, shooting_id=0, row_limit=0):
    """shooting_id별로 데이터를 조회합니다."""
    cur = conn.cursor()

    # 조건 설정 - status 컬럼 존재 여부 확인 필요
    # lb_shooting_image_object 테이블은 status가 없을 수 있음
    where_clause = "WHERE 1=1"
    if shooting_id > 0:
        where_clause += f" AND shooting_id = {shooting_id}"

    # 쿼리 생성 - 참조 파일과 동일한 컬럼 사용
    query = f"""
        SELECT
            object_id,
            shooting_id,
            bbox_x1, bbox_y1,  -- bbox_x2, bbox_y2는 테이블에 없음
            center_x, center_y,  -- DB의 center는 실제 floor 좌표 (크롭된 이미지 기준)
            data_url,
            class_name,
            level,
            mask_percent,
            cut_percent,
            degree
        FROM lb_shooting_image_object
        {where_clause}
        ORDER BY object_id
    """

    if row_limit > 0:
        query += f" LIMIT {row_limit}"

    # 데이터 조회
    cur.execute(query)
    rows = cur.fetchall()

    print(f"✓ 조회된 데이터: {len(rows)}개")
    return rows


def get_all_shooting_ids(conn):
    """모든 고유한 shooting_id를 조회합니다."""
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT shooting_id
        FROM lb_shooting_image_object
        WHERE shooting_id IS NOT NULL
        ORDER BY shooting_id
    """)

    result = cur.fetchall()
    shooting_ids = [row['shooting_id'] for row in result]

    return shooting_ids


def process_data(rows, output_folder, image_size):
    """조회된 데이터를 처리하여 이미지와 labels.txt를 생성합니다."""
    # 출력 폴더 생성
    output_path = ROOT_DIR / output_folder
    output_path.mkdir(parents=True, exist_ok=True)

    # labels.txt 파일 경로
    labels_file = output_path / "labels.txt"

    # 파일 열기 모드 결정
    file_mode = 'a' if labels_file.exists() else 'w'
    write_header = not labels_file.exists()

    processed_count = 0
    failed_count = 0

    with open(labels_file, file_mode, encoding='utf-8') as f:
        # 헤더 작성
        if write_header:
            f.write("ID,투명도,파일명,center_x,center_y,floor_x,floor_y,front_x,front_y,side_x,side_y\n")

        # 데이터 처리
        for row in tqdm(rows, desc="데이터 처리", unit="개"):
            try:
                # 이미지 디코드
                data_url = row.get('data_url')
                if not data_url:
                    print(f"  경고: data_url이 비어있음 (object_id={row.get('object_id', 'unknown')})")
                    failed_count += 1
                    continue

                # lb_shooting_image_object의 이미지는 이미 크롭된 이미지
                image = decode_base64_image(data_url)
                if image is None:
                    # 디버깅: data_url의 처음 100자 출력
                    url_preview = str(data_url)[:100] if data_url else "None"
                    print(f"  경고: 이미지 디코드 실패 (object_id={row.get('object_id', 'unknown')}, url_preview={url_preview})")
                    failed_count += 1
                    continue

                # 이미지가 이미 크롭되어 있으므로 리사이즈만 수행
                # (참조 파일: lb_shooting_image_object는 이미 크롭된 이미지이므로 전체를 bbox로 설정)
                try:
                    resized_image = cv2.resize(image, (image_size, image_size))
                except Exception as e:
                    print(f"  경고: 이미지 리사이즈 실패 (object_id={row.get('object_id', 'unknown')}): {e}")
                    failed_count += 1
                    continue

                # DB의 center_x, center_y는 원본 이미지에서의 절대 좌표 (floor 좌표)
                # bbox_x1, bbox_y1을 빼서 크롭된 이미지 기준으로 변환해야 함
                db_center_x = row.get('center_x', 0)
                db_center_y = row.get('center_y', 0)
                bbox_x1 = row.get('bbox_x1', 0)
                bbox_y1 = row.get('bbox_y1', 0)

                # 참조 파일 로직: center에서 bbox_x1, bbox_y1을 빼서 크롭된 이미지 기준으로 변환
                cropped_floor_x = db_center_x - bbox_x1
                cropped_floor_y = db_center_y - bbox_y1

                # 크롭된 이미지 크기에서 target_size(224)로 스케일링
                img_h, img_w = image.shape[:2]
                if img_w > 0 and img_h > 0:
                    floor_x = (cropped_floor_x * image_size) / img_w
                    floor_y = (cropped_floor_y * image_size) / img_h

                    # 경계 체크 - 좌표가 이미지 밖에 있을 수 있음
                    # 이 경우 경계값으로 클리핑
                    floor_x = max(0, min(image_size - 1, floor_x))
                    floor_y = max(0, min(image_size - 1, floor_y))
                else:
                    floor_x = floor_y = 0

                # 디버깅 정보 (첫 번째 항목만)
                if processed_count == 0:
                    print(f"\n[디버깅] object_id={row.get('object_id')}:")
                    print(f"  원본 center: ({db_center_x}, {db_center_y})")
                    print(f"  bbox 시작: ({bbox_x1}, {bbox_y1})")
                    print(f"  크롭 이미지 center: ({cropped_floor_x}, {cropped_floor_y})")
                    print(f"  이미지 크기: {img_w}x{img_h}")
                    print(f"  최종 floor: ({floor_x:.1f}, {floor_y:.1f})")

                # 파일명 생성
                shooting_id = row.get('shooting_id', 0)
                object_id = row['object_id']
                filename = f"{shooting_id:06d}-obj{object_id:08d}.jpg"

                # 이미지 저장 (리사이즈된 이미지)
                image_path = output_path / filename
                cv2.imwrite(str(image_path), resized_image)

                # labels.txt에 데이터 추가
                # ID, 투명도(mask_percent 사용), 파일명
                record_id = f"{shooting_id:06d}-{object_id:08d}"
                # mask_percent를 투명도로 사용 (참조 파일 로직)
                transparency = row.get('mask_percent', 1.0)

                # center, front, side는 모두 0
                center_x = center_y = 0
                front_x = front_y = 0
                side_x = side_y = 0

                # 레코드 작성
                line = f"{record_id},{transparency:.2f},{filename},"
                line += f"{center_x},{center_y},"
                line += f"{floor_x:.1f},{floor_y:.1f},"
                line += f"{front_x},{front_y},"
                line += f"{side_x},{side_y}\n"

                f.write(line)
                processed_count += 1

            except Exception as e:
                print(f"처리 실패 (object_id={row.get('object_id', 'unknown')}): {e}")
                failed_count += 1
                continue

    return processed_count, failed_count


def main():
    """메인 실행 함수"""
    print("=== DB 학습 데이터 생성 도구 ===\n")

    # 인자 파싱
    args = parse_arguments()

    # DB 연결
    conn = connect_to_db(
        args.host,
        args.database,
        args.user,
        args.password,
        args.port
    )

    try:
        # 출력 폴더 확인
        output_path = ROOT_DIR / args.output_folder
        labels_file = output_path / "labels.txt"

        # 기존 데이터 확인
        if labels_file.exists() and not args.append:
            existing_records = sum(1 for _ in open(labels_file, 'r', encoding='utf-8')) - 1  # 헤더 제외
            if existing_records > 0:
                print(f"기존 레코드가 {existing_records}개 있습니다.")
                action = input("어떻게 처리하시겠습니까? (a: 추가, r: 초기화, c: 취소): ").strip().lower()

                if action == 'c':
                    print("작업을 취소합니다.")
                    sys.exit(0)
                elif action == 'r':
                    # 초기화
                    labels_file.unlink()
                    # 이미지 파일도 삭제
                    for img_file in output_path.glob("*.jpg"):
                        img_file.unlink()
                    for img_file in output_path.glob("*.png"):
                        img_file.unlink()
                    print("기존 데이터를 모두 삭제했습니다.")
                elif action != 'a':
                    print("잘못된 선택입니다. 작업을 취소합니다.")
                    sys.exit(0)

        # shooting_id 처리
        if args.shooting_id == 0:
            # 모든 shooting_id 처리
            shooting_ids = get_all_shooting_ids(conn)
            if not shooting_ids:
                print("처리할 shooting_id가 없습니다.")
                sys.exit(0)

            print(f"\n발견된 shooting_id: {shooting_ids}")
            print(f"총 {len(shooting_ids)}개의 shooting_id를 처리합니다.\n")

            total_processed = 0
            total_failed = 0

            for idx, sid in enumerate(shooting_ids, 1):
                print(f"\n[{idx}/{len(shooting_ids)}] shooting_id {sid} 처리 중...")
                print("=" * 60)

                # 데이터 조회
                rows = fetch_shooting_data(conn, sid, args.row_limit)

                if rows:
                    # 데이터 처리
                    processed, failed = process_data(rows, args.output_folder, args.image_size)
                    total_processed += processed
                    total_failed += failed

                    print(f"✓ shooting_id {sid}: 성공 {processed}개, 실패 {failed}개")
                else:
                    print(f"✗ shooting_id {sid}: 데이터 없음")

            print("\n" + "=" * 60)
            print(f"전체 처리 완료: 성공 {total_processed}개, 실패 {total_failed}개")

        else:
            # 특정 shooting_id만 처리
            print(f"shooting_id {args.shooting_id} 처리 중...")

            # 데이터 조회
            rows = fetch_shooting_data(conn, args.shooting_id, args.row_limit)

            if rows:
                # 데이터 처리
                processed, failed = process_data(rows, args.output_folder, args.image_size)
                print(f"\n✓ 처리 완료: 성공 {processed}개, 실패 {failed}개")
            else:
                print(f"✗ shooting_id {args.shooting_id}에 대한 데이터가 없습니다.")

        # 최종 결과 출력
        if labels_file.exists():
            total_records = sum(1 for _ in open(labels_file, 'r', encoding='utf-8')) - 1  # 헤더 제외
            print(f"\n최종 labels.txt 레코드 수: {total_records}개")
            print(f"저장 위치: {output_path}")

    except KeyboardInterrupt:
        print("\n\n작업이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # DB 연결 종료
        if conn:
            conn.close()
            print("\nDB 연결이 종료되었습니다.")


if __name__ == "__main__":
    main()