#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_labels.py
학습 데이터의 labels.txt 파일을 검증하는 프로그램

체크 항목:
1. 해당 이미지가 없다
2. 각 좌표가 허용 범위를 벗어난다 (이미지 크기의 ±10% 여유 허용)
3. 좌표가 유효한 숫자가 아니다 (정수 또는 실수 허용)
"""

import os
import csv
from pathlib import Path
from PIL import Image
from collections import defaultdict
import json
import sys

class LabelChecker:
    def __init__(self, labels_file, images_dir):
        """
        레이블 검증기 초기화

        Args:
            labels_file: labels.txt 파일 경로
            images_dir: 이미지 파일들이 있는 디렉토리 경로
        """
        self.labels_file = Path(labels_file)
        self.images_dir = Path(images_dir)
        self.errors = {
            'missing_images': [],      # 이미지가 없는 경우
            'out_of_bounds': [],        # 좌표가 범위를 벗어난 경우
            'invalid_coords': [],       # 좌표가 유효한 숫자가 아닌 경우
            'other_errors': []          # 기타 오류
        }
        self.stats = {
            'total_labels': 0,
            'valid_labels': 0,
            'error_labels': 0,
            'image_sizes': defaultdict(int)
        }

    def is_numeric(self, value_str):
        """문자열이 유효한 숫자(정수 또는 실수)인지 확인"""
        try:
            float(value_str)
            return True
        except:
            return False

    def get_image_size(self, image_path):
        """이미지의 실제 크기를 반환"""
        try:
            with Image.open(image_path) as img:
                return img.width, img.height
        except Exception as e:
            return None, None

    def check_single_label(self, line_num, row):
        """
        단일 레이블을 검증

        Args:
            line_num: 라인 번호
            row: CSV 행 데이터

        Returns:
            bool: 유효한 레이블이면 True, 오류가 있으면 False
        """
        # 5개 컬럼(floor만) 또는 11개 컬럼(모든 좌표) 지원
        if len(row) not in [5, 11]:
            self.errors['other_errors'].append({
                'line': line_num,
                'error': f'잘못된 필드 개수: {len(row)}개 (5개 또는 11개여야 함)',
                'data': row
            })
            return False

        try:
            # 레이블 파싱
            id_val = row[0]
            class_or_transparency = row[1]
            filename = row[2]

            # 좌표 필드 (형식에 따라 다름)
            if len(row) == 5:
                # 5개 컬럼: ID,class,파일명,floor_x,floor_y
                coord_fields = {
                    'floor_x': row[3],
                    'floor_y': row[4]
                }
            else:
                # 11개 컬럼: ID,투명도,파일명,center_x,center_y,floor_x,floor_y,front_x,front_y,side_x,side_y
                coord_fields = {
                    'center_x': row[3],
                    'center_y': row[4],
                    'floor_x': row[5],
                    'floor_y': row[6],
                    'front_x': row[7],
                    'front_y': row[8],
                    'side_x': row[9],
                    'side_y': row[10]
                }

            has_error = False

            # 1. 이미지 파일 존재 확인
            image_path = self.images_dir / filename
            if not image_path.exists():
                self.errors['missing_images'].append({
                    'line': line_num,
                    'id': id_val,
                    'filename': filename,
                    'error': '이미지 파일이 존재하지 않음'
                })
                has_error = True
                # 이미지가 없으면 좌표 범위 체크 불가
                img_width, img_height = None, None
            else:
                # 이미지 크기 가져오기
                img_width, img_height = self.get_image_size(image_path)
                if img_width and img_height:
                    self.stats['image_sizes'][f'{img_width}x{img_height}'] += 1

            # 2. 좌표가 유효한 숫자인지 확인
            for coord_name, coord_value in coord_fields.items():
                # 숫자 체크 (정수 또는 실수)
                if not self.is_numeric(coord_value):
                    self.errors['invalid_coords'].append({
                        'line': line_num,
                        'id': id_val,
                        'filename': filename,
                        'coord': coord_name,
                        'value': coord_value,
                        'error': f'{coord_name}이 유효한 숫자가 아님'
                    })
                    has_error = True

                # 3. 이미지가 있는 경우에만 범위 체크 (10% 여유 허용)
                if img_width and img_height:
                    try:
                        coord_val = float(coord_value)

                        # 10% 여유 범위 계산
                        margin_x = img_width * 0.1
                        margin_y = img_height * 0.1
                        min_x = -margin_x
                        max_x = img_width - 1 + margin_x
                        min_y = -margin_y
                        max_y = img_height - 1 + margin_y

                        # X 좌표 체크
                        if '_x' in coord_name:
                            if coord_val < min_x or coord_val > max_x:
                                self.errors['out_of_bounds'].append({
                                    'line': line_num,
                                    'id': id_val,
                                    'filename': filename,
                                    'coord': coord_name,
                                    'value': coord_val,
                                    'valid_range': f'{min_x:.1f}~{max_x:.1f} (10% 여유)',
                                    'image_size': f'{img_width}x{img_height}',
                                    'error': f'{coord_name}={coord_val:.1f}가 허용 범위({min_x:.1f}~{max_x:.1f})를 벗어남'
                                })
                                has_error = True

                        # Y 좌표 체크
                        elif '_y' in coord_name:
                            if coord_val < min_y or coord_val > max_y:
                                self.errors['out_of_bounds'].append({
                                    'line': line_num,
                                    'id': id_val,
                                    'filename': filename,
                                    'coord': coord_name,
                                    'value': coord_val,
                                    'valid_range': f'{min_y:.1f}~{max_y:.1f} (10% 여유)',
                                    'image_size': f'{img_width}x{img_height}',
                                    'error': f'{coord_name}={coord_val:.1f}가 허용 범위({min_y:.1f}~{max_y:.1f})를 벗어남'
                                })
                                has_error = True
                    except:
                        pass  # 이미 invalid_coords에서 처리됨

            return not has_error

        except Exception as e:
            self.errors['other_errors'].append({
                'line': line_num,
                'error': f'처리 중 오류 발생: {str(e)}',
                'data': row
            })
            return False

    def check_all_labels(self):
        """모든 레이블을 검증"""
        print(f"레이블 파일 검증 시작: {self.labels_file}")
        print(f"이미지 디렉토리: {self.images_dir}")
        print("-" * 80)

        if not self.labels_file.exists():
            print(f"오류: {self.labels_file} 파일이 존재하지 않습니다.")
            return

        with open(self.labels_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            # 헤더 건너뛰기
            header = next(reader)

            # 각 레이블 검증
            for line_num, row in enumerate(reader, start=2):
                self.stats['total_labels'] += 1

                if self.check_single_label(line_num, row):
                    self.stats['valid_labels'] += 1
                else:
                    self.stats['error_labels'] += 1

                # 진행 상황 표시
                if self.stats['total_labels'] % 10000 == 0:
                    print(f"  처리 중... {self.stats['total_labels']:,}개 완료")

        print(f"\n검증 완료: 총 {self.stats['total_labels']:,}개 레이블 처리")

    def print_report(self):
        """검증 결과 보고서 출력"""
        print("\n" + "=" * 80)
        print("레이블 검증 결과 보고서")
        print("=" * 80)

        # 전체 통계
        print("\n[전체 통계]")
        print(f"- 전체 레이블 수: {self.stats['total_labels']:,}개")
        print(f"- 정상 레이블: {self.stats['valid_labels']:,}개 ({self.stats['valid_labels']/max(1,self.stats['total_labels'])*100:.1f}%)")
        print(f"- 오류 레이블: {self.stats['error_labels']:,}개 ({self.stats['error_labels']/max(1,self.stats['total_labels'])*100:.1f}%)")

        # 오류 종류별 개수만 표시
        print("\n[오류 종류별 개수]")
        print("-" * 40)
        print(f"1. 이미지 파일 없음: {len(self.errors['missing_images']):,}개")
        print(f"2. 좌표가 허용 범위를 벗어남 (10% 여유 초과): {len(self.errors['out_of_bounds']):,}개")
        print(f"3. 좌표가 유효한 숫자가 아님: {len(self.errors['invalid_coords']):,}개")
        print(f"4. 기타 오류: {len(self.errors['other_errors']):,}개")

        # 요약
        print("\n" + "=" * 80)
        total_errors = (len(self.errors['missing_images']) +
                       len(self.errors['out_of_bounds']) +
                       len(self.errors['invalid_coords']) +
                       len(self.errors['other_errors']))

        if total_errors == 0:
            print("✓ 모든 레이블이 정상입니다!")
        else:
            print(f"⚠ 총 {total_errors:,}개의 오류가 발견되었습니다.")
            print("\n[권장 조치사항]")
            if self.errors['missing_images']:
                print("- 누락된 이미지 파일 확인 및 복사")
            if self.errors['out_of_bounds']:
                print("- 허용 범위(10% 여유)를 초과하는 좌표 클리핑 또는 수정")
            if self.errors['invalid_coords']:
                print("- 숫자가 아닌 좌표 값 수정")

    def save_error_report(self):
        """오류 보고서를 result 폴더에 JSON 파일로 저장"""
        # result 폴더 생성
        result_dir = Path(__file__).parent.parent / "result"
        result_dir.mkdir(exist_ok=True)

        # 현재 시간을 포함한 파일명 생성
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = result_dir / f"label_check_report_{timestamp}.json"

        report = {
            'timestamp': timestamp,
            'labels_file': str(self.labels_file),
            'images_dir': str(self.images_dir),
            'stats': dict(self.stats),
            'errors': {
                'missing_images': self.errors['missing_images'],
                'out_of_bounds': self.errors['out_of_bounds'],
                'invalid_coords': self.errors['invalid_coords'],
                'other_errors': self.errors['other_errors']
            },
            'summary': {
                'total_labels': self.stats['total_labels'],
                'valid_labels': self.stats['valid_labels'],
                'error_labels': self.stats['error_labels'],
                'total_missing_images': len(self.errors['missing_images']),
                'total_out_of_bounds': len(self.errors['out_of_bounds']),
                'total_invalid_coords': len(self.errors['invalid_coords']),
                'total_other_errors': len(self.errors['other_errors'])
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n상세 오류 보고서가 저장되었습니다:")
        print(f"  파일: {output_file.name}")
        print(f"  경로: {output_file.parent}")

        # 간단한 텍스트 보고서도 저장
        txt_output_file = result_dir / f"label_check_summary_{timestamp}.txt"
        with open(txt_output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("레이블 검증 결과 요약\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"검증 시간: {timestamp}\n")
            f.write(f"레이블 파일: {self.labels_file}\n")
            f.write(f"이미지 디렉토리: {self.images_dir}\n\n")

            f.write("[전체 통계]\n")
            f.write(f"- 전체 레이블 수: {self.stats['total_labels']:,}개\n")
            f.write(f"- 정상 레이블: {self.stats['valid_labels']:,}개\n")
            f.write(f"- 오류 레이블: {self.stats['error_labels']:,}개\n\n")

            f.write("[오류 종류별 개수]\n")
            f.write(f"1. 이미지 파일 없음: {len(self.errors['missing_images']):,}개\n")
            f.write(f"2. 좌표가 허용 범위를 벗어남 (10% 여유 초과): {len(self.errors['out_of_bounds']):,}개\n")
            f.write(f"3. 좌표가 유효한 숫자가 아님: {len(self.errors['invalid_coords']):,}개\n")
            f.write(f"4. 기타 오류: {len(self.errors['other_errors']):,}개\n")

            # 오류 상세 내용 추가
            if self.errors['missing_images']:
                f.write("\n[이미지 파일 없음 상세]\n")
                for i, err in enumerate(self.errors['missing_images'][:100], 1):
                    f.write(f"  {i}. 라인 {err['line']}: {err['filename']} (ID: {err['id']})\n")
                if len(self.errors['missing_images']) > 100:
                    f.write(f"  ... 그 외 {len(self.errors['missing_images'])-100}개\n")

            if self.errors['out_of_bounds']:
                f.write("\n[허용 범위 초과 좌표 상세]\n")
                for i, err in enumerate(self.errors['out_of_bounds'][:100], 1):
                    f.write(f"  {i}. 라인 {err['line']}: {err['error']} - {err['filename']}\n")
                if len(self.errors['out_of_bounds']) > 100:
                    f.write(f"  ... 그 외 {len(self.errors['out_of_bounds'])-100}개\n")

            if self.errors['invalid_coords']:
                f.write("\n[유효하지 않은 좌표 상세]\n")
                for i, err in enumerate(self.errors['invalid_coords'][:100], 1):
                    f.write(f"  {i}. 라인 {err['line']}: {err['coord']}={err['value']} - {err['filename']}\n")
                if len(self.errors['invalid_coords']) > 100:
                    f.write(f"  ... 그 외 {len(self.errors['invalid_coords'])-100}개\n")

        print(f"  요약: {txt_output_file.name}")

    def get_error_lines(self):
        """모든 오류가 있는 라인 번호를 수집"""
        error_lines = set()

        for err in self.errors['missing_images']:
            error_lines.add(err['line'])
        for err in self.errors['out_of_bounds']:
            error_lines.add(err['line'])
        for err in self.errors['invalid_coords']:
            error_lines.add(err['line'])
        for err in self.errors['other_errors']:
            error_lines.add(err['line'])

        return sorted(error_lines)

    def get_error_images(self):
        """오류가 있는 모든 이미지 파일명을 수집"""
        error_images = set()

        for err in self.errors['missing_images']:
            error_images.add(err['filename'])
        for err in self.errors['out_of_bounds']:
            error_images.add(err['filename'])
        for err in self.errors['invalid_coords']:
            error_images.add(err['filename'])

        return sorted(error_images)

    def delete_errors(self):
        """오류가 있는 레이블과 이미지를 삭제"""
        error_lines = self.get_error_lines()
        error_images = self.get_error_images()

        if not error_lines and not error_images:
            print("\n삭제할 오류가 없습니다.")
            return

        # 삭제 대상 표시
        print("\n" + "=" * 80)
        print("삭제 대상")
        print("=" * 80)
        print(f"\n- 삭제할 레이블: {len(error_lines):,}개")
        print(f"- 삭제할 이미지: {len(error_images):,}개")

        # 샘플 표시
        if error_lines:
            print(f"\n삭제할 레이블 라인 (처음 10개): {error_lines[:10]}")
        if error_images:
            print(f"\n삭제할 이미지 파일 (처음 5개):")
            for img in error_images[:5]:
                print(f"  - {img}")
            if len(error_images) > 5:
                print(f"  ... 그 외 {len(error_images)-5}개")

        # 사용자 확인
        print("\n" + "=" * 80)
        confirm = input("정말로 오류 데이터를 삭제하시겠습니까? (y/n): ").strip().lower()

        if confirm != 'y':
            print("삭제를 취소했습니다.")
            return

        # 1. 레이블 파일 정리 (오류 라인 제외)
        print("\n레이블 파일 정리 중...")
        new_labels_file = self.labels_file.with_suffix('.cleaned.txt')
        cleaned_count = 0

        with open(self.labels_file, 'r', encoding='utf-8') as f:
            with open(new_labels_file, 'w', encoding='utf-8') as out:
                for line_num, line in enumerate(f, start=1):
                    if line_num not in error_lines:
                        out.write(line)
                        if line_num > 1:  # 헤더 제외
                            cleaned_count += 1

        # 백업 생성
        backup_file = self.labels_file.with_suffix('.backup.txt')
        import shutil
        shutil.copy2(self.labels_file, backup_file)
        print(f"  - 원본 백업 생성: {backup_file.name}")

        # 정리된 파일로 교체
        shutil.move(str(new_labels_file), str(self.labels_file))
        print(f"  - 정리된 레이블: {cleaned_count:,}개")
        print(f"  - 삭제된 레이블: {len(error_lines):,}개")

        # 2. 이미지 파일 삭제
        if error_images:
            print("\n이미지 파일 삭제 중...")
            deleted_images = 0
            failed_deletes = []

            for img_file in error_images:
                img_path = self.images_dir / img_file
                try:
                    if img_path.exists():
                        img_path.unlink()
                        deleted_images += 1
                except Exception as e:
                    failed_deletes.append((img_file, str(e)))

            print(f"  - 삭제된 이미지: {deleted_images:,}개")
            if failed_deletes:
                print(f"  - 삭제 실패: {len(failed_deletes)}개")
                for img, err in failed_deletes[:5]:
                    print(f"    {img}: {err}")

        print("\n✓ 오류 데이터 삭제가 완료되었습니다!")

def main():
    """메인 실행 함수"""
    # 경로 설정
    if len(sys.argv) > 1:
        labels_file = sys.argv[1]
        images_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(labels_file)
    else:
        # 기본 경로
        labels_file = "../data/learning/labels.txt"
        images_dir = "../data/learning"

    # 검증기 생성 및 실행
    checker = LabelChecker(labels_file, images_dir)

    # 레이블 검증
    checker.check_all_labels()

    # 보고서 출력
    checker.print_report()

    # result 폴더에 보고서 저장
    checker.save_error_report()

    # 오류가 있는 경우 삭제 옵션 제공
    total_errors = (len(checker.errors['missing_images']) +
                   len(checker.errors['out_of_bounds']) +
                   len(checker.errors['invalid_coords']) +
                   len(checker.errors['other_errors']))

    if total_errors > 0:
        print("\n" + "=" * 80)
        response = input("오류 데이터를 삭제하시겠습니까? (y/n): ").strip().lower()
        if response == 'y':
            checker.delete_errors()

if __name__ == "__main__":
    main()