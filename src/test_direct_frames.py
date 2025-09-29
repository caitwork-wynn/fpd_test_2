#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
frames.json 직접 처리 테스트
"""

import json
from pathlib import Path
import cv2
import numpy as np

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent

def test_frames_json_processing():
    """frames.json 처리 테스트"""
    folder_path = ROOT_DIR / "data" / "base" / "20250812_lc"
    frames_json_path = folder_path / "frames.json"

    print(f"frames.json 경로: {frames_json_path}")

    if not frames_json_path.exists():
        print("frames.json 파일이 없습니다.")
        return

    # frames.json 읽기
    with open(frames_json_path, 'r', encoding='utf-8') as f:
        frames_data = json.load(f)

    frames = frames_data.get('frames', [])
    print(f"총 프레임 수: {len(frames)}")

    # 첫 번째 프레임만 테스트
    if frames:
        frame = frames[0]
        timestamp = frame.get('timestamp', '')
        images = frame.get('images', {})

        print(f"\n첫 번째 프레임:")
        print(f"  - timestamp: {timestamp}")
        print(f"  - 이미지 수: {len(images)}")

        # 각 CCTV 이미지 확인
        for cctv_no, image_rel_path in images.items():
            image_path = folder_path / image_rel_path
            user_json_path = folder_path / f"{image_rel_path}.user.json"

            print(f"\n  CCTV {cctv_no}:")
            print(f"    - 이미지: {image_path.name}")
            print(f"    - 이미지 존재: {image_path.exists()}")
            print(f"    - user.json 존재: {user_json_path.exists()}")

            if user_json_path.exists():
                # user.json 읽기
                with open(user_json_path, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)

                objects = user_data.get('objects', [])
                print(f"    - 객체 수: {len(objects)}")

                # 첫 번째 객체만 테스트
                if objects:
                    obj = objects[0]
                    print(f"      첫 번째 객체:")
                    print(f"        - objectId: {obj.get('objectId')}")
                    print(f"        - objectType: {obj.get('objectType')}")
                    print(f"        - confidence: {obj.get('confidence'):.2f}")

                    bbox = obj.get('bbox', {})
                    detected = obj.get('detected', {})

                    if bbox and detected:
                        print(f"        - bbox: x={bbox.get('x', 0):.1f}, y={bbox.get('y', 0):.1f}")
                        print(f"        - detected: x={detected.get('x', 0):.1f}, y={detected.get('y', 0):.1f}")

if __name__ == "__main__":
    print("=== frames.json 처리 테스트 ===\n")
    test_frames_json_processing()
    print("\n테스트 완료!")