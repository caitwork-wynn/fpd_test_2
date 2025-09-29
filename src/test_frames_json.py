#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
frames.json 기반 처리 테스트
"""

import subprocess
import sys
import time

# 자동으로 1번 (20250812_lc)을 선택하고 r로 초기화
input_text = "1\nr\ny\n"

print("110.merge_learning_lc_data.py 실행 중...")
print("입력: 1 (20250812_lc), r (초기화), y (이미지 삭제)")

process = subprocess.Popen(
    [sys.executable, "110.merge_learning_lc_data.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace'
)

# 타임아웃 설정
try:
    stdout, _ = process.communicate(input=input_text, timeout=120)
    print("\n출력:")
    print(stdout)
except subprocess.TimeoutExpired:
    process.kill()
    print("프로세스가 타임아웃되었습니다.")
    stdout, _ = process.communicate()
    print("\n부분 출력:")
    print(stdout)