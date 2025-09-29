#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20250812_lc 폴더 자동 처리 테스트
"""

import subprocess
import sys

# 자동으로 1번 (20250812_lc)을 선택하고 r로 초기화
input_text = "1\nr\ny\n"

process = subprocess.Popen(
    [sys.executable, "110.merge_learning_lc_data.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='utf-8'
)

stdout, stderr = process.communicate(input=input_text)

print("출력:")
print(stdout)

if stderr:
    print("에러:")
    print(stderr)