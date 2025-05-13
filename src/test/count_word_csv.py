#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSV(탭 구분) 파일에서 text 열의 어절(공백 단위) 개수를 센다
"""

import csv
from pathlib import Path

tsv_path = Path("data/dialect_output.tsv")

total_rows   = 0
total_tokens = 0

with tsv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    if not {"wav_filepath", "text"} <= set(reader.fieldnames or []):
        raise ValueError("TSV 헤더에 'wav_filepath', 'text' 열이 없습니다.")

    print(f"{'wav_filepath':50s}  tokens")
    print("-" * 60)

    for row in reader:
        try:
            total_rows += 1
            text = row["text"].strip()
            n_tok = len(text.split())          # 스페이스 단위
            total_tokens += n_tok

            # # 행별 출력 (원하면 주석 처리)
            # print(f"{row['wav_filepath'][:50]:50s}  {n_tok:6d}")
        except Exception as e:
            continue

print("\n======== 요약 ========")
print(f"총 행 수           : {total_rows:,}")
print(f"총 어절 수         : {total_tokens:,}")
if total_rows:
    print(f"행당 평균 어절 수  : {total_tokens / total_rows:,.2f}")
