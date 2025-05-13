#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dialect_output.tsv를 이용해 MFA 강제정렬 수행
1) TSV -> corpus 디렉터리(.wav + .lab) 생성(심볼릭 링크 권장)
2) mfa align 실행
3) 결과 TextGrid를 tg_dir(폴더별)로 복사
"""
import os, subprocess, shutil, pickle, sys, time
from pathlib import Path
import pandas as pd
import re

# ==== 사용자 설정 ====
tsv_path   = Path("test-data/dialect_output.tsv")      # dialect_output.tsv 위치
source_dir = Path("/home/yugwon/nas/audio/ASR/dialect1/01-1.정식개방데이터/Training/01.원천데이터")
corpus_dir = Path("dialect_corpus")                   # 임시 코퍼스 디렉터리
cache_path = Path("wav_index.pkl")          # 인덱스 캐시
dict_path  = Path("korean_mfa")                  # MFA용 발음 사전 (.dict)
acoustic   = "korean_mfa"                             # 설치된 음향모델 이름/경로
out_dir    = Path("align_out")                        # MFA align 결과
tg_dir     = Path("tg_dir")                           # 최종 TextGrid 저장 위치
njobs      = 8                                        # 병렬 스레드 수
# =====================
FILE_RE = re.compile(r'^say_set[\w_]+$')   # WAV 파일명 패턴
FILENAME_RE = re.compile(r'^[\w\-]+$')   # say_set... 형태만 허용
TARGET_KEYWORD = "충청도"

# ---------- ① WAV 인덱스 로드 / 생성 ----------
def load_wav_index():
    if cache_path.exists():
        with cache_path.open("rb") as f:
            wav_table = pickle.load(f)
        print(f"📂  캐시 로드 완료 ({len(wav_table):,}개)")
        return wav_table

    print("WAV 인덱스 생성 중…")
    wav_table = {}
    for wav in source_dir.rglob("*.wav"):
        # ★ 경로 필터: ‘충청도’ 포함한 폴더만 수집
        if TARGET_KEYWORD not in str(wav.parent):
            continue
        wav_table.setdefault(wav.stem, wav)    # 중복 시 최초 경로 유지
    print(f"총 {len(wav_table):,}개 WAV 인덱싱 완료 → 캐시 저장")
    with cache_path.open("wb") as f:
        pickle.dump(wav_table, f)
    return wav_table

def read_tsv(path: Path):
    """헤더 기반으로 'wav_filepath', 'text' 컬럼을 읽어 튜플 반환"""
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            wav = row.get("wav_filepath")
            text = row.get("text")
            if not wav or not text:
                print(f"[WARN] {i+2}행: 필드 없음 또는 빈 값 → 건너뜀")
                continue
            wav = wav.strip()
            text = text.strip()
            if wav and text:
                pairs.append((wav, text))
    return pairs

# ---------- ② TSV → corpus 디렉터리 ----------
import csv, re
MAX_LINE = 2000        # 줄 전체 길이 제한 (byte)
MAX_TOKEN = 100        # 한 토큰 최대 길이 (byte)
TEST_MAX   = 0

def safe_write_lab(path: Path, text: str):
    clean = (
        text.replace("\u3000", " ")        # 전각 스페이스 → 일반 스페이스
            .replace("\t", " ")            # 탭 → 스페이스
            .replace("\r", " ")
    )

    parts, line, line_bytes = [], [], 0
    for tok in clean.split():
        # ①  초긴 토큰이면  MAX_TOKEN 단위로 쪼갬
        tok_chunks = [tok[i:i+MAX_TOKEN] for i in range(0, len(tok), MAX_TOKEN)]
        for chunk in tok_chunks:
            b = len(chunk.encode("utf-8")) + 1
            if line_bytes + b > MAX_LINE:
                parts.append(" ".join(line))
                line, line_bytes = [], 0
            line.append(chunk)
            line_bytes += b
    if line:
        parts.append(" ".join(line))

    try:
        path.write_text("\n".join(parts), encoding="utf-8")
    except Exception as e:                 # 어떤 이유로든 쓰기 실패 시
        print(f"[ERR] {path.name}: {e}")

def is_text_safe(text: str) -> bool:
    """MeCab이 처리하기 안전한지 빠르게 판단"""
    b = len(text.encode('utf-8'))
    if b > MAX_LINE:
        return False
    for tok in text.split():
        if len(tok.encode('utf-8')) > MAX_TOKEN:
            return False
    return True

def build_corpus(wav_table, pairs):
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)

    linked, skipped = 0, 0
    for wav_id, text in pairs:
        if TEST_MAX and linked >= TEST_MAX:         # N개 채우면 중단
            break

        wav_src = wav_table.get(wav_id)
        if not wav_src or not text or not is_text_safe(text):
            skipped += 1
            continue

        wav_dst = corpus_dir / wav_src.relative_to(source_dir)
        wav_dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav_dst.symlink_to(wav_src)
        except FileExistsError:
            pass

        wav_dst.with_suffix(".lab").write_text(text, encoding="utf-8")
        linked += 1

    print(f"corpus(샘플) 작성: {linked}개 링크, {skipped}개 건너뜀")


# ---------- ③ MFA 실행 ----------
def run_mfa():
    subprocess.run([
        "mfa", "align", corpus_dir, dict_path, acoustic, out_dir,
        "-j", str(njobs), "--clean", "--verbose"
    ], check=True)


# ---------- ④ TextGrid 수집 ----------
def collect_textgrids():
    if tg_dir.exists():
        shutil.rmtree(tg_dir)
    copied = 0
    for tg in out_dir.rglob("*.TextGrid"):
        rel = tg.relative_to(out_dir)
        dst = tg_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tg, dst)
        copied += 1
    print(f"TextGrid {copied}개를 {tg_dir}에 복사 완료")


# ------------------ main ------------------
if __name__ == "__main__":
    wav_table = load_wav_index()
    pairs = read_tsv(tsv_path)
    build_corpus(wav_table, pairs)
    run_mfa()
    collect_textgrids()
    print("전체 파이프라인 완료!")