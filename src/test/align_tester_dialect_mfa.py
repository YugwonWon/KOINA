import os
import json
import pickle
import pandas as pd
import csv
import re, unicodedata as ud
from pathlib import Path
from textgrid import TextGrid, IntervalTier
from multiprocessing import Pool
from tqdm import tqdm

# JSON 및 TextGrid 파일 목록 저장용 pkl 파일
json_pkl_path = "json_file_list.pkl"
tg_pkl_path = "tg_file_list.pkl"
tsv_original = Path("data/dialect_output.tsv")
# JSON 및 TextGrid 파일 경로 설정
json_root = "/home/yugwon/nas/audio/ASR/dialect1/01-1.정식개방데이터/Training/02.라벨링데이터"
textgrid_root = "/home/yugwon/mfa/tg_dir"

# 허용 오차 (tolerance) 설정 (ms 단위)
tolerance_levels = [10, 25, 50, 100]

_HANGUL_RE = re.compile(r'[가-힣]+')
def _clean(txt: str) -> str:
    """NFC 정규화 뒤 완성형 한글 음절만 남긴다"""
    return ''.join(_HANGUL_RE.findall(ud.normalize('NFC', txt)))

# === JSON 파일 목록 검색 및 저장 ===
def get_json_files():
    json_files = {}

    if os.path.exists(json_pkl_path):
        print(f"📂 기존 JSON 파일 목록 로드 중: {json_pkl_path}")
        with open(json_pkl_path, "rb") as f:
            json_list = pickle.load(f)  # 기존 pkl 파일은 리스트로 저장됨

        # 리스트를 {basename: path} 형태의 딕셔너리로 변환
        for json_path in json_list:
            basename = os.path.splitext(os.path.basename(json_path))[0]  # 확장자 제거하여 basename 추출
            json_files[basename] = json_path

        return json_files  # 변환된 딕셔너리 반환

    # 기존 pkl 파일이 없으면 새로 검색
    json_list = []
    for root, _, files in os.walk(json_root):
        if "충청도" in root:  # '충청도'가 포함된 경로만 필터링
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    json_list.append(json_path)
                    basename = os.path.splitext(file)[0]
                    json_files[basename] = json_path

    # JSON 파일 목록을 pickle로 저장
    with open(json_pkl_path, "wb") as f:
        pickle.dump(json_list, f)  # 리스트 형태로 저장

    print(f"✅ JSON 파일 목록 저장 완료: {json_pkl_path} ({len(json_files)}개 파일)")
    return json_files


# === TextGrid 파일 목록 검색 및 저장 ===
def get_textgrid_files():
    if os.path.exists(tg_pkl_path):
        print(f"📂 기존 TextGrid 파일 목록 로드 중: {tg_pkl_path}")
        with open(tg_pkl_path, "rb") as f:
            return pickle.load(f)

    textgrid_files = []
    for root, _, files in os.walk(textgrid_root):
        for file in files:
            if file.endswith(".TextGrid"):
                textgrid_files.append(os.path.join(root, file))

    # TextGrid 파일 목록을 pickle로 저장
    with open(tg_pkl_path, "wb") as f:
        pickle.dump(textgrid_files, f)

    print(f"✅ TextGrid 파일 목록 저장 완료: {tg_pkl_path} ({len(textgrid_files)}개 파일)")
    return textgrid_files

# JSON 파일에서 어절별 시작/종료 시간 추출
def parse_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ JSON 파일 오류 (건너뜀): {file_path}")
        return None

    word_segments = []
    for seg in data.get("transcription", {}).get("segments", []):
        try:
            start_time = float(seg["startTime"].split(":")[-1])  # 초 단위 변환
            end_time = float(seg["endTime"].split(":")[-1])
            word = seg["dialect"]  # 어절 텍스트
            word_segments.append((start_time, end_time, word))
        except (KeyError, ValueError):
            print(f"⚠️ JSON 데이터 오류 (건너뜀): {file_path}")

    return word_segments if word_segments else None

def load_original_words(tsv_path: Path) -> dict[str, list[str]]:
    """
    TSV(탭 구분) → {wav_id: [word1, word2, ...]} 딕셔너리
    필요한 헤더: 'wav_filepath', 'text'
    """
    mapping: dict[str, list[str]] = {}

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not {"wav_filepath", "text"} <= set(reader.fieldnames or []):
            raise ValueError("TSV 헤더에 'wav_filepath' 또는 'text' 열이 없습니다.")

        for idx, row in enumerate(reader, start=2):   # 2 = 첫 데이터 행
            wav_id = (row.get("wav_filepath") or "").strip()
            sentence = (row.get("text") or "").strip()
            if not wav_id or not sentence:
                print(f"[WARN] {idx}행: 빈 필드 → 건너뜀")
                continue
            mapping[wav_id] = sentence.split()        # 공백 기준 분할

    print(f"✅ 원문 어절 매핑 로드: {len(mapping):,}개")
    return mapping

# TextGrid 파일에서 어절별 시작/종료 시간 추출
def parse_textgrid(file_path: str):
    """
    TextGrid 'word(s)' tier의 interval 조각을 이어붙여
    TSV 원문 어절과 매칭 → (start, end, word) 리스트 반환
    * 한글 음절(가-힣)만 비교, 문장부호·영어·자모 분리 제거
    """
    basename = Path(file_path).stem
    target_words = [_clean(w) for w in orig_words.get(basename, []) if _clean(w)]
    if not target_words:
        return None

    try:
        tg = TextGrid.fromFile(file_path)
    except Exception as e:
        print(f"❌ TG 읽기 오류: {file_path} – {e}")
        return None

    # words tier 찾기
    word_tier = next(
        (t for t in tg.tiers if isinstance(t, IntervalTier)
         and t.name.lower().startswith("word")),
        None
    )
    if word_tier is None:
        return None

    results, cur_word, start_t, tidx = [], "", None, 0

    for iv in word_tier.intervals:
        piece = _clean(iv.mark)
        if not piece:           # 빈 라벨 → 건너뜀
            continue

        if cur_word == "":
            start_t = iv.minTime
        cur_word += piece

        if tidx >= len(target_words):   # 원문 어절 초과 → 중단
            break
        target = target_words[tidx]

        if cur_word == target:          # 정확히 매칭
            results.append((start_t, iv.maxTime, target))
            cur_word, start_t = "", None
            tidx += 1
        elif len(cur_word) > len(target) or not target.startswith(cur_word):
            # 불일치: 현재 interval부터 다시 시작
            print(f"[WARN] 불일치: {basename} | '{cur_word}' vs '{target}'")
            cur_word, start_t = "", None  # 버퍼·시각 리셋

    return results if results else None

# JSON과 TextGrid 비교 후 시간 차이 계산
def compare_times(json_words, tg_words):
    results = []
    min_len = min(len(json_words), len(tg_words))

    for i in range(min_len):
        json_start, json_end, json_word = json_words[i]
        tg_start, tg_end, tg_word = tg_words[i]

        start_diff_ms = abs((tg_start - json_start) * 1000)  # ms 단위 변환
        end_diff_ms = abs((tg_end - json_end) * 1000)

        results.append([json_word, tg_word, start_diff_ms, end_diff_ms])

    return results

# 개별 TextGrid 파일을 처리하는 함수 (멀티프로세싱용)
def process_file(textgrid_file):
    results = []
    basename = os.path.basename(textgrid_file).split(".TextGrid")[0]  # "_" 이전 부분을 basename으로 사용

    json_path = json_files.get(basename)  # JSON 딕셔너리에서 빠르게 검색
    if json_path:
        json_words = parse_json(json_path)
        if json_words:
            tg_words = parse_textgrid(textgrid_file)
            if tg_words:
                matched_results = compare_times(json_words, tg_words)
                for row in matched_results:
                    results.append([textgrid_file] + row)

    return results

# JSON 및 TextGrid 파일 목록 가져오기
json_files = get_json_files()  # JSON을 {파일명: 경로} 딕셔너리로 저장
textgrid_files = get_textgrid_files()
orig_words = load_original_words(tsv_original)

# 멀티프로세싱 실행 (num_workers=4로 고정)
if __name__ == "__main__":
    # SAMPLE_SIZE = 0
    # if SAMPLE_SIZE and SAMPLE_SIZE > 0:
    #     textgrid_files = textgrid_files[:SAMPLE_SIZE]
    #     print(f"🧪 샘플 {len(textgrid_files)}개만 테스트 실행")

    # results = []
    # for tg_path in tqdm(textgrid_files, desc="Processing"):
    #     results.extend(process_file(tg_path))

    # # ---------------- 결과 후처리 ----------------
    # df = pd.DataFrame(
    #     results,
    #     columns=["파일", "JSON 단어", "TG 단어", "시작 시간 차이 (ms)", "종료 시간 차이 (ms)"]
    # )
    # df.to_csv("aligner_time_difference_ms.csv", index=False)
    # print("\n==== CSV 저장 완료 ====")

    # # 정확도/통계
    # accuracy_data = {"Category": ["Start time alignment", "End time alignment"]}
    # for tol in tolerance_levels:
    #     accuracy_data[f"<{tol}ms"] = [
    #         (df["시작 시간 차이 (ms)"] < tol).mean(),
    #         (df["종료 시간 차이 (ms)"] < tol).mean()
    #     ]
    # accuracy_df = pd.DataFrame(accuracy_data)

    # stats = {
    #     "총 샘플 개수": len(df),
    #     "시작 평균 (ms)": df["시작 시간 차이 (ms)"].mean(),
    #     "시작 표준편차 (ms)": df["시작 시간 차이 (ms)"].std(),
    #     "종료 평균 (ms)": df["종료 시간 차이 (ms)"].mean(),
    #     "종료 표준편차 (ms)": df["종료 시간 차이 (ms)"].std(),
    # }

    # print("\n==== 통계 분석 결과 ====")
    # for k, v in sㄴtats.items():
    #     print(f"{k}: {v:.4f}")

    # print("\n==== 정확도 테이블 ====")
    # print(accuracy_df.to_string(index=False))

    # accuracy_df.to_csv("aligner_accuracy_table.csv", index=False)
    # print("\n==== CSV 저장 완료 ====")
    num_workers = 100  # 멀티프로세싱 워커 수 고정
    tsv_original = Path("data/dialect_output.tsv")   # 원본 TSV 경로
    orig_words = load_original_words(tsv_original)
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, textgrid_files), total=len(textgrid_files)))

    # Flatten the results
    results = [item for sublist in results for item in sublist]

    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=["파일", "JSON 단어", "TG 단어", "시작 시간 차이 (ms)", "종료 시간 차이 (ms)"])

    # 결과 저장
    df.to_csv("aligner_time_difference_ms.csv", index=False)

    print("\n==== CSV 저장 완료 ====")
    
    # 정확도 테이블 계산
    accuracy_data = {
        "Category": ["Start time alignment", "End time alignment"]
    }

    for tol in tolerance_levels:
        accuracy_data[f"<{tol}ms"] = [
            (df["시작 시간 차이 (ms)"] < tol).mean(),
            (df["종료 시간 차이 (ms)"] < tol).mean()
        ]

    # 정확도 테이블 생성
    accuracy_df = pd.DataFrame(accuracy_data)

    # 시작/종료 시간 차이의 평균 및 표준편차 계산
    stats = {
        "총 샘플 개수": len(df),
        "시작 시간 차이 평균 (ms)": df["시작 시간 차이 (ms)"].mean(),
        "시작 시간 차이 표준편차 (ms)": df["시작 시간 차이 (ms)"].std(),
        "종료 시간 차이 평균 (ms)": df["종료 시간 차이 (ms)"].mean(),
        "종료 시간 차이 표준편차 (ms)": df["종료 시간 차이 (ms)"].std(),
    }

    # 통계 데이터 출력
    print("\n==== 통계 분석 결과 ====")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    # 정확도 테이블 출력
    print("\n==== 정확도 테이블 ====")
    print(accuracy_df.to_string(index=False))

    # 결과 저장
    df.to_csv("aligner_time_difference_ms.csv", index=False)
    accuracy_df.to_csv("aligner_accuracy_table.csv", index=False)

    print("\n==== CSV 저장 완료 ====")