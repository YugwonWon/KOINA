import os
import json
import pickle
import pandas as pd
from textgrid import TextGrid, IntervalTier
from multiprocessing import Pool
from tqdm import tqdm

# JSON 및 TextGrid 파일 목록 저장용 pkl 파일
json_pkl_path = "json_file_list.pkl"
tg_pkl_path = "tg_file_list.pkl"

# JSON 및 TextGrid 파일 경로 설정
json_root = "/home/yugwon/nas/audio/ASR/dialect1/01-1.정식개방데이터/Training/02.라벨링데이터"
textgrid_root = "/home/yugwon/nas/audio/ASR/dialect1/01-1.정식개방데이터/Training"

# 허용 오차 (tolerance) 설정 (ms 단위)
tolerance_levels = [10, 25, 50, 100]

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
        if "TL" in root or "TS" in root:  # 특정 폴더 제외
            continue
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

# TextGrid 파일에서 어절별 시작/종료 시간 추출
def parse_textgrid(file_path):
    try:
        grid = TextGrid.fromFile(file_path)
    except Exception as e:
        print(f"❌ TextGrid 파일 읽기 오류: {file_path}, 오류: {e}")
        return None

    word_intervals = []
    for tier in grid.tiers:
        if tier.name == "word" and isinstance(tier, IntervalTier):
            for interval in tier.intervals:
                if interval.mark.strip():  # 빈 공백 제외
                    word_intervals.append((interval.minTime, interval.maxTime, interval.mark.strip()))
            break

    return word_intervals if word_intervals else None

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
    basename = os.path.basename(textgrid_file).split("_.TextGrid")[0]  # "_" 이전 부분을 basename으로 사용

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

# 멀티프로세싱 실행 (num_workers=4로 고정)
if __name__ == "__main__":
    num_workers = 100  # 멀티프로세싱 워커 수 고정

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
