import os
import json
import glob
import numpy as np
import pandas as pd
from textgrid import TextGrid, IntervalTier
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# JSON 및 TextGrid 파일의 루트 경로 설정
json_root = "/home/yugwon/nas/audio/ASR/free2022/NIKL_DIALOGUE_2022"
textgrid_root = "out/outputs2"

# 허용 오차 (tolerance) 설정 (ms 단위)
tolerance_levels = [10, 25, 50, 100]

# 정확도 계산용 리스트
accuracy_data = {
    "Category": [],
    "<10": [],
    "<25": [],
    "<50": [],
    "<100": []
}

# JSON의 누적 시간 보정
def adjust_json_times(utterances):
    adjusted_times = {}
    prev_end = 0  # 이전 발화 종료 시간

    for utt_id, (start, end, form) in utterances.items():
        if prev_end > 0:
            start = prev_end  # 이전 발화 종료 시간을 기준으로 시작점 조정
            end = start + (end - start)  # 상대적으로 유지
        adjusted_times[utt_id] = (start, end, form)
        prev_end = end  # 현재 발화 종료 시간을 다음 시작점으로 사용

    return adjusted_times

# TextGrid에서 첫 단어 시작(Xmin)과 마지막 어절 종료(Xmax) 찾고, 문장 생성
def parse_textgrid(file_path):
    try:
        grid = TextGrid.fromFile(file_path)
    except Exception as e:
        print(f"❌ TextGrid 파일 읽기 오류: {file_path}, 오류: {e}")
        return None, None, None
    
    word_intervals = None
    for tier in grid.tiers:
        if tier.name == "word" and isinstance(tier, IntervalTier):
            word_intervals = [interval for interval in tier.intervals if interval.mark.strip()]  # 공백 제거
            break

    if word_intervals:
        # 유효한 첫 단어 찾기
        first_word = next((interval for interval in word_intervals if interval.mark.strip()), None)
        last_word = next((interval for interval in reversed(word_intervals) if interval.mark.strip()), None)

        if first_word and last_word:
            tg_start = first_word.minTime  # 첫 유효 단어의 xmin
            tg_end = last_word.maxTime  # 마지막 유효 단어의 xmax
            tg_text = " ".join([interval.mark.strip() for interval in word_intervals])  # 단어 Join하여 문장 생성
            return tg_start, tg_end, tg_text

    return None, None, None  # 유효한 데이터 없을 경우 처리

# JSON 파일을 파싱하여 발화 시작/종료 시간 및 텍스트 추출
def parse_json(file_path):
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        print(f"❌ 빈 JSON 파일 (건너뜀): {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ JSON 파일 오류 (건너뜀): {file_path}")
        return None
    
    utterances = {}
    for doc in data.get("document", []):
        for utt in doc.get("utterance", []):
            utt_id = utt.get("id")
            if utt_id:
                try:
                    start_time = float(utt["start"])
                    end_time = float(utt["end"])
                    form = utt["form"]  # JSON의 문장 텍스트
                    utterances[utt_id] = (start_time, end_time, form)
                except (KeyError, ValueError):
                    print(f"⚠️ JSON 데이터 오류 (건너뜀): {file_path}, 발화 ID: {utt_id}")
    
    return adjust_json_times(utterances) if utterances else None  # JSON 시간 보정 후 반환

# 개별 JSON 파일과 관련된 TextGrid 파일을 처리하는 함수 (멀티프로세싱용)
def process_file(json_file):
    results = []
    
    json_data = parse_json(json_file)
    if not json_data:
        return results  # JSON이 유효하지 않으면 스킵
    
    base_name = os.path.basename(json_file).replace(".json", "")
    
    # TextGrid 파일이 있는 폴더 검색
    textgrid_files = glob.glob(os.path.join(textgrid_root, base_name, "*.TextGrid"))
    
    for textgrid_file in textgrid_files:
        file_id = os.path.basename(textgrid_file).replace("_M.TextGrid", "").replace("_F.TextGrid", "")
        
        # JSON에 해당 발화 ID가 존재하는 경우만 처리
        if file_id in json_data:
            json_start, json_end, json_text = json_data[file_id]
            tg_start, tg_end, tg_text = parse_textgrid(textgrid_file)

            # TextGrid에서 유효한 문장 시작/끝 정보가 있을 때만 비교
            if tg_start is not None and tg_end is not None:
                # ms 단위 변환
                start_diff_ms = abs((tg_start + json_start) - json_start) * 1000
                end_diff_ms = abs((tg_end + json_start) - json_end) * 1000

                results.append([base_name, file_id, start_diff_ms, end_diff_ms, json_text, tg_text])

    return results

# 전체 JSON 파일 리스트 가져오기
json_files = glob.glob(os.path.join(json_root, "*.json"))

# 멀티프로세싱 실행
if __name__ == "__main__":
    num_workers = max(1, cpu_count() - 2)
    chunk_size = max(1, len(json_files) // (num_workers * 2))

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, json_files, chunksize=chunk_size), total=len(json_files)))

    # Flatten the results
    results = [item for sublist in results for item in sublist]

    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=["파일", "발화 ID", "시작 시간 차이 (ms)", "종료 시간 차이 (ms)", "json text", 'tg text'])

    # 정확도 테이블 계산
    for tol in tolerance_levels:
        accuracy_data[f"<{tol}"].append((df["시작 시간 차이 (ms)"] < tol).mean())
        accuracy_data[f"<{tol}"].append((df["종료 시간 차이 (ms)"] < tol).mean())

    accuracy_data["Category"] = ["Start time alignment", "End time alignment"]

    # 정확도 테이블 생성
    accuracy_df = pd.DataFrame(accuracy_data)
    print("\n==== 정확도 테이블 ====\n")
    print(accuracy_df)

    # 결과 저장
    df.to_csv("aligner_time_difference_ms.csv", index=False)
    accuracy_df.to_csv("aligner_accuracy_table.csv", index=False)

    print("\n==== CSV 저장 완료 ====")
