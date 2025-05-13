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
textgrid_root = "/home/yugwon/nas/audio/ASR/free2022"

# TextGrid에서 첫 어절 시작(Xmin)과 마지막 어절 종료(Xmax) 찾고, 문장 생성
def parse_textgrid(file_path):
    grid = TextGrid.fromFile(file_path)
    
    word_intervals = None
    for tier in grid.tiers:
        if tier.name == "word" and isinstance(tier, IntervalTier):
            word_intervals = [interval for interval in tier.intervals if interval.mark.strip()]
            break

    if word_intervals:
        tg_start = word_intervals[0].minTime  # 첫 어절 시작 시간 (xmin)
        tg_end = word_intervals[-1].maxTime  # 마지막 어절 종료 시간 (xmax)
        tg_text = " ".join([interval.mark.strip() for interval in word_intervals])  # 단어 Join하여 문장 생성
        return tg_start, tg_end, tg_text
    else:
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
    
    return utterances if utterances else None  # 유효한 데이터 없으면 None 반환

# 개별 JSON 파일과 관련된 TextGrid 파일을 처리하는 함수 (멀티프로세싱용)
def process_file(json_file):
    results = []
    
    json_data = parse_json(json_file)
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
                # **JSON 기준으로 TextGrid의 상대 시간을 실제 WAV 시간으로 변환**
                tg_start_adj = tg_start + json_start
                tg_end_adj = tg_end + json_start

                # 비교: 변환된 TextGrid 시간 vs. JSON 시간
                start_diff = abs(tg_start_adj - json_start)
                end_diff = abs(tg_end_adj - json_end)
                
                results.append([base_name, file_id, tg_start_adj, tg_end_adj, json_start, json_end, start_diff, end_diff, tg_text, json_text])
    print(f'{json_file}: Done!')
    return results

# 전체 JSON 파일 리스트 가져오기
json_files = glob.glob(os.path.join(json_root, "*.json"))

# 멀티프로세싱 실행
if __name__ == "__main__":
    num_workers = max(1, cpu_count() - 2)  # CPU 코어 개수에서 2개 빼고 사용
    chunk_size = len(json_files) // num_workers  # 파일을 균등하게 분배

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, json_files, chunksize=chunk_size), total=len(json_files)))

    # Flatten the results
    results = [item for sublist in results for item in sublist]

    # 결과 저장
    df = pd.DataFrame(results, columns=["파일", "발화 ID", "TG 시작(보정)", "TG 종료(보정)", "JSON 시작", "JSON 종료", "시작 시간 차이", "종료 시간 차이", "TG 문장", "JSON 문장"])
    df.to_csv("aligner_sentence_time_difference_2.csv", index=False)

    # 통계 분석
    stats = {
        "총 매칭 문장 수": len(df),
        "문장 시작 시간 차이 평균": df["시작 시간 차이"].mean(),
        "문장 시작 시간 차이 표준편차": df["시작 시간 차이"].std(),
        "문장 종료 시간 차이 평균": df["종료 시간 차이"].mean(),
        "문장 종료 시간 차이 표준편차": df["종료 시간 차이"].std(),
    }

    # 통계 결과 출력
    print("==== 분석 결과 ====")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
