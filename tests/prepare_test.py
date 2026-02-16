#!/usr/bin/env python3
"""
splitted 폴더에서 개별 발화(splitted_voices) 단위 WAV 30개를 선택하여 테스트 TSV를 생성.
각 카테고리 폴더에서 골고루 선택하며, JSON의 splitted_voices에서 sent_id로 
개별 문장 WAV를 매핑합니다.
"""
import os, json, csv, shutil, random

BASE = "/data3/yugwon/auto-trans-k-intonation"
SPLITTED = os.path.join(BASE, "data/splitted")
TARGET = 30

# 각 카테고리 폴더에서 개별 발화 수집
categories = sorted([d for d in os.listdir(SPLITTED) if os.path.isdir(os.path.join(SPLITTED, d))])
print(f"카테고리: {categories}")

per_cat = max(1, TARGET // len(categories)) + 1  # 여유분 포함
selected = []  # (sent_id, wav_path, text, sex)

for cat in categories:
    cat_dir = os.path.join(SPLITTED, cat)
    subdirs = sorted(os.listdir(cat_dir))
    count = 0
    
    for sd in subdirs:
        if count >= per_cat or len(selected) >= TARGET:
            break
        sd_path = os.path.join(cat_dir, sd)
        if not os.path.isdir(sd_path):
            continue
        
        files_in_dir = set(os.listdir(sd_path))
        json_files = sorted([f for f in files_in_dir if f.endswith('.json')])
        
        for jf in json_files:
            if count >= per_cat or len(selected) >= TARGET:
                break
            
            json_path = os.path.join(sd_path, jf)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            
            gender = data.get("reciter", {}).get("gender", "MALE")
            sex = "M" if "MALE" in gender.upper() else "F"
            
            sv_list = data.get("splitted_voices", [])
            if not sv_list:
                continue
            
            # 이 JSON에서 1~2개만 선택 (다양하게)
            picks = sv_list[:2] if len(sv_list) >= 2 else sv_list[:1]
            for sv in picks:
                if count >= per_cat or len(selected) >= TARGET:
                    break
                
                sent_id = sv.get("sent_id", "")
                if not sent_id:
                    continue
                
                wav_name = sent_id + ".wav"
                if wav_name not in files_in_dir:
                    continue
                
                text = sv.get("normalized", sv.get("origin", ""))
                if not text or not text.strip():
                    continue
                
                wav_path = os.path.join(sd_path, wav_name)
                selected.append((sent_id, wav_path, text.strip(), sex))
                count += 1

selected = selected[:TARGET]
print(f"\n선택된 개별 발화: {len(selected)}개")
for i, (sid, wp, txt, sex) in enumerate(selected):
    print(f"  [{i+1:2d}] {sid} | {sex} | {txt[:50]}...")

# TSV 생성 및 WAV 복사
tsv_path = os.path.join(BASE, "tests/test_30.tsv")
test_wav_dir = os.path.join(BASE, "tests/test_wav")
os.makedirs(test_wav_dir, exist_ok=True)

rows = []
for sent_id, wav_path, text, sex in selected:
    wav_name = os.path.basename(wav_path)
    dst = os.path.join(test_wav_dir, wav_name)
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(wav_path), dst)
    rows.append({"filename": wav_name, "sex": sex, "text": text})

with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "sex", "text"], delimiter='\t')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"\n=== TSV 생성 완료 ===")
print(f"경로: {tsv_path}")
print(f"항목 수: {len(rows)}")
print(f"\n처음 5개:")
for i, row in enumerate(rows[:5]):
    print(f"  [{i+1}] {row['filename']} | {row['sex']} | {row['text'][:60]}...")
print(f"\nWAV 디렉토리: {test_wav_dir}")
