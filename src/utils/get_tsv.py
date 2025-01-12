import os
import json
import csv
import logging

# 로그 설정
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 경로 설정
json_folder = '/nas/audio/ASR/free2022/NIKL_DIALOGUE_2022'
root_folder = '/nas/audio/ASR/free2022/'

# TSV 파일을 저장할 경로
output_tsv = 'output.tsv'

# WAV 폴더에 속한 모든 WAV 파일을 담을 리스트
all_wav_files = []

# 첫 번째 레벨의 폴더 이름이 WAV로 끝나는 폴더를 대상으로 파일 탐색
for folder in os.listdir(root_folder):
    full_path = os.path.join(root_folder, folder)
    
    # 폴더가 WAV로 끝나는지 확인
    if os.path.isdir(full_path) and folder.endswith('WAV'):
        # 해당 폴더 하위의 모든 wav 파일 수집
        for root, dirs, files in os.walk(full_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    all_wav_files.append(wav_path)

# 수집된 WAV 파일 이름 기준으로 정렬
all_wav_files.sort()

# TSV 파일 작성
with open(output_tsv, 'w', newline='', encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(['wav_filepath', 'sex', 'text'])  # 필요한 칼럼만 작성

    # WAV 파일 목록을 순회하며 처리
    for wav_path in all_wav_files:
        wav_filename = os.path.basename(wav_path)
        json_filename = wav_filename.split('.')[0] + '.json'
        json_path = os.path.join(json_folder, json_filename)
        
        # JSON 파일이 존재하는 경우만 처리
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    document = data.get('document', [])
                    gender_dict = {}
                    for doc in document:
                        utterances = doc.get('utterance', [])
                        for gd in doc['metadata']['speaker']:
                            id = gd.get("id", "")
                            sex = gd.get("sex", "")
                            if sex == "":
                                print(f'{doc["id"]}: 성별이 없습니다.')
                            if sex == "남성":
                                sex = "M"
                            else:
                                sex = "F"
                            gender_dict[id] = sex
                        for utterance in utterances:
                            if utterance['id'] + '.wav' == wav_filename:
                                # 원하는 칼럼만 기록
                                utter = utterance['form'].replace('.', '').replace(',', '').replace('?', '').replace('!', '')
                                target_sex = gender_dict[f"{utterance['speaker_id']}"]
                                writer.writerow([wav_path, target_sex, utter])

            except json.JSONDecodeError as e:
                logging.error(f"JSONDecodeError in file: {json_path} - {e}")
            except Exception as e:
                logging.error(f"Error processing file: {json_path} - {e}")

print(f"{len(all_wav_files)}개의 WAV 파일이 처리되었습니다.")
