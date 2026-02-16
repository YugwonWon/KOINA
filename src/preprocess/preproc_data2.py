
import os
import pandas as pd
import shutil

# CSV 파일 경로
csv_path = 'training_data_filtered.csv'

# 원본 파일들이 있는 폴더와 복사할 대상 폴더
source_base = os.path.join('out', 'human-annote')
target_base = os.path.join('out', 'human-annote-target')

# CSV 파일 읽기
df = pd.read_csv(csv_path)

for file_name in df['filename']:
    # 파일명 예: "SDRW2200000450.1.1.122_F"
    # 폴더명은 파일명에서 첫 번째 '.' 이전 값, 즉 "SDRW2200000450"
    folder_name = file_name.split('.')[0]
    
    # 대상 폴더 경로 생성
    dest_folder = os.path.join(target_base, folder_name)
    os.makedirs(dest_folder, exist_ok=True)
    
    # 텍스트그리드 파일 경로 및 복사
    src_textgrid = os.path.join(source_base, folder_name, file_name + ".TextGrid")
    dest_textgrid = os.path.join(dest_folder, file_name + ".TextGrid")
    try:
        shutil.copy(src_textgrid, dest_textgrid)
        print(f"Copied {src_textgrid} to {dest_textgrid}")
    except Exception as e:
        print(f"Error copying {src_textgrid}: {e}")
    
    # jpg 파일 경로 및 복사 (예: _momel_pitch_contour_minimalized.jpg)
    src_jpg = os.path.join(source_base, folder_name, file_name + "_momel_pitch_contour_minimalized.jpg")
    dest_jpg = os.path.join(dest_folder, file_name + "_momel_pitch_contour_minimalized.jpg")
    try:
        shutil.copy(src_jpg, dest_jpg)
        print(f"Copied {src_jpg} to {dest_jpg}")
    except Exception as e:
        print(f"Error copying {src_jpg}: {e}")