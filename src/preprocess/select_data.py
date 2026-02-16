import os
import shutil

# 소스 폴더 경로
source_dir = os.path.expanduser('~/nas/yugwon/SDRW')
# 복사 대상 루트 폴더
dest_base = 'out/human-annote'

# 소스 폴더 내 모든 서브폴더 순회 (폴더명 예: SDRW2200000450)
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if os.path.isdir(folder_path) and len(folder) >= 3:
        folder_suffix = folder[-3:]  # 뒤 3글자 추출
        if folder_suffix.isdigit():
            num = int(folder_suffix)
            if 450 <= num <= 885:
                # 목적지에 폴더명 유지하여 생성
                dest_folder = os.path.join(dest_base, folder)
                os.makedirs(dest_folder, exist_ok=True)
                # 폴더 내 파일 순회
                for file_name in os.listdir(folder_path):
                    if "_momel_pitch_contour_minimalized" in file_name or "TextGrid" in file_name:
                        src_file = os.path.join(folder_path, file_name)
                        dst_file = os.path.join(dest_folder, file_name)
                        shutil.copy(src_file, dst_file)
                        print(f"Copied: {src_file} -> {dst_file}")