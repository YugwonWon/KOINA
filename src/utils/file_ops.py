import os
import logging
from utils.logger import main_logger
logger = logging.getLogger('KOINA.file_ops')

def collect_wav_files(base_dir: str):
    """
    지정된 디렉토리에서 모든 WAV 파일을 검색하고, 파일명-경로 매핑 딕셔너리를 생성합니다.
    """
    wav_dict = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_name = os.path.basename(file)
                wav_dict[file_name] = os.path.join(root, file)
    logger.info(f"WAV 파일 {len(wav_dict)}개를 검색했습니다.")
    return wav_dict

def detect_delimiter(file_path: str):
    """
    파일 확장자에 따라 구분자를 반환합니다.
    """
    if file_path.endswith(".tsv"):
        return '\t'
    elif file_path.endswith(".csv"):
        return ','
    else:
        raise ValueError(f"{file_path}: 지원하지 않는 파일 형식(확장자)입니다. CSV(TSV) 파일만 가능합니다.")
