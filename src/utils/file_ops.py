import os
import subprocess

# 로거 설정
from utils.logger import main_logger
logger = main_logger.getChild('file_ops')

# 지원하는 오디오 확장자 (WAV 포함)
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def collect_wav_files(base_dir: str):
    """
    지정된 디렉토리에서 모든 오디오 파일을 검색하고, 파일명-경로 매핑 딕셔너리를 생성합니다.
    WAV가 아닌 오디오 파일(mp3, flac, ogg 등)도 수집하며,
    이후 ensure_wav()를 통해 WAV로 변환하여 처리할 수 있습니다.
    """
    wav_dict = {}
    non_wav_count = 0
    for root, _, files in os.walk(base_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_AUDIO_EXTENSIONS:
                file_name = os.path.basename(file)
                wav_dict[file_name] = os.path.join(root, file)
                if ext != ".wav":
                    non_wav_count += 1
    if non_wav_count > 0:
        logger.info(f"[FILE] 사용자가 지정한 폴더에서 오디오 파일 총 {len(wav_dict)}개를 검색했습니다. "
                     f"(WAV: {len(wav_dict) - non_wav_count}개, 기타: {non_wav_count}개 → WAV 변환 예정)")
    else:
        logger.info(f"[FILE] 사용자가 지정한 폴더에서 WAV 파일 총 {len(wav_dict)}개를 검색했습니다.")
    return wav_dict


def ensure_wav(audio_path: str) -> str:
    """
    오디오 파일이 WAV가 아닌 경우 ffmpeg를 사용하여 WAV(PCM 16kHz mono)로 변환합니다.
    이미 WAV인 경우 원본 경로를 그대로 반환합니다.
    
    변환된 WAV 파일은 원본과 같은 디렉토리에 같은 이름(.wav)으로 생성됩니다.
    
    Args:
        audio_path: 오디오 파일 경로
    
    Returns:
        WAV 파일 경로
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path

    base_name = os.path.splitext(audio_path)[0]
    wav_path = f"{base_name}.wav"

    # 이미 변환된 WAV가 존재하면 재변환하지 않음
    if os.path.exists(wav_path):
        logger.info(f"[FILE] 이미 변환된 WAV 파일이 존재합니다: {os.path.basename(wav_path)}")
        return wav_path

    logger.info(f"[FILE] 오디오 변환 중: {os.path.basename(audio_path)} → WAV (PCM 16kHz mono)")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info(f"[FILE] 변환 완료: {os.path.basename(wav_path)}")
        return wav_path
    except FileNotFoundError:
        logger.error("[FILE] ffmpeg가 설치되어 있지 않습니다. WAV 이외의 형식을 변환할 수 없습니다.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"[FILE] 오디오 변환 실패: {os.path.basename(audio_path)} (ffmpeg 에러)")
        raise

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
