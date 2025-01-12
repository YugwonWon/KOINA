import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 부모 로거 설정
    logger = logging.getLogger('IntonationTranscriber')

    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # 새로운 핸들러 추가
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_dir = 'out/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger_path = os.path.join(log_dir, 'main.log')
    rotating_handler = RotatingFileHandler(
        logger_path, maxBytes=500 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    rotating_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(rotating_handler)

    # 로거 레벨 설정
    logger.setLevel(level)

    # 자식 로거 설정
    child_loggers = ['transcriber', 'front']
    for child_name in child_loggers:
        child_logger = logger.getChild(child_name)
        child_logger.propagate = True  # 부모 핸들러 상속 활성화
        # 자식 로거에 핸들러 추가 금지 (핸들러 중복 방지)

    return logger


# 로거 초기화
main_logger = setup_logger()
