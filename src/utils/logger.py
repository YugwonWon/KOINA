
import logging, os
from logging.handlers import RotatingFileHandler
from utils.banner import SPERM_WHALE_BANNER

def setup_logger(level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger('KOINA')
    
    # 중복 초기화 방지
    if getattr(logger, "_configured", False):
        return logger                    # 이미 한 번 끝냈으면 재사용
    logger._configured = True

    # 상위 로거 깨끗이 비우기
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    # 내 기존 핸들러도 제거
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    # 새로운 핸들러 추가
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_dir = 'out/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger_path = os.path.join(log_dir, 'main.log')
    rotating_handler = RotatingFileHandler(
        logger_path, maxBytes=100 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )  # 100MB 최대, 백업 파일 5개 유지 (main.log.1 ~ main.log.5)
    rotating_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(rotating_handler)

    # 로거 레벨 설정
    logger.setLevel(level)

    # root로 전파 끊기
    logger.propagate = False
    
    # 자식 로거 설정
    child_loggers = ['transcriber', 'front', 'aligner', 'file_ops']
    for child_name in child_loggers:
        child_logger = logger.getChild(child_name)
        child_logger.propagate = True  # 부모 핸들러 상속 활성화
        # 자식 로거에 핸들러 추가 금지 (핸들러 중복 방지)

    return logger

def force_rollover(logger):
    lgr = logger
    while lgr:                       # 자식 → 부모 → … → root
        for h in lgr.handlers:
            if isinstance(h, RotatingFileHandler):
                h.doRollover()
        lgr = lgr.parent

# 로거 초기화
main_logger = setup_logger()
logger = main_logger.getChild("banner")
logger.info(SPERM_WHALE_BANNER.strip("\n"))