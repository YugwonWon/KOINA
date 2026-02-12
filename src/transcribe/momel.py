"""
Momel(MOdelling of MELody) 기반 음높이 포인트 생성 모듈

- Momel 바이너리 실행
- 음성 구간별 f0 파일 생성 → Momel 실행 → Points 티어 구성
- Points 티어에서 (시간, f0) 추출
"""

import os
import subprocess

from parselmouth.praat import call
from textgrid import PointTier

from transcribe.pitch import extract_pitch_data, remove_doubling_halving
from utils.logger import main_logger

logger = main_logger.getChild('momel')

__all__ = [
    "generate_momel_labels",
    "get_pitch_points",
    "run_momel_subprocess",
]


# ─────────────────────────────────────────────────────────────────────────────
# Momel 바이너리 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_momel_subprocess(momel_cmd, f0_file, momel_file, momel_parameters):
    """
    Momel 바이너리를 실행합니다.

    Parameters
    ----------
    momel_cmd : str   – Momel 실행 파일 경로
    f0_file : str     – 입력 .f0 파일 이름 (out/models/ 하위)
    momel_file : str  – 출력 .model 파일 이름 (out/models/ 하위)
    momel_parameters : str
    """
    try:
        momel_cmd = os.path.abspath(momel_cmd)
        f0_path = os.path.abspath(f'out/models/{f0_file}')
        momel_out = os.path.abspath(f'out/models/{momel_file}')

        env = os.environ.copy()
        env['PATH'] = f'{os.path.dirname(momel_cmd)}:{env["PATH"]}'

        command = f'{momel_cmd} {momel_parameters} <"{f0_path}" >"{momel_out}"'
        subprocess.run(command, shell=True, check=True, env=env)
        logger.info(f'[RUN] Momel 실행 완료: {momel_file}')
    except subprocess.CalledProcessError as e:
        logger.error(f'[RUN] Momel 실행 중 오류 발생: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Points 티어 유틸
# ─────────────────────────────────────────────────────────────────────────────

def get_pitch_points(points_tier):
    """
    Points 티어에서 시간과 f0 값을 추출합니다.

    Returns
    -------
    times : list[float]
    f0_values : list[float]
    """
    times = [p.time for p in points_tier.points]
    f0_values = [float(p.mark) for p in points_tier.points]
    return times, f0_values


# ─────────────────────────────────────────────────────────────────────────────
# Momel 기반 Points 티어 생성
# ─────────────────────────────────────────────────────────────────────────────

def generate_momel_labels(sound, pitch, settings, textgrid, duration):
    """
    Momel을 이용하여 Points 티어를 생성하고 *textgrid* 에 추가합니다.

    1. 피치 프레임에서 Doubling/Halving 제거
    2. 침묵 구간 산출 → 음성 구간별로 Momel 실행
    3. Momel 결과를 Points 티어에 반영

    Parameters
    ----------
    sound : parselmouth.Sound
    pitch : Praat Pitch 객체 (이미 추출된 것)
    settings : dict
    textgrid : TextGrid – Points 티어가 *in-place* 로 추가됨
    duration : float

    Returns
    -------
    corrected_times : list[float]
    corrected_f0_values : list[float]
        Doubling/Halving 제거 후의 프레임 데이터
    """
    points_tier = PointTier(name="Points", minTime=0, maxTime=duration)
    textgrid.append(points_tier)

    # 1) 프레임별 (시간, f0) 추출
    frame_times, frame_f0 = extract_pitch_data(pitch)
    num_frames = len(frame_times)

    # 2) Doubling/Halving 제거
    corrected_times, corrected_f0 = remove_doubling_halving(frame_times, frame_f0)
    corrected_set = set(round(t, 6) for t in corrected_times)

    # 3) 침묵 / 음성 구간 계산
    sil_tg = call(
        sound, "To TextGrid (silences)",
        70,
        settings["time_step"],
        settings["sil_thresh"],
        0.25, 0.05,
        settings["sil_label"],
        settings["snd_label"],
    )
    n_intervals = call(sil_tg, "Get number of intervals", 1)
    snd_intervals = []
    for i in range(1, n_intervals + 1):
        label = call(sil_tg, "Get label of interval", 1, i)
        if label == settings["snd_label"]:
            start = call(sil_tg, "Get start time of interval", 1, i)
            end = call(sil_tg, "Get end time of interval", 1, i)
            snd_intervals.append((start, end))

    # f0 유효 범위 (클리핑 용)
    valid_f0 = [fv for fv in corrected_f0 if fv > 0]
    min_f0 = min(valid_f0) if valid_f0 else 0.0
    max_f0 = max(valid_f0) if valid_f0 else 0.0

    momel_cmd = settings["momel_path"]
    os.makedirs('out/models', exist_ok=True)

    temp_f0_min = float('inf')
    temp_f0_max = float('-inf')

    # 4) 음성 구간별로 Momel 실행
    for start_time, end_time in snd_intervals:
        snd_name = f"part_{start_time:.3f}_{end_time:.3f}"
        f0_file = f"{snd_name}.f0"
        momel_file = f"{snd_name}.model"
        f0_path = os.path.join('out/models', f0_file)
        momel_path = os.path.join('out/models', momel_file)

        for p in (f0_path, momel_path):
            if os.path.exists(p):
                os.remove(p)

        start_idx = max(0, int(start_time / settings["time_step"]))
        end_idx = min(num_frames - 1, int(end_time / settings["time_step"]))

        # .f0 파일 생성
        with open(f0_path, 'w') as f:
            for fi in range(start_idx, end_idx + 1):
                t_rounded = round(frame_times[fi], 6)
                f0_val = frame_f0[fi] if t_rounded in corrected_set else 0.0
                f.write(f"{f0_val}\n")

        # Momel 실행
        run_momel_subprocess(momel_cmd, f0_file, momel_file, settings["momel_parameters"])

        # 결과 파싱
        with open(momel_path, 'r') as file:
            for line in file:
                ms_str, f0_str = line.strip().split()
                time_val = start_time + float(ms_str) / 1000.0
                f0_val = max(min(float(f0_str), max_f0), min_f0)
                time_val = max(0.0, min(time_val, duration))

                if f0_val > temp_f0_max:
                    temp_f0_max = f0_val
                if f0_val < temp_f0_min:
                    temp_f0_min = f0_val

                # 중복 포인트 방지
                if not any(pt.time == time_val for pt in points_tier.points):
                    points_tier.add(time_val, f"{f0_val:.2f}")

        # 임시 파일 정리
        os.remove(f0_path)
        os.remove(momel_path)

    logger.info(
        f"[RUN] Momel 음높이 포인트 생성 완료: "
        f"F0 범위: {temp_f0_min:.2f} ~ {temp_f0_max:.2f}"
    )
    return corrected_times, corrected_f0
