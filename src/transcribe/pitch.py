"""
피치(F0) 추출 및 신호 처리 함수 모듈

- Praat 기반 피치 추출 (성별별 파라미터)
- Doubling/Halving 제거
- 기울기 기반 F0 목표점 최소화
- 3차 스플라인 보간
- TCoG 계산
- 피치 변조 합성 음성 생성
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline
from parselmouth.praat import call

from utils.logger import main_logger

logger = main_logger.getChild('pitch')

__all__ = [
    "extract_pitch",
    "extract_pitch_data",
    "remove_doubling_halving",
    "simplify_by_slope",
    "apply_cubic_spline",
    "calculate_tcog",
    "synthesize_modified_wav",
]

# ─────────────────────────────────────────────────────────────────────────────
# 피치 추출
# ─────────────────────────────────────────────────────────────────────────────

_SEX_MALE = {"M", "m", "male", "man", "boy", "남성", "남", "남자"}
_SEX_FEMALE = {"F", "f", "female", "woman", "girl", "여성", "여", "여자"}


def extract_pitch(sound, sex, settings):
    """
    Praat 피치 추출 (성별별로 min_pitch / max_pitch 를 다르게 적용).

    Returns
    -------
    pitch : Praat Pitch 객체
    """
    local_min = settings["min_pitch"]
    local_max = settings["max_pitch"]

    if sex in _SEX_MALE:
        local_min = settings.get("min_pitch_male", local_min)
        local_max = settings.get("max_pitch_male", local_max)
    elif sex in _SEX_FEMALE:
        local_min = settings.get("min_pitch_female", local_min)
        local_max = settings.get("max_pitch_female", local_max)

    return call(
        sound, "To Pitch (ac)",
        settings["time_step"],
        local_min,
        settings["number_of_candidates"],
        settings["very_accurate"],
        settings["silence_threshold"],
        settings["voicing_threshold"],
        settings["octave_cost"],
        settings["octave_jump_cost"],
        settings["voice_unvoiced_cost"],
        local_max,
    )


def extract_pitch_data(pitch):
    """
    Pitch 객체에서 전체 프레임의 (시간, f0) 리스트를 추출합니다.
    무음·NaN 프레임은 f0 = 0.0 으로 처리합니다.

    Returns
    -------
    times : list[float]
    f0_values : list[float]
    """
    num_frames = call(pitch, "Get number of frames")
    times = []
    f0_values = []
    for i in range(1, num_frames + 1):
        t = call(pitch, "Get time from frame number", i)
        f0 = call(pitch, "Get value in frame", i, "Hertz")
        if f0 is None or f0 <= 0 or math.isnan(f0):
            f0 = 0.0
        times.append(t)
        f0_values.append(f0)
    return times, f0_values


# ─────────────────────────────────────────────────────────────────────────────
# Doubling / Halving 제거
# ─────────────────────────────────────────────────────────────────────────────

def remove_doubling_halving(
    times,
    f0_values,
    threshold_ratio=0.5,
    min_stable_count=9,
    global_deviation_factor=2.0,
):
    """
    Doubling/Halving 현상을 감지·제거합니다.

    - 전체 평균 f0 와 비교하여 초기에 지나치게 높은(또는 낮은) 값도 자동 필터링
    - 안정 구간(stable region) 판정 후에는 ratio 방식으로 판정

    Returns
    -------
    corrected_times : list[float]
    corrected_f0 : list[float]
    """
    if not f0_values:
        return [], []

    nonzero = [fv for fv in f0_values if fv > 0]
    if not nonzero:
        return [], []

    global_avg = sum(nonzero) / len(nonzero)
    corrected_times = []
    corrected_f0 = []
    stable_start_index = None
    temp_buffer = []  # (t, f) 임시 버퍼

    for i, (t, f) in enumerate(zip(times, f0_values)):
        # 0Hz 처리
        if f <= 0:
            if stable_start_index is not None:
                corrected_times.append(t)
                corrected_f0.append(f)
            continue

        # 안정 구간 이전: 글로벌 이상치 필터링
        if stable_start_index is None and f > global_avg * global_deviation_factor:
            continue

        # 안정 구간 탐색 단계
        if stable_start_index is None:
            temp_buffer.append((t, f))

            if len(temp_buffer) >= min_stable_count:
                all_stable = True
                for j in range(1, len(temp_buffer)):
                    prev_f, curr_f = temp_buffer[j - 1][1], temp_buffer[j][1]
                    if prev_f > 0 and curr_f > 0:
                        ratio = curr_f / prev_f
                        if not (threshold_ratio <= ratio <= 1.0 / threshold_ratio):
                            all_stable = False
                            break

                if all_stable:
                    stable_start_index = i - (min_stable_count - 1)
                    for bt, bf in temp_buffer:
                        corrected_times.append(bt)
                        corrected_f0.append(bf)
                    temp_buffer.clear()
        else:
            # 안정 구간 내: 최근 유효 f0 와 ratio 비교
            prev_f0 = next((v for v in reversed(corrected_f0) if v != 0), 0)
            if prev_f0 > 0 and f > 0:
                ratio = f / prev_f0
                if not (threshold_ratio <= ratio <= 1.0 / threshold_ratio):
                    continue  # Doubling/Halving → 버림

            corrected_times.append(t)
            corrected_f0.append(f)

    return corrected_times, corrected_f0


# ─────────────────────────────────────────────────────────────────────────────
# F0 목표점 최소화
# ─────────────────────────────────────────────────────────────────────────────

def simplify_by_slope(times, f0_values, slope_threshold=27):
    """
    기울기 기반으로 직선 구간의 중간 포인트를 제거합니다.

    Returns
    -------
    simplified_times : list[float]
    simplified_f0 : list[float]
    """
    simplified_t = [times[0]]
    simplified_f = [f0_values[0]]

    i = 1
    while i < len(times) - 1:
        main_slope = (f0_values[i + 1] - f0_values[i - 1]) / (times[i + 1] - times[i - 1])
        mid_slope = (f0_values[i] - f0_values[i - 1]) / (times[i] - times[i - 1])
        if abs(main_slope - mid_slope) > slope_threshold:
            simplified_t.append(times[i])
            simplified_f.append(f0_values[i])
        i += 1

    simplified_t.append(times[-1])
    simplified_f.append(f0_values[-1])
    return simplified_t, simplified_f


# ─────────────────────────────────────────────────────────────────────────────
# 3차 스플라인 보간
# ─────────────────────────────────────────────────────────────────────────────

def apply_cubic_spline(times, f0_values, num_points=100):
    """
    3차 스플라인 보간을 적용합니다.

    Returns
    -------
    interp_times : np.ndarray
    interp_f0 : np.ndarray
    """
    spline = CubicSpline(times, f0_values, bc_type='natural')
    interp_t = np.linspace(min(times), max(times), num_points)
    return interp_t, spline(interp_t)


# ─────────────────────────────────────────────────────────────────────────────
# TCoG (Tonal Center of Gravity)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_tcog(pitch):
    """
    TCoG(Tonal Center of Gravity)를 계산합니다.

    Returns
    -------
    float or None
    """
    num_frames = call(pitch, "Get number of frames")
    total_weighted_time = 0.0
    total_f0 = 0.0
    for i in range(1, num_frames + 1):
        t = call(pitch, "Get time from frame number", i)
        f0 = call(pitch, "Get value in frame", i, "Hertz")
        if f0 > 0:
            total_weighted_time += t * f0
            total_f0 += f0
    return total_weighted_time / total_f0 if total_f0 > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# 합성 음성 생성
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_modified_wav(sound, output_path, times, f0_values, duration):
    """
    피치를 변조하여 새 WAV 파일을 생성합니다.

    Momel 목표점 최소화 · 스플라인 합성 등 목적에 공통으로 사용됩니다.
    """
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Remove points between", 0, duration)

    for t, f in zip(times, f0_values):
        call(pitch_tier, "Add point", t, f)

    call([pitch_tier, manipulation], "Replace pitch tier")
    result = call(manipulation, "Get resynthesis (overlap-add)")
    result.save(output_path, 'WAV')
    logger.info(f"[RUN] 변조 음성을 저장했습니다: {output_path}")
