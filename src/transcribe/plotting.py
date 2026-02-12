"""
시각화 관련 함수 모듈

- 한글·IPA 폰트 설정
- 음높이 윤곽 그래프
- TextGrid 주석 오버레이
- 백분율 정규화 그래프
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from textgrid import IntervalTier

from utils.logger import main_logger

logger = main_logger.getChild('plotting')

__all__ = [
    "get_fontprop",
    "plot_with_annotations",
    "plot_pitch_contour",
    "plot_momel_points",
    "plot_simplified_contour",
    "plot_corrected_contour",
    "plot_spline_contour",
    "plot_percentage_contour",
]

# ─────────────────────────────────────────────────────────────────────────────
# 폰트 설정
# ─────────────────────────────────────────────────────────────────────────────
_fontprop = None


def get_fontprop():
    """캐시된 FontProperties 를 반환합니다 (최초 호출 시 자동 초기화)."""
    global _fontprop
    if _fontprop is None:
        _fontprop = _setup_korean_font()
    return _fontprop


def _setup_korean_font():
    """한글 + IPA 가 모두 표시되도록 다중-폰트(fallback) 설정."""
    candidates = [
        # 한글·라틴
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # IPA / 기호
        "/usr/local/share/fonts/NotoSansPhonetic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansPhonetic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/charis/CharisSIL-Regular.ttf",
    ]

    fams = []
    for fp in candidates:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            fams.append(fm.FontProperties(fname=fp).get_name())

    if not fams:
        logger.warning("⚠️  한글·IPA 글리프를 가진 폰트를 찾지 못했습니다.")
        return None

    mpl.rcParams["font.family"] = fams
    mpl.rcParams["axes.unicode_minus"] = False
    return fm.FontProperties(family=fams)


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _save_jpg(fig, output_path):
    """JPEG 저장 후 figure 를 닫습니다."""
    plt.savefig(output_path, format="jpg", pil_kwargs={"quality": 85})
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 공통 그래프 그리기
# ─────────────────────────────────────────────────────────────────────────────

def plot_with_annotations(
    ax, times, f0_values, title, label,
    textgrid=None, fontprop=None,
    show_textgrid=True, show_spline=False,
    corrected_times=None, corrected_f0_values=None,
):
    """
    주어진 음높이 데이터를 사용하여 그래프를 그리고 TextGrid 주석을 추가합니다.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    times, f0_values : list[float]
    title, label : str
    textgrid : TextGrid, optional
    fontprop : FontProperties, optional
    show_textgrid : bool
    show_spline : bool
        True 이면 스플라인 윤곽(초록 점선) 및 교정된 포인트(빨간 점) 표시
    corrected_times, corrected_f0_values : list[float], optional
    """
    # f0 = 0 인 구간 필터링
    filtered_times = [t for t, f in zip(times, f0_values) if f > 0]
    filtered_f0 = [f for f in f0_values if f > 0]

    # 기본 음높이 곡선
    if not show_spline:
        ax.plot(filtered_times, filtered_f0,
                linestyle='-', marker='o', markersize=3, label=label)

    ax.set_xlabel("시간 (초)", labelpad=5, loc='right')

    # Doubling/Halving 교정 포인트 (빨간 점)
    if show_spline and corrected_times is not None:
        ct_filt = [ct for ct, cf in zip(corrected_times, corrected_f0_values) if cf > 0]
        cf_filt = [cf for cf in corrected_f0_values if cf > 0]
        ax.scatter(ct_filt, cf_filt, color='red', marker='o', s=30,
                   label='Corrected Points')

    # 스플라인 윤곽 (초록 점선)
    if show_spline:
        ax.plot(filtered_times, filtered_f0,
                color='green', linestyle='--', linewidth=2,
                label='Spline Contour')

    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontproperties=fontprop)
    ax.legend(loc="upper right")

    # TextGrid 주석 (word / phoneme 티어)
    if show_textgrid and textgrid is not None:
        word_y = -0.20
        phoneme_y = -0.35

        for tier in textgrid.tiers:
            if not (isinstance(tier, IntervalTier) and tier.name in ('word', 'phoneme')):
                continue
            y_pos = word_y if tier.name == "word" else phoneme_y
            color = 'red' if tier.name == "word" else 'yellow'
            for interval in tier.intervals:
                mid = (interval.minTime + interval.maxTime) / 2
                ax.text(mid, y_pos, interval.mark,
                        ha='center', va='top', color='black',
                        fontproperties=fontprop,
                        transform=ax.get_xaxis_transform())
                ax.axvline(x=interval.minTime, color=color,
                           linestyle='--', linewidth=0.5)

        plt.subplots_adjust(bottom=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 개별 그래프
# ─────────────────────────────────────────────────────────────────────────────

def plot_pitch_contour(times, f0_values, output_path,
                       textgrid, fontprop, show_spline=False):
    """원시 음높이 윤곽과 TextGrid 주석을 시각화합니다."""
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_with_annotations(
        ax, times, f0_values,
        "음높이 포인트 및 TextGrid 주석", "Pitch Point",
        textgrid=textgrid, fontprop=fontprop,
        show_textgrid=show_spline,
    )
    _save_jpg(fig, output_path)
    logger.info(f"[RUN] 원시 음높이 그래프가 저장되었습니다: {output_path}")


def plot_momel_points(textgrid, output_path, fontprop):
    """Momel Points 티어를 기반으로 음높이 포인트를 시각화합니다."""
    points_tier = next((t for t in textgrid.tiers if t.name == "Points"), None)
    if not points_tier:
        logger.warning("Points 티어를 찾을 수 없습니다.")
        return
    times = [p.time for p in points_tier.points]
    f0_values = [float(p.mark) for p in points_tier.points]

    fig, ax = plt.subplots(figsize=(15, 5))
    plot_with_annotations(
        ax, times, f0_values,
        "Momel 음높이 포인트와 TextGrid 주석", "Momel Pitch Point",
        textgrid=textgrid, fontprop=fontprop,
    )
    _save_jpg(fig, output_path)


def plot_simplified_contour(times, f0_values, output_path, textgrid, fontprop):
    """F0 목표점 최소화 포인트 그래프를 저장합니다."""
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_with_annotations(
        ax, times, f0_values,
        "Momel 음높이 포인트와 TextGrid 주석 (음높이 목표점 최소화)",
        "pitch target minimalized Momel Pitch Point",
        textgrid=textgrid, fontprop=fontprop,
    )
    _save_jpg(fig, output_path)
    logger.info(f"[RUN] F0 목표점 최소화 포인트 그래프가 저장되었습니다: {output_path}")


def plot_corrected_contour(times, f0_values, output_path, textgrid, fontprop):
    """Doubling/Halving 제거 후 음높이 포인트 그래프를 저장합니다."""
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_with_annotations(
        ax, times, f0_values,
        "배증/반감 제거된 음높이 포인트", "Corrected Pitch Points",
        textgrid=textgrid, fontprop=fontprop,
    )
    _save_jpg(fig, output_path)
    logger.info(f"[RUN] Doubling/Halving 제거된 음높이 포인트 그래프가 저장되었습니다: {output_path}")


def plot_spline_contour(times, f0_values, output_path,
                        corrected_times, corrected_f0_values,
                        textgrid, fontprop, settings):
    """삼차 스플라인 음높이 윤곽 그래프를 저장합니다."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_ylim(settings['fixed_y_min'], settings['fixed_y_max'])
    plot_with_annotations(
        ax, times, f0_values,
        "삼차 스플라인 음높이 윤곽", "Spline Pitch Contour",
        textgrid=textgrid, fontprop=fontprop,
        show_spline=settings['show_spline'],
        corrected_times=corrected_times,
        corrected_f0_values=corrected_f0_values,
    )
    _save_jpg(fig, output_path)
    logger.info(f"[RUN] 삼차 스플라인 그래프가 저장되었습니다: {output_path}")


def plot_percentage_contour(times, f0_values, output_path,
                            duration, fontprop, settings):
    """백분율 정규화 음높이 그래프를 저장합니다."""
    pct_times = [(t / duration) * 100 for t in times]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(pct_times, f0_values, color='blue', linestyle='-',
            marker='o', markersize=3, label="Percentage Pitch Contour")
    ax.set_ylim(settings['fixed_y_min'], settings['fixed_y_max'])
    ax.set_xlabel("Time (%)", labelpad=5, loc='right')
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Percentage Normalized Pitch Contour", fontproperties=fontprop)
    ax.legend(loc="upper right")
    _save_jpg(fig, output_path)
    logger.info(f"[RUN] 백분율 기반 음높이 그래프가 저장되었습니다: {output_path}")
