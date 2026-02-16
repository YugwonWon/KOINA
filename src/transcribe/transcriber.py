"""
억양 자동 전사 메인 모듈

IntonationTranscriber 클래스와 process_files 오케스트레이션 함수를 제공합니다.
실제 연산 로직은 하위 모듈에 위임합니다:

- transcribe.pitch    : 피치 추출 · 신호 처리
- transcribe.momel    : Momel 실행 · Points 티어 생성
- transcribe.plotting : 시각화 · 그래프 저장
"""

import os
import re
import json
import csv
import traceback
import multiprocessing as mp
from pathlib import Path

# TextGrid 출력 시 제거할 구두점 패턴
_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)

import parselmouth
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect
from textgrid import TextGrid, IntervalTier, PointTier

from utils.file_ops import (
    collect_wav_files, detect_delimiter, ensure_wav, SUPPORTED_AUDIO_EXTENSIONS,
)
from transcribe.aligner import MFAAligner, tg_to_alignment
from transcribe.plotting import (
    get_fontprop,
    plot_pitch_contour,
    plot_momel_points,
    plot_simplified_contour,
    plot_corrected_contour,
    plot_spline_contour,
    plot_percentage_contour,
)
from transcribe.pitch import (
    extract_pitch,
    extract_pitch_data,
    simplify_by_slope,
    apply_cubic_spline,
    calculate_tcog,
    synthesize_modified_wav,
)
from transcribe.momel import generate_momel_labels, get_pitch_points

from utils.logger import main_logger

logger = main_logger.getChild('transcriber')

CONFIG_PATH = "out/config.json"
MOMEL_PATH = "src/lib/momel/momel_linux"


# ─────────────────────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path=CONFIG_PATH, momel_path=MOMEL_PATH):
    """Config 파일을 로드합니다. 파일이 없으면 기본값을 반환합니다."""
    defaults = {
        "min_pitch": 75,
        "min_pitch_male": 75,
        "min_pitch_female": 100,
        "max_pitch": 600,
        "max_pitch_male": 500,
        "max_pitch_female": 600,
        "time_step": 0.01,
        "sil_thresh": -25.0,
        "sil_label": "#",
        "snd_label": "sound",
        "number_of_candidates": 15,
        "very_accurate": 1,
        "silence_threshold": 0.03,
        "voicing_threshold": 0.5,
        "octave_cost": 0.05,
        "octave_jump_cost": 0.5,
        "voice_unvoiced_cost": 0.2,
        "show_spline": False,
        "is_synthesis_save": False,
        "is_spline_syntheis_save": False,
        "is_only_alignment": False,
        "alignment_njobs": 4,
        "alignment_single_spk": False,
        "fixed_y_min": 0,
        "fixed_y_max": 600,
        "momel_parameters": "30 60 750 1.04 20 5 0.05",
        "momel_path": momel_path,
    }

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 런타임 고정 값 덮어쓰기
        config["sil_label"] = "#"
        config["snd_label"] = "sound"
        config["sil_thresh"] = -25.0
        config["momel_parameters"] = "30 60 750 1.01 20 5 0.05"
        config["momel_path"] = momel_path

        # 누락된 기본값 보장
        for key, val in defaults.items():
            config.setdefault(key, val)

        return config

    logger.warning("Config 파일이 없습니다. 기본값을 사용합니다.")
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# IntonationTranscriber
# ─────────────────────────────────────────────────────────────────────────────

class IntonationTranscriber:
    """억양 자동 전사 클래스"""

    def __init__(
        self,
        wav_file: str,
        transcript: str,
        sex: str,
        output_textgrid: str,
        settings=None,
        momel_path: str = MOMEL_PATH,
    ):
        self.wav_file = wav_file
        # 구두점 제거: TextGrid에는 한글과 공백만 포함
        self.transcript = re.sub(r'\s+', ' ', _PUNCT_RE.sub('', transcript)).strip()
        self.output_textgrid = output_textgrid
        self.momel_path = momel_path
        self.textgrid = TextGrid()
        self.sound = parselmouth.Sound(self.wav_file)
        self.duration = self.sound.get_total_duration()
        self.sex = sex
        self.settings = settings or load_config(momel_path=momel_path)
        self.fontprop = get_fontprop()
        self.alignment = None

    # ── 경로 헬퍼 ──

    def _output_path(self, suffix):
        """출력 파일 경로를 생성합니다. (확장자 앞에 suffix 추가)"""
        return os.path.splitext(self.output_textgrid)[0] + suffix

    # ── TextGrid 생성 ──

    def create_textgrid(self):
        """TextGrid 생성 및 정렬 데이터 기반 티어 추가."""
        logger.info(f"[RUN] TextGrid를 생성합니다...: {self.wav_file}")

        # 기본 티어 생성
        utterance_tier = IntervalTier(name="utterance", minTime=0, maxTime=self.duration)
        utterance_tier.add(0, self.duration, self.transcript)
        self.textgrid.append(utterance_tier)

        word_tier = IntervalTier(name="word", minTime=0, maxTime=self.duration)
        self.textgrid.append(word_tier)

        syllable_tier = IntervalTier(name="syllable", minTime=0, maxTime=self.duration)
        self.textgrid.append(syllable_tier)

        phoneme_tier = IntervalTier(name="phoneme", minTime=0, maxTime=self.duration)
        self.textgrid.append(phoneme_tier)

        phonkr_tier = IntervalTier("phoneme_kr", 0, self.duration)
        self.textgrid.append(phonkr_tier)

        if not self.alignment:
            return

        # word 티어 채우기
        words = self.alignment.get('words', [])
        prev_end = 0.0
        for word in words:
            start, end = word.get('start', 0), word.get('end', 0)
            text = word.get('text', '') or 'SP'
            if start > prev_end + 0.001:
                word_tier.add(prev_end, min(start, word_tier.maxTime), 'SP')
            if end > word_tier.maxTime:
                end = word_tier.maxTime
            word_tier.add(start, end, text)
            prev_end = end
        if prev_end < word_tier.maxTime - 0.001:
            word_tier.add(prev_end, word_tier.maxTime, 'SP')

        # phoneme 티어 채우기
        for phoneme in self.alignment.get('phonemes', []):
            start, end = phoneme.get('start', 0), phoneme.get('end', 0)
            text = phoneme.get('text', '') or 'SP'
            if end > phoneme_tier.maxTime:
                end = phoneme_tier.maxTime
            phoneme_tier.add(start, end, text)

        # phoneme_kr 티어 채우기
        for phoneme in self.alignment.get('phonemes_kr', []):
            start, end = phoneme.get('start', 0), phoneme.get('end', 0)
            text = phoneme.get('text', '') or 'SP'
            if end > phonkr_tier.maxTime:
                end = phonkr_tier.maxTime
            phonkr_tier.add(start, end, text)

        # syllable 티어 채우기
        self._fill_syllable_tier(syllable_tier)

    def _fill_syllable_tier(self, syllable_tier):
        """alignment 의 음절 구간 정보로 syllable 티어를 채웁니다."""
        syllables = self.alignment.get('syllables', [])
        if not syllables:
            return

        prev_end = 0.0
        for syl in syllables:
            start, end = syl.get('start', 0), syl.get('end', 0)
            text = syl.get('text', '')
            if start > prev_end + 0.001:
                syllable_tier.add(prev_end, min(start, syllable_tier.maxTime), 'SP')
            if end > syllable_tier.maxTime:
                end = syllable_tier.maxTime
            syllable_tier.add(start, end, text)
            prev_end = end
        if prev_end < syllable_tier.maxTime - 0.001:
            syllable_tier.add(prev_end, syllable_tier.maxTime, 'SP')

    # ── 저장 ──

    def save_textgrid(self):
        """TextGrid 를 파일로 저장합니다."""
        self.textgrid.write(self.output_textgrid)
        logger.info(f"[RUN] TextGrid가 성공적으로 저장되었습니다: {self.output_textgrid}")

    # ── TCoG ──

    def add_tcog_tier(self, pitch):
        """TCoG 티어를 TextGrid 에 추가합니다."""
        tcog = calculate_tcog(pitch)
        if tcog is not None:
            tcog_tier = PointTier(name="TCoG", minTime=0, maxTime=self.duration)
            tcog_tier.add(tcog, "TCoG")
            self.textgrid.append(tcog_tier)
            logger.info(f"[RUN] TCoG 티어를 추가했습니다: {tcog:.2f}")
        else:
            logger.warning("[RUN] TCoG 계산에 실패했습니다.")

    # ── 백분율 정규화 ──

    def add_percentage_points_tier(self, times, f0_values, pitch, output_path):
        """
        백분율 시간축(0~100)으로 정규화한 Points(pct) 티어를 별도 TextGrid 에 저장합니다.
        """
        pct_tg = TextGrid()

        pct_points = PointTier(name="Points(pct)", minTime=0, maxTime=100)
        for t, f in zip(times, f0_values):
            pct_points.add((t / self.duration) * 100, f"{f:.2f}")
        pct_tg.append(pct_points)

        tcog = calculate_tcog(pitch)
        if tcog is not None:
            tcog_pct = PointTier(name="TCoG(pct)", minTime=0, maxTime=100)
            tcog_pct.add((tcog / self.duration) * 100, "TCoG")
            pct_tg.append(tcog_pct)

        pct_tg.write(output_path)
        logger.info(f"[RUN] 백분율 정규화 TextGrid가 저장되었습니다: {output_path}")

    # ── Momel 후처리 (목표점 최소화 · 스플라인 · 합성) ──

    def _apply_momel_modulation(self, pitch):
        """Momel Points 티어에서 목표점 최소화 · 스플라인 · 합성을 적용합니다."""
        points_tier = next(
            (t for t in self.textgrid.tiers if t.name == "Points"), None
        )
        if not points_tier:
            return

        times, f0_values = get_pitch_points(points_tier)

        # 기울기 기반 목표점 최소화
        simplified_t, simplified_f = simplify_by_slope(times, f0_values)

        # 그래프
        plot_simplified_contour(
            simplified_t, simplified_f,
            self._output_path("_momel_pitch_contour_minimalized.jpg"),
            self.textgrid, self.fontprop,
        )

        # 합성 음성 (옵션)
        if self.settings['is_synthesis_save']:
            synthesize_modified_wav(
                self.sound,
                self._output_path("_modified_minimalization.wav"),
                simplified_t, simplified_f, self.duration,
            )

        # 백분율 정규화 TextGrid (옵션)
        if self.settings.get('is_percentage_save', True):
            self.add_percentage_points_tier(
                simplified_t, simplified_f, pitch,
                self._output_path("_pct.TextGrid"),
            )

        # 백분율 그래프
        plot_percentage_contour(
            simplified_t, simplified_f,
            self._output_path("_momel_pitch_percentage.jpg"),
            self.duration, self.fontprop, self.settings,
        )

        # Points 티어를 최소화된 데이터로 업데이트
        points_tier.points.clear()
        for t, f in zip(simplified_t, simplified_f):
            points_tier.add(t, f"{f:.2f}")

        # 3차 스플라인
        spline_t, spline_f = apply_cubic_spline(simplified_t, simplified_f)

        if self.settings['show_spline']:
            plot_spline_contour(
                spline_t, spline_f,
                self._output_path("_spline_contour.jpg"),
                simplified_t, simplified_f,
                self.textgrid, self.fontprop, self.settings,
            )

        if self.settings['is_spline_syntheis_save']:
            synthesize_modified_wav(
                self.sound,
                self._output_path("_spline_contour.wav"),
                spline_t, spline_f, self.duration,
            )

    # ── 메인 실행 ──

    def run(self):
        """전체 전사 프로세스를 실행합니다."""
        # 이미 모든 결과물이 있으면 건너뛰기
        required_outputs = [
            self.output_textgrid,
            self._output_path("_pitch_contour.jpg"),
            self._output_path("_momel_pitch_contour.jpg"),
            self._output_path("_momel_pitch_contour_minimalized.jpg"),
            self._output_path("_modified_minimalization.wav"),
            self._output_path("_corrected_doubling_halving_contour.jpg"),
        ]
        if all(os.path.exists(p) for p in required_outputs):
            logger.info(
                f"[RUN] 모든 출력 파일이 이미 존재하여 건너뜁니다: {self.output_textgrid}"
            )
            return

        try:
            # 1) TextGrid 생성 (정렬 데이터 반영)
            self.create_textgrid()

            if self.settings['is_only_alignment']:
                self.save_textgrid()
                return

            # 2) 피치 추출 (한 번만)
            pitch = extract_pitch(self.sound, self.sex, self.settings)

            # 3) Momel 기반 Points 티어 생성
            corrected_times, corrected_f0 = generate_momel_labels(
                self.sound, pitch, self.settings, self.textgrid, self.duration,
            )

            # 4) Doubling/Halving 제거 그래프
            plot_corrected_contour(
                corrected_times, corrected_f0,
                self._output_path("_corrected_doubling_halving_contour.jpg"),
                self.textgrid, self.fontprop,
            )

            # 5) TCoG
            self.add_tcog_tier(pitch)

            # 6) 원시 음높이 그래프
            pitch_times, pitch_f0 = extract_pitch_data(pitch)
            plot_pitch_contour(
                pitch_times, pitch_f0,
                self._output_path("_pitch_contour.jpg"),
                self.textgrid, self.fontprop,
                show_spline=self.settings['show_spline'],
            )

            # 7) Momel 포인트 그래프
            plot_momel_points(
                self.textgrid,
                self._output_path("_momel_pitch_contour.jpg"),
                self.fontprop,
            )

            # 8) 목표점 최소화 · 스플라인 · 합성
            self._apply_momel_modulation(pitch)

            # 9) 최종 저장
            self.save_textgrid()

        except Exception as e:
            logger.error(f"억양 전사 중 오류 발생: {e}")
            logger.error(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# 멀티프로세싱 워커
# ─────────────────────────────────────────────────────────────────────────────

_worker_settings = None
_worker_momel_path = None


def _init_worker(settings, momel_path):
    """멀티프로세싱 워커 초기화 (각 프로세스 시작 시 1회 호출)."""
    global _worker_settings, _worker_momel_path
    _worker_settings = settings
    _worker_momel_path = momel_path

    import matplotlib
    matplotlib.use('Agg')


def _process_single_file(task):
    """
    단일 파일 처리 워커 함수.

    Args:
        task: (info_dict, alignment_dict) 튜플

    Returns:
        (success: bool, wav_path: str, error_msg: str | None)
    """
    global _worker_settings, _worker_momel_path

    info, alignment = task
    wav_path = info["wav_path"]

    try:
        transcriber = IntonationTranscriber(
            wav_file=str(wav_path),
            transcript=info["transcript"],
            sex=info["sex"],
            output_textgrid=info["output_textgrid"],
            settings=_worker_settings,
            momel_path=_worker_momel_path,
        )
        transcriber.alignment = alignment
        transcriber.run()
        return (True, wav_path, None)
    except Exception as e:
        return (False, wav_path, f"{e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# 파일 오케스트레이션
# ─────────────────────────────────────────────────────────────────────────────

def process_files(
    tsv_file: str,
    output_dir: str,
    stop_flag,
    runner=None,
    momel_path=MOMEL_PATH,
    wav_root_dir: str = "data/source-audio",
    save_dir: str = "out/results",
    n_jobs: int = 4,
):
    """
    TSV 파일을 읽어 MFA 배치 정렬 → 멀티프로세싱 억양 전사를 수행합니다.

    Parameters
    ----------
    tsv_file : str      – 입력 TSV 파일 경로
    output_dir : str    – (하위 호환) save_dir 사용 권장
    stop_flag : Event   – 중지 플래그
    runner : object     – 프론트엔드 Runner (current_aligner 전달용)
    momel_path : str
    wav_root_dir : str  – 오디오 파일 루트
    save_dir : str      – 결과 저장 루트
    n_jobs : int        – 병렬 프로세스 수
    """
    logger.info(f"[FILE] 입력 파일을 처리합니다: {os.path.basename(tsv_file)}")
    logger.info(f"[FILE] WAV 파일 경로: {wav_root_dir}")
    logger.info(f"[FILE] 저장 경로: {save_dir}")
    logger.info(f"[FILE] 병렬 처리 프로세스 수: {n_jobs}")
    os.makedirs(save_dir, exist_ok=True)

    settings = load_config(config_path=CONFIG_PATH, momel_path=momel_path)
    wav_dict = collect_wav_files(wav_root_dir)

    # ── TSV 읽기 → (wav_path, transcript) 쌍 구성 ──
    pairs = []
    info_list = []
    try:
        delimiter = detect_delimiter(tsv_file)
        with open(tsv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                if stop_flag.is_set():
                    logger.info("[RUN] 처리 중지 요청이 감지되어 작업을 중단합니다")
                    return

                wav_file_name = row.get("filename", "").strip()
                file_ext = os.path.splitext(wav_file_name)[1].lower()
                if file_ext not in SUPPORTED_AUDIO_EXTENSIONS:
                    wav_file_name = f"{wav_file_name}.wav"

                transcript = row.get("text", "")
                sex = row.get("sex", "")

                if wav_file_name not in wav_dict:
                    logger.warning(f"[FILE] 오디오 파일을 찾을 수 없습니다: {wav_file_name}")
                    continue

                wav_file_path = wav_dict[wav_file_name]
                try:
                    wav_file_path = ensure_wav(wav_file_path)
                except Exception as e:
                    logger.error(f"[FILE] 오디오 변환 실패, 건너뜁니다: {wav_file_name} ({e})")
                    continue

                base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
                out_subdir = f"{save_dir}/{base_name.split('.')[0]}"
                os.makedirs(out_subdir, exist_ok=True)
                output_textgrid = os.path.join(out_subdir, f"{base_name}_{sex}.TextGrid")

                pairs.append((wav_file_path, transcript))
                info_list.append({
                    "wav_path": wav_file_path,
                    "sex": sex,
                    "output_textgrid": output_textgrid,
                    "transcript": transcript,
                })
    except Exception as e:
        logger.error(f"[FILE] 파일을 준비하는 도중 에러 발생:\n{traceback.format_exc()}")
        return

    if not pairs:
        logger.info("[FILE] 처리할 파일이 없습니다")
        return

    logger.info(f"[FILE] 총 {len(pairs)}개 파일 준비 완료")

    # ── MFA 배치 정렬 ──
    aligner = MFAAligner()
    if runner is not None:
        runner.current_aligner = aligner

    try:
        grid_dict = aligner.align_batch(pairs, stop_flag=stop_flag)
    except Exception as e:
        logger.error(f"[ALIGNER] 배치 정렬 실패: {e}")
        logger.error(traceback.format_exc())
        return
    finally:
        if runner is not None:
            runner.current_aligner = None

    if stop_flag.is_set():
        logger.info("[RUN] 처리 중지 요청이 감지되어 작업을 중단합니다")
        return

    # ── 태스크 구성 ──
    tasks = []
    skipped_count = 0
    for info in info_list:
        wav_path = Path(info["wav_path"]).expanduser().resolve()
        tg = grid_dict.get(wav_path.stem)

        if tg is None:
            logger.warning(f"[RUN] MFA 정렬 결과 없음 (건너뜀): {wav_path}")
            skipped_count += 1
            continue

        alignment = tg_to_alignment(tg, info["transcript"])
        tasks.append((info, alignment))

    if skipped_count > 0:
        logger.warning(f"[RUN] MFA 정렬 결과 없음으로 건너뛴 파일: {skipped_count}개")

    if not tasks:
        logger.info("[FILE] 처리할 태스크가 없습니다")
        return

    total = len(tasks)
    logger.info(f"[RUN] {total}개 파일에 대해 {n_jobs}개 프로세스로 병렬 처리를 시작합니다...")

    # ── 멀티프로세싱 실행 ──
    success_count = 0
    error_count = 0
    LOG_EVERY = max(1, total // 100)

    ctx = mp.get_context('spawn')

    try:
        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(settings, momel_path),
        ) as pool:
            results = pool.imap_unordered(_process_single_file, tasks, chunksize=10)

            with tqdm_logging_redirect(logger):
                for idx, result in enumerate(
                    tqdm(results, total=total, desc="Transcribing",
                         unit="file", ascii=True,
                         mininterval=1.0, dynamic_ncols=False),
                    start=1,
                ):
                    success, wav_path, error_msg = result

                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.error(f"[RUN] 오류 발생: {wav_path}")
                        logger.error(error_msg)

                    if idx == 1 or idx % LOG_EVERY == 0 or idx == total:
                        pct = 100 * idx / total
                        logger.info(
                            f"[PROGRESS] {idx}/{total} ({pct:5.1f}%) "
                            f"- 성공: {success_count}, 실패: {error_count}"
                        )

                    if stop_flag.is_set():
                        logger.info("[RUN] 처리 중지 요청이 감지되어 작업을 중단합니다")
                        pool.terminate()
                        break

    except KeyboardInterrupt:
        logger.info("[RUN] 키보드 인터럽트로 작업을 중단합니다")
        return
    except Exception as e:
        logger.error(f"[RUN] 멀티프로세싱 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return

    # ── 결과 요약 ──
    if stop_flag.is_set():
        logger.info(
            f"[RUN] 작업이 중단되었습니다. "
            f"처리 완료: {success_count}, 실패: {error_count}"
        )
    else:
        logger.info(
            f"[RUN] 모든 파일 처리가 완료되었습니다. "
            f"성공: {success_count}, 실패: {error_count}, 건너뜀: {skipped_count}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI 엔트리 포인트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from threading import Event

    parser = argparse.ArgumentParser(description="억양 자동 주석 도구 (TSV 입력)")
    parser.add_argument(
        "tsv_file", type=str, nargs='?',
        default="data/133_parsed_output_sample.tsv",
        help="입력 TSV 파일 경로 (wavfile_path와 text 컬럼 포함)",
    )
    parser.add_argument("--wav_root_dir", type=str, default='data/source-audio',
                        help="WAV 파일이 있는 디렉토리 경로")
    parser.add_argument("--save_dir", type=str, default='out/results',
                        help="출력 TextGrid 파일들이 저장될 디렉토리 경로")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="병렬 처리할 프로세스 수 (기본값: 4)")

    args = parser.parse_args()
    stop_flag = Event()

    process_files(
        tsv_file=args.tsv_file,
        output_dir=args.save_dir,
        stop_flag=stop_flag,
        runner=None,
        wav_root_dir=args.wav_root_dir,
        save_dir=args.save_dir,
        n_jobs=args.n_jobs,
    )
