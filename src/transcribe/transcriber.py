
import os
import shutil
import json
import csv
import subprocess
import traceback
import numpy as np
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from scipy.interpolate import CubicSpline

import parselmouth
from parselmouth.praat import call

from textgrid import TextGrid, IntervalTier, PointTier

from client.aligner import BaikalSTTClient
from utils.logger import main_logger

# 자식 로거 설정
logger = main_logger.getChild('transcriber')

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
logger.handlers[0].flush()

logger.info("transcriber 시작!")

class IntonationTranscriber:
    """
    억양 자동 전사 클래스
    """
    _baikal_client = None
    _fontprop = None
    _settings = None

    @classmethod
    def get_baikal_client(cls):
        if cls._baikal_client is None:
            cls._baikal_client = BaikalSTTClient()
        return cls._baikal_client

    @classmethod
    def get_fontprop(cls):
        if cls._fontprop is None:
            cls._fontprop = cls.set_korean_font(cls)
        return cls._fontprop

    @classmethod
    def get_settings(cls, config_path, momel_path):
        if cls._settings is None:
            cls._settings = cls.load_config(config_path, momel_path)
        return cls._settings

    @classmethod
    def set_korean_font(cls):
        """
        한글 폰트를 설정하여 반환합니다.
        """
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if not os.path.exists(font_path):
            print("경고: 한글 폰트를 찾을 수 없습니다.")
            return None
        else:
            fontprop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=fontprop.get_name())
            return fontprop

    @classmethod
    def load_config(cls, config_path="out/config.json", momel_path="src/lib/momel/momel_linux"):
        """
        Config 파일 로드
        """
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                config["sil_label"] = "#"
                config["snd_label"] = "sound"
                config["sil_thresh"] = -25.0
                config["momel_parameters"] = "30 60 750 1.04 20 5 0.05"
                config["momel_path"] = momel_path
                return config
        logger.warning(f"Config 파일이 없습니다. 기본값을 사용합니다.")
        return {
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
            "fixed_y_range": 600,
            "momel_parameters": "30 60 750 1.04 20 5 0.05",
            "momel_path": momel_path
        }

    def __init__(self, wav_file: str, transcript: str, sex: str, output_textgrid: str,
                 config_path="out/config.json",
                 momel_path: str = "src/lib/momel/momel_linux"):
        # 초기화 변수
        self.wav_file = wav_file
        self.transcript = transcript
        self.output_textgrid = output_textgrid
        self.momel_path = momel_path
        self.textgrid = TextGrid()
        self.sound = parselmouth.Sound(self.wav_file)
        self.duration = self.sound.get_total_duration()
        self.sex = sex

        # cls 변수
        self.settings = self.get_settings(config_path, momel_path)
        self.baiakal_client = self.get_baikal_client()
        self.fontprop = self.get_fontprop()
        self.alignment = None



    def set_korean_font(self):
        """
        한글 폰트를 설정하여 반환합니다.
        """
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 시스템에 설치된 한글 폰트 경로 설정
        if not os.path.exists(font_path):
            print("경고: 한글 폰트를 찾을 수 없습니다. 시스템 폰트를 확인하거나 경로를 수정하십시오.")
            return None
        else:
            fontprop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=fontprop.get_name())
            return fontprop

    def perform_alignment(self):
        """
        강제 정렬 수행
        """
        logger.info(f"강제 정렬을 시작합니다... (파일: {self.wav_file})")
        self.alignment = self.baiakal_client.align(self.wav_file, self.transcript)
        if not self.alignment:
            raise ValueError(f"강제 정렬에 실패했습니다. (파일: {self.wav_file})")
        logger.info(f"강제 정렬이 완료되었습니다. (파일: {self.wav_file})")
        # alignment 내용 출력 (디버깅 용도)
        logger.debug(f"Alignment 결과: {json.dumps(self.alignment, indent=4, ensure_ascii=False)}")

    def extract_pitch(self, sex):
        """
        파생된 pitch 추출 (gender별로 min_pitch와 max_pitch를 다르게 적용)
        :param sex: 해당 음성 파일의 성별 정보
        """
        # 우선 기본값(또는 config.json의 값)을 사용
        local_min_pitch = self.settings["min_pitch"]
        local_max_pitch = self.settings["max_pitch"]

        # 성별 정보가 있다면, 해당 정보를 우선 적용
        if sex == "M":
            local_min_pitch = self.settings["min_pitch_male"]
            local_max_pitch = self.settings["max_pitch_male"]
        elif sex == "F":
            local_min_pitch = self.settings["min_pitch_female"]
            local_max_pitch = self.settings["max_pitch_female"]

        logger.info(
            f"[extract_pitch] sex={sex}, "
            f"min_pitch={local_min_pitch}, max_pitch={local_max_pitch}, file={self.wav_file}"
        )

        pitch = call(
            self.sound, "To Pitch (ac)",
            self.settings["time_step"],
            local_min_pitch,
            self.settings["number_of_candidates"],
            self.settings["very_accurate"],
            self.settings["silence_threshold"],
            self.settings["voicing_threshold"],
            self.settings["octave_cost"],
            self.settings["octave_jump_cost"],
            self.settings["voice_unvoiced_cost"],
            local_max_pitch
        )
        return pitch

    def run_momel_based_labels(self):
        """
        Momel을 이용하여 Points 티어를 생성
        """
        logger.info(f"Momel을 사용하여 Points 티어를 생성합니다... (파일: {self.wav_file})")
        points_tier = PointTier(name="Points", minTime=0, maxTime=self.duration)
        self.textgrid.append(points_tier)

        duration = call(self.sound, "Get total duration")

        pitch = self.extract_pitch(sex=self.sex)

        matrix = call(pitch, "To Matrix")
        min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

        sil_tg = call(self.sound, "To TextGrid (silences)", 70, self.settings["time_step"], self.settings["sil_thresh"], 0.25, 0.05, self.settings["sil_label"], self.settings["snd_label"])
        n_intervals = call(sil_tg, "Get number of intervals", 1)

        snd_intervals = []
        for i in range(1, n_intervals + 1):
            label = call(sil_tg, "Get label of interval", 1, i)
            if label == self.settings["snd_label"]:
                start_time = call(sil_tg, "Get start time of interval", 1, i)
                end_time = call(sil_tg, "Get end time of interval", 1, i)
                snd_intervals.append((start_time, end_time))

        momel_cmd = self.settings["momel_path"]

        temp_f0_min = float('inf')
        temp_f0_max = float('-inf')

        for start_time, end_time in snd_intervals:
            snd_interval_name = f"part_{start_time:.3f}_{end_time:.3f}"
            f0_file = f"{snd_interval_name}.f0"
            momel_file = f"{snd_interval_name}.model"

            os.makedirs('out/models', exist_ok=True)

            if os.path.exists(os.path.join('out/models', f0_file)):
                os.remove(os.path.join('out/models', f0_file))
            if os.path.exists(os.path.join('out/models', momel_file)):
                os.remove(os.path.join('out/models', momel_file))

            for t in range(int(start_time / self.settings["time_step"]), int(end_time / self.settings["time_step"])):
                try:
                    pitch_value = call(matrix, "Get value in cell", 1, t + 1)
                    with open(os.path.join('out/models', f0_file), 'a') as file:
                        file.write(f"{pitch_value}\n")
                except:
                    continue

            # Momel 실행 함수
            self.run_momel(momel_cmd, momel_file, f0_file)

            with open(os.path.join('out/models', momel_file), 'r') as file:
                lines = file.readlines()

            for line in lines:
                ms, f0 = line.strip().split()
                ms = float(ms)
                f0 = float(f0)
                time = ms / 1000 + start_time

                f0 = max(min(f0, max_f0), min_f0)

                time = max(min(time, duration), 0)

                if f0 > temp_f0_max:
                    temp_f0_max = f0
                if f0 < temp_f0_min:
                    temp_f0_min = f0

                # 중복 포인트 방지
                existing_points = [point for point in points_tier.points if point.time == time]
                if not existing_points:
                    points_tier.add(time, f"{float(f0):.2f}")

            os.remove(os.path.join('out/models', f0_file))
            os.remove(os.path.join('out/models', momel_file))

        # f0 범위 계산 (현재 로직에서는 Ranges 티어를 사용하지 않으므로 생략)

    def run_momel(self, momel_cmd: str, momel_file: str, f0_file: str):
        try:
            momel_cmd = os.path.abspath(momel_cmd)  # 절대 경로 변환
            f0_path = os.path.abspath(f'out/models/{f0_file}')
            momel_out = os.path.abspath(f'out/models/{momel_file}')

            # 경로가 포함된 환경 변수 설정
            env = os.environ.copy()
            env['PATH'] = f'{os.path.dirname(momel_cmd)}:{env["PATH"]}'

            command = f'{momel_cmd} {self.settings["momel_parameters"]} <"{f0_path}" >"{momel_out}"'
            subprocess.run(command, shell=True, check=True, env=env)
            logger.info(f'Momel 실행 완료: {momel_file}')
        except subprocess.CalledProcessError as e:
            logger.error(f'Momel 실행 중 오류 발생: {e}')

    def create_textgrid(self):
        """
        TextGrid 생성 및 티어 추가
        """
        logger.info(f"TextGrid를 생성합니다... (파일: {self.wav_file})")

        # utterance 티어 생성
        utterance_tier = IntervalTier(name="utterance", minTime=0, maxTime=self.duration)
        utterance_tier.add(0, self.duration, self.transcript)
        self.textgrid.append(utterance_tier)

        # word 티어 생성
        word_tier = IntervalTier(name="word", minTime=0, maxTime=self.duration)
        self.textgrid.append(word_tier)

        # phoneme 티어 생성
        phoneme_tier = IntervalTier(name="phoneme", minTime=0, maxTime=self.duration)
        self.textgrid.append(phoneme_tier)

        # alignment 데이터가 존재할 경우, word 및 phoneme 티어 채우기
        if self.alignment:
            # word 티어 채우기
            words = self.alignment.get('words', [])
            for word in words:
                start = word.get('start', 0)
                end = word.get('end', 0)
                text = word.get('text', '')
                word_tier.add(start, end, text)

            # phoneme 티어 채우기
            phonemes = self.alignment.get('phonemes', [])
            for phoneme in phonemes:
                start = phoneme.get('start', 0)
                end = phoneme.get('end', 0)
                text = phoneme.get('text', '')
                phoneme_tier.add(start, end, text)

    def save_textgrid(self):
        """
        TextGrid 저장
        """
        logger.info(f"TextGrid를 {self.output_textgrid}에 저장합니다...")
        self.textgrid.write(self.output_textgrid)
        logger.info(f"TextGrid가 성공적으로 저장되었습니다. (파일: {self.output_textgrid})")
        # # WAV 파일을 TextGrid 위치로 복사
        # wav_output_path = self.output_textgrid.replace('.TextGrid', '.wav')
        # try:
        #     shutil.copy(self.wav_file, wav_output_path)
        #     logger.info(f"WAV 파일을 {wav_output_path}에 복사했습니다.")
        # except Exception as e:
        #     logger.error(f"WAV 파일을 복사하는 중 오류가 발생했습니다: {e}")

    def calculate_tcog(self, pitch):
        """
        TCoG(Tonal Center of Gravity) 계산
        """
        num_frames = call(pitch, "Get number of frames")
        total_weighted_time = 0
        total_f0 = 0

        for i in range(1, num_frames + 1):
            time = call(pitch, "Get time from frame number", i)
            f0 = call(pitch, "Get value in frame", i, "Hertz")
            if f0 > 0:  # 무음 구간은 제외
                total_weighted_time += time * f0
                total_f0 += f0

        if total_f0 == 0:
            return None

        tcog = total_weighted_time / total_f0
        return tcog

    def add_tcog_tier(self):
        """
        TCoG 티어를 TextGrid에 추가
        """
        pitch = self.extract_pitch(sex=self.sex)
        tcog = self.calculate_tcog(pitch)

        if tcog is not None:
            # TCoG PointTier 생성 및 추가
            tcog_tier = PointTier(name="TCoG", minTime=0, maxTime=self.duration)
            tcog_tier.add(tcog, "TCoG")
            self.textgrid.append(tcog_tier)
            logger.info(f"TCoG 티어를 추가했습니다: {tcog}초")
        else:
            logger.warning("TCoG 계산에 실패했습니다.")

    def add_percentage_points_tier(self, corrected_times, corrected_f0_values):
        """
        최종 조정된 포인트를 기반으로 Points(pct) 티어를 추가합니다.
        """
        percentage_points_tier = PointTier(name="Points(pct)", minTime=0, maxTime=100)

        for time, f0 in zip(corrected_times, corrected_f0_values):
            percentage_time = (time / self.duration) * 100
            percentage_points_tier.add(percentage_time, f"{f0:.2f}")

        self.textgrid.append(percentage_points_tier)
        logger.info("최종 조정된 Points(pct) 티어가 성공적으로 추가되었습니다.")

    def simplify_pitch_points_by_slope(self, times, f0_values, slope_threshold=33):
        """
        음높이 포인트를 기울기 기반으로 단순화하여 직선 상의 중간 포인트를 제거합니다.

        Parameters:
            times (list): 음높이 포인트의 시간 리스트.
            f0_values (list): 음높이 포인트의 f0 값 리스트.
            slope_threshold (float): 기울기 차이 임계값.

        Returns:
            list: 단순화된 시간과 f0 값 리스트.
        """
        simplified_times = [times[0]]
        simplified_f0_values = [f0_values[0]]

        i = 1
        while i < len(times) - 1:
            # 첫 번째와 세 번째 포인트의 기울기 계산
            main_slope = (f0_values[i + 1] - f0_values[i - 1]) / (times[i + 1] - times[i - 1])
            # 두 번째 포인트에서의 기울기 계산
            mid_slope = (f0_values[i] - f0_values[i - 1]) / (times[i] - times[i - 1])

            # 기울기 차이가 임계값 이하이면 중간 포인트 제거
            if abs(main_slope - mid_slope) <= slope_threshold:
                i += 1  # 중간 포인트 건너뜀
            else:
                # 중요 변화가 있으므로 포인트 유지
                simplified_times.append(times[i])
                simplified_f0_values.append(f0_values[i])
                i += 1

        # 마지막 포인트를 추가
        simplified_times.append(times[-1])
        simplified_f0_values.append(f0_values[-1])

        return simplified_times, simplified_f0_values


    def synthesize_pitch_modified_wav(self, output_wav_path, times, f0_values):
        """
        Momel의 Points 티어 또는 단순화된 음높이를 기반으로 pitch를 변조하여 새로운 WAV 파일로 저장합니다.
        """
        manipulation = call(self.sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(pitch_tier, "Remove points between", 0, self.duration)

        for i, time in enumerate(times):
            call(pitch_tier, "Add point", time, f0_values[i])

        call([pitch_tier, manipulation], "Replace pitch tier")
        manipulated_sound = call(manipulation, "Get resynthesis (overlap-add)")
        manipulated_sound.save(output_wav_path, 'WAV')
        logger.info(f"변조된 pitch 음성을 {output_wav_path}에 저장했습니다.")

    def get_momel_pitch_points(self, points_tier):
        """
        Momel Points 티어에서 시간과 음높이 값을 추출합니다.
        """
        times = [point.time for point in points_tier.points]
        f0_values = [float(point.mark) for point in points_tier.points]
        return times, f0_values

    def plot_graph_with_annotations(self, ax, times, f0_values, title, label, show_textgrid=True, show_spline=False, corrected_times=None, corrected_f0_values=None):
        """
        주어진 음높이 데이터를 사용하여 그래프를 그리고 TextGrid 주석을 추가합니다.

        Parameters:
            ax: matplotlib Axes 객체
            times: 음높이 포인트 시간 리스트
            f0_values: 음높이 포인트 f0 값 리스트
            title: 그래프 제목
            label: 범례에 사용할 레이블
            show_textgrid: 텍스트 그리드를 표시할지 여부
        """
        if not show_spline:
            ax.plot(times, f0_values, color='blue', linestyle='-', marker='o', markersize=3, label=label)


        # x축 제목을 오른쪽 끝에 배치
        ax.set_xlabel("시간 (초)", labelpad=5, loc='right')

        # Doubling/Halving 제거된 포인트 (show_spline=True일 때만 표시)
        if show_spline and corrected_times is not None and corrected_f0_values is not None:
            ax.scatter(corrected_times, corrected_f0_values, color='red', marker='o', s=30, label='Corrected Points')

        # 스플라인 윤곽 추가
        if show_spline:
            ax.plot(times, f0_values, color='green', linestyle='--', linewidth=2, label='Spline Contour')

        # # y축 고정(개별 파일의 억양 비교를 위해 필요한 경우 설정)
        # if os.path.basename(self.wav_file).split('.wav')[0] == 'SDRW2200000001.1.1.1':
        #     ax.set_ylim(0, 300)

        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title, fontproperties=self.fontprop)
        ax.legend(loc="upper right")

        # TextGrid 주석 추가 (show_textgrid가 True일 때만)
        if show_textgrid:
            # word와 phoneme 주석을 x축 제목 아래에 배치
            word_y_position = -0.20  # word 레이블 위치 (데이터 영역을 기준으로 상대 위치)
            phoneme_y_position = -0.35  # phoneme 레이블 위치

            for tier in self.textgrid.tiers:
                if isinstance(tier, IntervalTier) and tier.name in ['word', 'phoneme']:
                    y_position = word_y_position if tier.name == "word" else phoneme_y_position
                    color = 'red' if tier.name == "word" else 'yellow'
                    for interval in tier.intervals:
                        start_time = interval.minTime
                        mid_time = (start_time + interval.maxTime) / 2
                        ax.text(mid_time, y_position, interval.mark, ha='center', va='top', color='black', fontproperties=self.fontprop, transform=ax.get_xaxis_transform())
                        ax.axvline(x=start_time, color=color, linestyle='--', linewidth=0.5)

        # 하단 여백 조정
        plt.subplots_adjust(bottom=0.3)  # 하단 여백을 적절하게 조정

    def plot_pitch_and_textgrid(self, pitch):
        """
        음높이 윤곽과 TextGrid 주석을 시각화하여 이미지로 저장합니다.
        """
        times = []
        f0_values = []

        # 음높이 객체에서 음높이 포인트를 추출
        num_frames = call(pitch, "Get number of frames")
        for i in range(1, num_frames + 1):
            time = call(pitch, "Get time from frame number", i)
            f0 = call(pitch, "Get value in frame", i, "Hertz")
            if f0 > 0:  # 무음 구간 제외
                times.append(time)
                f0_values.append(f0)

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(15, 5))
        self.plot_graph_with_annotations(ax, times, f0_values, "음높이 포인트 및 TextGrid 주석", "Pitch Point", show_textgrid=self.settings['show_spline'])

        # JPG로 저장
        output_image_path = os.path.splitext(self.output_textgrid)[0] + "_pitch_contour.jpg"

        plt.savefig(output_image_path, format="jpg", pil_kwargs={"quality": 85})  # JPEG로 저장 시 품질 설정

        plt.close()
        logger.info(f"그래프가 저장되었습니다: {output_image_path}")

    def plot_momel_pitch_points(self):
        """
        Momel에서 생성된 Points 티어를 기반으로 음높이 포인트들을 시각화하여 이미지로 저장합니다.
        """
        # Points 티어 검색
        points_tier = next((tier for tier in self.textgrid.tiers if tier.name == "Points"), None)
        if not points_tier:
            logger.warning("Points 티어를 찾을 수 없습니다.")
            return

        # Points 티어에서 시간과 음높이 값을 추출
        times = [point.time for point in points_tier.points]
        f0_values = [float(point.mark) for point in points_tier.points]

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(15, 5))
        self.plot_graph_with_annotations(ax, times, f0_values, "Momel 음높이 포인트와 TextGrid 주석", "Momel Pitch Point", show_textgrid=True)

        # JPEG로 바로 저장
        output_image_path = os.path.splitext(self.output_textgrid)[0] + "_momel_pitch_contour.jpg"
        plt.savefig(output_image_path, format="jpg", pil_kwargs={"quality": 85})  # JPEG로 저장 시 품질 설정
        plt.close()

    def plot_simplified_pitch_contour(self, times, f0_values, output_path):
        """
        단순화된 음높이 포인트를 사용하여 Momel Pitch contour를 그려서 저장합니다.
        """
        fig, ax = plt.subplots(figsize=(15, 5))
        self.plot_graph_with_annotations(ax, times, f0_values, "Momel 음높이 포인트와 TextGrid 주석 (음높이 목표점 최소화)", "pitch target minimalized Momel Pitch Point")
        plt.savefig(output_path, format="jpg", pil_kwargs={"quality": 85})  # JPEG로 저장 시 품질 설정
        plt.close()
        logger.info(f"단순화된 Momel 음높이 포인트 그래프가 저장되었습니다: {output_path}")

    def plot_doubling_halving_corrected_pitch_contour(self, times, f0_values, output_path):
        """
        Doubling 및 Halving이 제거된 음높이 포인트를 사용하여 그래프를 그려서 저장합니다.
        """
        fig, ax = plt.subplots(figsize=(15, 5))
        self.plot_graph_with_annotations(ax, times, f0_values, "Doubling/Halving 제거된 Momel 음높이 포인트", "Corrected Pitch Points")
        plt.savefig(output_path, format="jpg", pil_kwargs={"quality": 85})
        plt.close()
        logger.info(f"Doubling/Halving 제거된 음높이 포인트 그래프가 저장되었습니다: {output_path}")

    def plot_spline_contour(self, times, f0_values, output_path, corrected_times, corrected_f0_values, y_fixed_range=None):
        """
        삼차 스플라인 음높이 포인트를 사용하여 그래프를 그려서 저장합니다.
        """
        fig, ax = plt.subplots(figsize=(15, 5))
        # y축 고정 (사용자가 입력한 범위 적용)
        if y_fixed_range:
            ax.set_ylim((0, y_fixed_range))
        self.plot_graph_with_annotations(ax, times, f0_values, "삼차 스플라인 음높이 윤곽", "Spline Pitch Contour", show_spline=self.settings['show_spline'], corrected_times=corrected_times, corrected_f0_values=corrected_f0_values)
        plt.savefig(output_path, format="jpg", pil_kwargs={"quality": 85})
        plt.close()
        logger.info(f"스플라인 음높이 포인트 그래프가 저장되었습니다: {output_path}")
        
    def plot_percentage_pitch_contour(self, times, f0_values, output_image_path, y_fixed_range=600):
        """
        백분율 기반으로 정규화된 음높이 포인트를 그래프로 그립니다.

        Parameters:
            times (list): 시간 리스트.
            f0_values (list): 음높이 값 리스트.
            output_image_path (str): 출력 이미지 경로.
            y_fixed_range (tuple, optional): y축의 최소값과 최대값. 예: (0, 600).
        """
        # 시간 값을 퍼센테이지로 변환
        percentage_times = [(time / self.duration) * 100 for time in times]

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(percentage_times, f0_values, color='blue', linestyle='-', marker='o', markersize=3, label="Percentage Pitch Contour")

        # y축 고정 (사용자가 입력한 범위 적용)
        if y_fixed_range:
            ax.set_ylim((0, y_fixed_range))
            
        # x축과 y축 레이블 설정
        ax.set_xlabel("Time (%)", labelpad=5, loc='right')
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Percentage Normalized Pitch Contour", fontproperties=self.fontprop)

        # 범례 추가 및 그래프 저장
        ax.legend(loc="upper right")
        plt.savefig(output_image_path, format="jpg", pil_kwargs={"quality": 85})
        plt.close()
        logger.info(f"퍼센테이지 기반 음높이 그래프가 저장되었습니다: {output_image_path}")

    def adjust_bottom_margin(self, fig, ax, times, f0_values):
        """
        TextGrid 주석을 위한 하단 여백을 자동으로 조정합니다.

        Parameters:
            fig: matplotlib Figure 객체
            ax: matplotlib Axes 객체
            times: 음높이 포인트 시간 리스트
            f0_values: 음높이 포인트 f0 값 리스트
        """
        # 텍스트 그리드가 있는 경우 텍스트를 추가하고 여백 측정
        if any(tier.name in ["word", "phoneme"] for tier in self.textgrid.tiers):
            sample_text = ax.text(0, 0, "샘플 텍스트", fontproperties=self.fontprop)
            renderer = fig.canvas.get_renderer()
            text_height = sample_text.get_window_extent(renderer=renderer).height / fig.dpi
            sample_text.remove()

            # 텍스트 높이를 기준으로 하단 여백을 동적으로 조정
            plt.subplots_adjust(bottom=text_height * 2.5)
        else:
            plt.subplots_adjust(bottom=0.1)  # 텍스트 그리드가 없는 경우 기본값으로 설정

    def get_text_height_adjustment(self, ax, scale=2.5):
        """
        그래프의 하단 여백을 텍스트 높이를 기준으로 조정할 때 사용할 값을 계산합니다.

        Parameters:
            ax: matplotlib Axes 객체
            scale: 텍스트 높이에 곱할 스케일 값 (기본값: 2.5)

        Returns:
            float: 텍스트 높이를 기반으로 한 여백 조정 값
        """
        sample_text = ax.text(0, 0, "샘플 텍스트", fontproperties=self.fontprop)
        renderer = ax.figure.canvas.get_renderer()
        text_height = sample_text.get_window_extent(renderer=renderer).height / ax.figure.dpi
        sample_text.remove()
        return text_height * scale

    def synthesize_spline_modified_wav(self, output_wav_path, spline_times, spline_f0_values):
        """
        삼차 스플라인으로 수정된 음높이를 기반으로 새로운 WAV 파일로 저장합니다.

        Parameters:
            output_wav_path (str): 저장할 WAV 파일 경로.
            spline_times (list): 스플라인 기반 시간 리스트.
            spline_f0_values (list): 스플라인 기반 F0 값 리스트.
        """
        manipulation = call(self.sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(pitch_tier, "Remove points between", 0, self.duration)

        # 스플라인 데이터를 Pitch Tier에 추가
        for time, f0 in zip(spline_times, spline_f0_values):
            call(pitch_tier, "Add point", time, f0)

        # 수정된 Pitch Tier를 Manipulation 객체에 통합
        call([pitch_tier, manipulation], "Replace pitch tier")

        # 변조된 음성을 재합성
        manipulated_sound = call(manipulation, "Get resynthesis (overlap-add)")
        manipulated_sound.save(output_wav_path, 'WAV')
        logger.info(f"스플라인 기반으로 수정된 음성을 {output_wav_path}에 저장했습니다.")

    def apply_cubic_spline(self, simplified_times, simplified_f0_values, num_points=100):
        """
        단순화된 F0 포인트를 대상으로 3차 스플라인을 적용하여 값을 반환합니다.

        Parameters:
            simplified_times (list): 단순화된 F0 포인트의 시간 리스트.
            simplified_f0_values (list): 단순화된 F0 포인트의 F0 값 리스트.
            num_points (int): 스플라인 윤곽을 계산할 샘플 포인트 수.

        Returns:
            tuple: (interpolated_times, interpolated_f0)
                - interpolated_times: 스플라인으로 계산된 시간 리스트.
                - interpolated_f0: 스플라인으로 계산된 F0 값 리스트.
        """
        # 3차 스플라인 계산
        spline = CubicSpline(simplified_times, simplified_f0_values, bc_type='natural')
        interpolated_times = np.linspace(min(simplified_times), max(simplified_times), num_points)
        interpolated_f0 = spline(interpolated_times)

        return interpolated_times, interpolated_f0

    def apply_momel_pitch_modulation(self, points_tier_name, modified_wav_path,
                                     minimalized_wav_path, minimalized_image_path,
                                     corrected_wav_path, corrected_image_path,
                                     spline_image_path, spline_wav_path,
                                     percentage_image_path):
        """
        Momel의 Points 티어를 기반으로 첫 번째 변조를 수행하고, 단순화된 음높이 포인트로 두 번째 변조 및 그래프를 생성합니다.

        Parameters:
            points_tier_name (str): 변조에 사용할 Momel의 Points 티어 이름.
            modified_wav_path (str): 첫 번째 변조된 음성의 출력 파일 경로.
            minimalized_wav_path (str): 단순화 후 두 번째 변조된 음성의 출력 파일 경로.
            minimalized_image_path (str): 단순화된 음높이 포인트 그래프의 출력 이미지 파일 경로.
            corrected_wav_path (str): Doubling/Halving 제거 후 음성의 출력 파일 경로.
            corrected_image_path (str): Doubling/Halving 제거 후 그래프의 출력 이미지 파일 경로.
        """
        points_tier = next((tier for tier in self.textgrid.tiers if tier.name == points_tier_name), None)
        if points_tier:
            # Momel Points 티어에서 시간과 음높이 값을 가져옴
            times, f0_values = self.get_momel_pitch_points(points_tier)

            # 첫 번째 변조된 음성 생성
            # self.synthesize_pitch_modified_wav(modified_wav_path, times, f0_values)

            # 기울기 기반 단순화 적용
            simplified_times, simplified_f0_values = self.simplify_pitch_points_by_slope(times, f0_values)

            # 단순화된 음높이 포인트 그래프 저장
            self.plot_simplified_pitch_contour(simplified_times, simplified_f0_values, minimalized_image_path)

            # 단순화된 음높이로 두 번째 변조된 음성 생성
            # self.synthesize_pitch_modified_wav(minimalized_wav_path, simplified_times, simplified_f0_values)

            # Doubling/Halving 현상 제거
            corrected_times, corrected_f0_values = self.remove_doubling_halving(simplified_times, simplified_f0_values)

            # Doubling/Halving 제거된 음높이 포인트 그래프 저장
            # self.plot_doubling_halving_corrected_pitch_contour(corrected_times, corrected_f0_values, corrected_image_path)

            # Doubling/Halving 제거 후 음성 파일 생성
            self.synthesize_pitch_modified_wav(corrected_wav_path, corrected_times, corrected_f0_values)

            # 최종 음높이 포인트 percentage로 저장
            self.add_percentage_points_tier(corrected_times, corrected_f0_values)
            
            # 백분율 그래프 산출
            self.plot_percentage_pitch_contour(corrected_times, corrected_f0_values, percentage_image_path, y_fixed_range=self.settings["fixed_y_range"])
            
            # Points 티어를 보정된 데이터로 업데이트
            points_tier.points.clear()
            for time, f0 in zip(corrected_times, corrected_f0_values):
                points_tier.add(time, f"{f0:.2f}")
                
            # 3차 스플라인 적용 및 결과 출력
            spline_times, spline_f0_values = self.apply_cubic_spline(corrected_times, corrected_f0_values)
            
            self.plot_spline_contour(spline_times, spline_f0_values, spline_image_path, corrected_times, corrected_f0_values, y_fixed_range=self.settings["fixed_y_range"])

            # 3차 스플라인 기반으로 WAV 파일 생성
            # self.synthesize_spline_modified_wav(spline_wav_path, spline_times, spline_f0_values)


    def remove_doubling_halving(self, times, f0_values, threshold_ratio=0.5):
        """
        Doubling 및 Halving 현상을 감지하고 해당 음높이 포인트를 제거합니다.

        Parameters:
            times (list): 각 음높이 값에 해당하는 시간 리스트.
            f0_values (list): 음높이 값 리스트.
            threshold_ratio (float): Doubling/Halving 감지 임계값 비율 (논문에서 권장하는 값은 0.5).

        Returns:
            corrected_times (list): Doubling/Halving이 제거된 시간 리스트.
            corrected_f0_values (list): Doubling/Halving이 제거된 음높이 값 리스트.
        """
        corrected_times = [times[0]]
        corrected_f0_values = [f0_values[0]]

        for i in range(1, len(f0_values)):
            previous_f0 = corrected_f0_values[-1]
            current_f0 = f0_values[i]

            # Doubling/Halving 검출: 이전 값과 현재 값의 비율이 threshold_ratio보다 큰지 확인
            if not (threshold_ratio <= current_f0 / previous_f0 <= (1 / threshold_ratio)):
                # Doubling/Halving으로 의심되면 현재 포인트를 제외
                continue

            # 정상적인 값이면 리스트에 추가
            corrected_times.append(times[i])
            corrected_f0_values.append(current_f0)

        return corrected_times, corrected_f0_values

    def run(self):
        """
        전체 전사 프로세스 실행
        """
        # 출력 파일 경로 설정
        output_pitch_contour = os.path.splitext(self.output_textgrid)[0] + "_pitch_contour.jpg"
        output_momel_pitch_contour = os.path.splitext(self.output_textgrid)[0] + "_momel_pitch_contour.jpg"
        output_momel_pitch_contour_minimalized = os.path.splitext(self.output_textgrid)[0] + "_momel_pitch_contour_minimalized.jpg"
        output_percentage_contour = os.path.splitext(self.output_textgrid)[0] + "_momel_pitch_percentage.jpg"
        modified_wav_path = os.path.splitext(self.output_textgrid)[0] + "_modified.wav"
        modified_minimalization_wav_path = os.path.splitext(self.output_textgrid)[0] + "_modified_minimalization.wav"
        corrected_wav_path = os.path.splitext(self.output_textgrid)[0] + "_corrected_doubling_halving.wav"
        corrected_image_path = os.path.splitext(self.output_textgrid)[0] + "_corrected_doubling_halving_contour.jpg"
        spline_image_path = os.path.splitext(self.output_textgrid)[0] + "_spline_contour.jpg"
        spline_wav_path = os.path.splitext(self.output_textgrid)[0] + "_spline_contour.wav"

        # 모든 출력 파일이 존재하는 경우, 건너뜁니다.
        if (os.path.exists(self.output_textgrid) and os.path.exists(output_pitch_contour)
            and os.path.exists(output_momel_pitch_contour) and os.path.exists(modified_wav_path)
            and os.path.exists(output_momel_pitch_contour_minimalized) and os.path.exists(modified_minimalization_wav_path)
            and os.path.exists(corrected_wav_path) and os.path.exists(corrected_image_path)):
            logger.info(f"모든 출력 파일이 이미 존재합니다. 건너뜁니다: {self.output_textgrid}")
            return

        try:
            # 기본 전사 및 TextGrid 생성
            self.perform_alignment()
            self.create_textgrid()
            self.run_momel_based_labels()
            self.add_tcog_tier()

            # 기본 음높이와 Momel 기반 그래프 생성
            pitch = self.extract_pitch(sex=self.sex)
            self.plot_pitch_and_textgrid(pitch)
            # self.plot_momel_pitch_points()
            
            

            # Momel의 Points 티어를 기반으로 변조 및 단순화 적용
            self.apply_momel_pitch_modulation(points_tier_name="Points",
                                            modified_wav_path=modified_wav_path,
                                            minimalized_wav_path=modified_minimalization_wav_path,
                                            minimalized_image_path=output_momel_pitch_contour_minimalized,
                                            corrected_wav_path=corrected_wav_path,
                                            corrected_image_path=corrected_image_path,
                                            spline_image_path=spline_image_path,
                                            spline_wav_path=spline_wav_path,
                                            percentage_image_path=output_percentage_contour)
            self.save_textgrid()
        except Exception as e:
            logger.error(f"억양 전사 중 오류 발생: {e}")
            logger.error(traceback.format_exc())



def process_files(tsv_file: str, output_dir: str, momel_path: str, stop_flag):
    """
    CSV, TSV 파일을 읽어 각 행의 WAV 파일과 전사를 처리하고 TextGrid를 생성
    """
    logger.info(f"CSV(TSV) 파일을 처리합니다: {tsv_file}")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)  # tqdm을 사용하기 위해 전체 rows를 리스트로 변환
            # logging_redirect_tqdm으로 tqdm 출력 연결
            with logging_redirect_tqdm():
                for row in tqdm(rows, desc="Processing WAV files", unit="file"):
                    if stop_flag and stop_flag.is_set():  # 작업 중지 플래그 확인
                        logger.info("작업이 중단되었습니다.")
                        return
                    wav_file = row.get("wav_filepath", "")
                    transcript = row.get("text", "")
                    sex = row.get("sex", "")
                    if not os.path.exists(wav_file):
                        logger.warning(f"WAV 파일이 존재하지 않습니다: {wav_file}")
                        continue

                    base_name = os.path.splitext(os.path.basename(wav_file))[0]
                    # output_textgrid = os.path.join(output_dir, f"{base_name}.TextGrid")
                    os.makedirs(f"{output_dir}/{base_name.split('.')[0]}", exist_ok=True)
                    output_textgrid = os.path.join(f"{output_dir}/{base_name.split('.')[0]}", f"{base_name}_{sex}.TextGrid")
                    # if 'SDRW2200000002.1.1.184' not in output_textgrid:
                    #     continue
                    transcriber = IntonationTranscriber(
                        wav_file=wav_file,
                        transcript=transcript,
                        sex=sex,
                        output_textgrid=output_textgrid,
                        momel_path=momel_path
                    )

                    logger.info(f"처리 중: {wav_file}")
                    transcriber.run()
    except:
        logger.error(f"파일을 처리하는 도중 에러가 발생했습니다.\n{traceback.format_exc()}")

    logger.info("모든 파일 처리가 완료되었습니다.")

if __name__ == '__main__':
    import argparse
    from threading import Event

    parser = argparse.ArgumentParser(description="억양 자동 전사 도구 (TSV 입력)")
    parser.add_argument("tsv_file", type=str, nargs='?', default="data/output.tsv",
                        help="입력 TSV 파일 경로 (wavfile_path와 text 컬럼 포함)")
    parser.add_argument("output_dir", type=str, nargs='?', default='out/outputs2',
                        help="출력 TextGrid 파일들이 저장될 디렉토리 경로")
    parser.add_argument("--momel_path", type=str, default="src/lib/momel/momel_linux",
                        help="Momel 실행 파일 경로")

    args = parser.parse_args()

    def create_symbolic_link(target, link_name):
        try:
            # 심볼릭 링크가 이미 존재하면 제거
            if os.path.islink(link_name) or os.path.exists(link_name):
                os.remove(link_name)
            # 심볼릭 링크 생성
            os.symlink(target, link_name)
            print(f"Symbolic link created: {link_name} -> {target}")
        except PermissionError:
            print("Permission denied. Trying with sudo...")
            # sudo를 사용해 심볼릭 링크 생성
            subprocess.run(["sudo", "ln", "-s", target, link_name], check=True)
            print(f"Symbolic link created with sudo: {link_name} -> {target}")
        except Exception as e:
            print(f"Error creating symbolic link: {e}")

    # 사용 예시
    target_path = "/data1/users/yugwon/SDRW2/"
    link_path = "out/outputs2"
    create_symbolic_link(target_path, link_path)
    
    # 중지 플래그 생성
    stop_flag = Event()

    process_files(
        tsv_file=args.tsv_file,
        output_dir=args.output_dir,
        momel_path=args.momel_path,
        stop_flag=stop_flag  # 중지 플래그 전달
    )