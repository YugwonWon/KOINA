import os
import struct
import traceback

from typing import List, Tuple
import matplotlib.pyplot as plt

import parselmouth
from parselmouth.praat import call

import logging
from utils.logger import main_logger

logger = main_logger.getChild('praat')

class Praat:
    def __init__(self, sound):
        self.sound = sound
        self.intensity = self.get_intensity()
    
    def get_pitch(self, 
                  time_step=0.02,
                  pitch_floor=75,
                  max_num_candi=15,
                  very_accurate=True,
                  silence_threshold=0.03,
                  voicing_threshold=0.45,
                  octave_cost=0.01,
                  octave_jump_cost=0.35,
                  voiced_unvoiced_cost=0.14,
                  pitch_ceiling=500,
                  very_accurate_str='yes',
                  sex='M'):
        pitch_floor = 75 if sex == 'M' else 100 # 남자면 75, 여자면 100
        pitch = call(self.sound, "To Pitch (cc)", time_step, pitch_floor, max_num_candi, very_accurate_str, silence_threshold, voicing_threshold, octave_cost, octave_jump_cost, voiced_unvoiced_cost, pitch_ceiling)
        # pitch = self.sound.to_pitch_ac(time_step, pitch_floor, max_num_candi, very_accurate, silence_threshold, voicing_threshold, octave_cost, octave_jump_cost, voiced_unvoiced_cost, pitch_ceiling) 
        return pitch
    
    def get_pitch_values(self, sex='M'):
        pitch = self.get_pitch(sex=sex)
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()
        return pitch_times, pitch_values

    def get_pitch_statistics(self, pitch_values):
        valid_pitch_values = [p for p in pitch_values if p > 0]  # 음이 아닌 값 필터링
        mean_pitch = sum(valid_pitch_values) / len(valid_pitch_values)
        stdev_pitch = (sum((x - mean_pitch) ** 2 for x in valid_pitch_values) / len(valid_pitch_values)) ** 0.5
        return mean_pitch, stdev_pitch
    
    def plot_pitch(self, filename, out_dir='out/jpg'):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pitch_times, pitch_values = self.get_pitch_values()

        # 0인 값을 제외하고 필터링
        filtered_times = [time for time, value in zip(pitch_times, pitch_values) if value > 0]
        filtered_values = [value for value in pitch_values if value > 0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_times, filtered_values, label='Pitch')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch (Hz)')
        plt.title('Pitch Over Time')
        plt.legend()
        plt.savefig(f'{out_dir}/{filename}.jpg')
    
    def get_speaking_time(self) -> float:
        """
        휴지를 제외한 발화 시간을 구한다
        :param sound: parselmouth로 생성한 sound 객체
        :return: speaking_time
        """
        threshold, threshold2, threshold3 = self.get_threshold(self.intensity, silence_db=-25)
        textgrid = self.get_textgrid(self.intensity, threshold3=threshold3, min_pause=0.3)
        silence_tier = self.get_silence_tier(textgrid)
        silence_table = self.get_silence_table(silence_tier)
        n_pauses = self.get_n_pauses(silence_table)
        
        speaking_time = 0
        for ipause in range(n_pauses):
            pause = ipause + 1
            begin_sound = call(silence_table, "Get value", pause, 1)
            end_sound = call(silence_table, "Get value", pause, 2)
            speaking_dur = end_sound - begin_sound
            speaking_time += speaking_dur
        return speaking_time
    
    def get_intensity(self, value=50) -> parselmouth.Intensity:
        """
        sound 객체로부터 intensity list를 계산한다.
        :param sound: signal, salplerate로부터 얻은 sound 객체
        :return: intensity
        """
        return self.sound.to_intensity(value)
    
    @staticmethod
    def get_threshold(intensity: parselmouth.Intensity, silence_db=-25) -> Tuple[float, float, float]:
        """
        intensity 임계값을 구한다.
        :param intensity: sound 객체의 intensity list
        :silence_db: 무음 감지를 위한 표준 설정
        :return: threshold, threshold2, threshold3
        """
        min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
        max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")
        # 포만트 분위수를 가져온다. 0~1 사이의 값을 가지며 기본 분포의 중앙값 추정치를 얻으려면 0.5를 지정한다. 
        max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

        threshold = max_99_intensity + silence_db
        threshold2 = max_intensity - max_99_intensity
        threshold3 = silence_db - threshold2
        if threshold < min_intensity:
            threshold = min_intensity
        return threshold, threshold2, threshold3

    @staticmethod
    def get_textgrid(intensity: parselmouth.Intensity, threshold3: float, min_pause=0.25) -> parselmouth.TextGrid:
        """
        sound의 무음 및 소리 간격이 표시되는 textgrid를 만든다.
        :param intensity: 강도
        :threshold3: 강도 임계값
        :min_pause: 최소 휴지(0.3초)
        :retrun: textgrid
        """
        textgrid = call(intensity, "To TextGrid (silences)", threshold3, min_pause, 0.1, "silent", "sounding")
        return textgrid

    @staticmethod
    def get_silence_tier(textgrid: parselmouth.TextGrid) -> parselmouth.Data:
        """
        praat의 textgrid tier 표시 기능을 이용하여 무음 구간을 표시한다.
        :param textgrid: textgrid 
        :return: praat "Extract tier" 함수 호출
        """
        return call(textgrid, "Extract tier", 1)

    @staticmethod
    def get_silence_table(silence_tier: parselmouth.Data) -> parselmouth.Data:
        """
        praat의 textgrid tier 표시 기능을 이용하여 무음 구간을 표시한다.
        :param silence_tier: silence_tier interval
        :return: praat "Down to TableOfReal", "sounding" 함수 호출
        """
        return call(silence_tier, "Down to TableOfReal", "sounding")

    @staticmethod
    def get_n_pauses(silence_table: parselmouth.Data) -> int:
        """
        sound에서 전체 pause의 개수를 구한다.
        :param silence_table: silence_table
        :retrun: praat "Get number of rows" 함수 호출 
        """
        return call(silence_table, "Get number of rows")

    @staticmethod
    def get_num_peaks(intensity: parselmouth.Intensity):
        """
        peak의 개수, 위치, 강도 매트릭스를 구한다.
        :param intensity: parselmouth sound 객체로부터 얻은 intensity
        :return num_peaks, time, sound_from_intensity_matrix
        """
        intensity_matrix = call(intensity, "Down to Matrix")
        sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
        point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
        num_peaks = call(point_process, "Get number of points")
        time = [call(point_process, "Get time from index", i + 1) for i in range(num_peaks)]
        
        return num_peaks, time, sound_from_intensity_matrix
    
    @staticmethod
    def get_time_peaks(num_peaks:int , time: List[float], sound_from_intensity_matrix: parselmouth.Sound, threshold: float):
        """
        시간 변화에 따른 peak의 배열을 구한다.
        :param num_peak: peak의 개수
        :time: 시간
        :sound_from_intensity_matrix: 강도 매트릭스
        :threshold: 강도 임계값
        :return: time_peaks, peak_count, intensities
        """
        time_peaks = []
        peak_count = 0
        intensities = []
        for i in range(num_peaks):
            value = call(sound_from_intensity_matrix, "Get value at time", time[i], "Cubic")
            if value > threshold:
                peak_count += 1
                intensities.append(value)
                time_peaks.append(time[i])
        return time_peaks, peak_count, intensities

    @staticmethod
    def get_valid_peak_count(time_peaks: List[float], peak_count: int, intensity: parselmouth.Intensity, intensities: List[float], min_dip = 1.5):
        """
        유효한 피크를 추출한다.
        :param time_peaks: peak의 시간
        :peak_count: peak의 개수
        :intensity: 강도
        :intensities: 강도 리스트
        :min_dip: 현재 intensity와 최소 intensity 차이 최소값
        :return: valid_peak_count, current_time, current_int, valid_time
        """
        valid_peak_count = 0
        current_time = time_peaks[0]
        current_int = intensities[0]
        valid_time = []
        for p in range(peak_count - 1):
            following = p + 1
            dip = call(intensity, "Get minimum", current_time, time_peaks[p + 1], "None")
            diff_int = abs(current_int - dip)
            if diff_int > min_dip:
                valid_peak_count += 1
                valid_time.append(time_peaks[p])  
            current_time = time_peaks[following]
            current_int = call(intensity, "Get value at time", time_peaks[following], "Cubic")

        return valid_peak_count, current_time, current_int, valid_time
    
    @staticmethod
    def decode_wav_stt(wave: bytes):
        """
        입력된 bytes 형태의 wave를 디코딩하여 음성분석에 필요한 signal, sample_rate, channel 정보를 찾는다.
        :param wave: bytes 형태의 wave 변수
        :retrun: signal, samplerate, channels
        """
        channel_cnt = 0
        sample_rate = 0
        byte_rate = 0
        offset = 0
        try:
            riff, size, fformat = struct.unpack('<4sI4s', wave[offset:offset + 12])
            if (riff != b'RIFF' or fformat != b'WAVE'):
                raise Exception(f'RIFF: {riff}, fformat: {fformat}')

            offset += 12

            # Read header
            chunk_header = wave[offset:offset + 8]
            offset += 8
            sub_chunk_id, sub_chunk_size = struct.unpack('<4sI', chunk_header)
            if (sub_chunk_id != b'fmt '):
                raise Exception(f'wave_header_44: {wave[:44]}')
            audio_format, channel_cnt, sample_rate, byte_rate, block_align, bps = struct.unpack('HHIIHH', wave[offset:offset + 16])
            offset += 16
                
            pcm_data = b''
            while (offset < size):
                sub_chunk2_id, sub_chunk2_size = struct.unpack('<4sI', wave[offset:offset + 8])
                offset += 8
                if (sub_chunk2_id == b'data'):
                    pcm_data = wave[offset:offset+sub_chunk2_size]
                offset += sub_chunk2_size
        except Exception as e:
            logger.error(f'Error decoding wav for stt: {e}')
            logger.error(f'Error decoding wav for stt: {e.__traceback__}')
            logger.error(f'{traceback.print_exc()}')
            logger.exception(e)
        # signal = (np.array(list(zip(*data))).astype(np.float32) / np.math.pow(2, bps - 1)).squeeze()
        logger.info('done decoding wav for stt')
        return pcm_data, sample_rate, channel_cnt, byte_rate    
    
