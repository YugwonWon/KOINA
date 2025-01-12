import os
import json
import shutil
import parselmouth
from parselmouth.praat import call
from textgrid import TextGrid, PointTier, Point

from preprocess.dataloader import TestDataLoader
# from preprocess.whisper_stt import WhisperSTT
from praat import Praat
from client.aligner import BaikalSTTClient

from utils.jamo import is_hangul, decompose_hangul, CHOSUNG_LIST, JUNGSEONG_LIST

import logging
from utils.logger import setup_logger

set_logger = setup_logger('out/logs/main.log')
logger = set_logger.getChild('transcriber')

class Transcriber():
    """
    음성에서 음절을 찾아 억양을 전사합니다.
    """

    def __init__(self, 
                 source_path: str, 
                 out_align_dir: str = 'out/align-out',
                 out_stt_dir: str = 'out/stt-out',
                 out_tg_dir: str = 'out/textgrid', 
                 csv_file: str = 'data/texts/sample.csv',
                 stt: bool = False,
                 align: bool = True):
        """
        Transcriber 객체를 초기화합니다.

        Args:
            source_path (str): 원본 파일이 저장된 경로.
            out_align_dir (str, optional): 정렬 결과 파일을 저장할 경로. 기본값은 'out/align-out'.
            out_stt_dir (str, optional): 음성 인식 결과 파일을 저장할 경로. 기본값은 'out/stt-out'.
            out_tg_dir (str, optional): TextGrid 파일을 저장할 경로. 기본값은 'out/textgrid'.
            csv_file (str, optional): CSV 파일 경로. 기본값은 'data/texts/sample.csv'.
            stt (bool, optional): 음성 인식 여부. 기본값은 False.
            align (bool, optional): 정렬 여부. 기본값은 True.
        """
        self.align = align
        self.stt = stt
        self.out_align_dir = out_align_dir
        self.out_stt_dir = out_stt_dir
        self.align_client = None
        self.stt_client = None
        if align:
            self.align_client = BaikalSTTClient(source_path, out_align_dir, csv_file=csv_file) # aligner 초기화
        elif stt:
            self.stt_client = WhisperSTT(source_path, out_stt_dir)
        self.out_tg_dir = out_tg_dir
        
        if not os.path.exists(self.out_tg_dir):
            os.makedirs(self.out_tg_dir)
        
        self.data_loader = TestDataLoader(source_path)
        self.speaker_avg_pitch = self.data_loader.speaker_avg_pitch
        self.base_name = ''

    def transcribe(self):
        """
        주어진 소스 경로의 파일들을 분석하여 억양을 전사합니다.
        """
        try:
            if self.align: # aligner 요청
                self.align_client.align_files()
            elif self.stt: # stt 요청
                self.stt_client.stt_files()
            data = self.data_loader.load_data()
            for wav_key, wav_values in data.items():
                self.base_name = os.path.basename(wav_key).replace('.wav', '')
                json_path = os.path.join(self.out_align_dir, f"{self.base_name}.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        transcription_data = json.load(f)
                    self.analyze_and_update(wav_values['sound'], wav_values['textgrid'], transcription_data, self.base_name)
        except Exception as e:
            logger.error(f"Error in transcribe: {str(e)}")

    def analyze_and_update(self, sound, textgrid, transcription_data, base_name):
        """
        주어진 sound 파일과 TextGrid를 분석하여 업데이트된 TextGrid를 저장합니다.

        Args:
            sound (parselmouth.Sound): 분석할 소리 파일.
            textgrid (TextGrid): 분석할 TextGrid 파일.
            transcription_data (dict): 전사 데이터.
            base_name (str): 파일의 기본 이름.
        """
        try:
            praat = Praat(sound)
            praat.plot_pitch(base_name)
            speaker_id = base_name.split('_')[-1]
            speaker_sex = speaker_id[0]
            pitch_times, pitch_values = praat.get_pitch_values(sex=speaker_sex)
            mean_pitch, stdev_pitch = praat.get_pitch_statistics(pitch_values)
            if self.align:
                ap_tier, ap_medial_tier = self.process_ap_tiers_using_align(praat, pitch_times, pitch_values, textgrid, transcription_data, speaker_id)
            elif self.stt:
                ap_tier, ap_medial_tier = self.process_ap_tiers_using_stt(praat, pitch_times, pitch_values, textgrid, transcription_data, speaker_id)
            new_textgrid = self.create_new_textgrid_with_updated_tiers(textgrid, ap_tier, ap_medial_tier)
            output_path = os.path.join(self.out_tg_dir, base_name + '.TextGrid')
            new_textgrid.write(output_path)
            
            # Copy the .wav file to the new directory
            wav_source_path = os.path.join(self.data_loader.folder_path, base_name + '.wav')
            wav_dest_path = os.path.join(self.out_tg_dir, base_name + '.wav')
            shutil.copyfile(wav_source_path, wav_dest_path)
        except Exception as e:
            logger.error(f"Error in analyze_and_update for {base_name}: {str(e)}")

    def create_new_textgrid_with_updated_tiers(self, textgrid: TextGrid, ap_tier: PointTier, ap_medial_tier: PointTier) -> TextGrid:
        """
        주어진 TextGrid를 기반으로 업데이트된 AP 및 AP-medial 티어를 포함하는 새로운 TextGrid를 생성합니다.

        Args:
            textgrid (TextGrid): 원본 TextGrid.
            ap_tier (PointTier): 업데이트된 AP 티어.
            ap_medial_tier (PointTier): 업데이트된 AP-medial 티어.

        Returns:
            TextGrid: 새로운 TextGrid 객체.
        """
        try:
            new_textgrid = TextGrid()
            new_textgrid.minTime = textgrid.minTime
            new_textgrid.maxTime = textgrid.maxTime

            for tier in textgrid.tiers:
                if tier.name not in ["AP", "AP-medial"]:
                    new_textgrid.append(tier)
            
            new_textgrid.append(ap_tier)
            if ap_medial_tier is not None:
                new_textgrid.append(ap_medial_tier)
            else:
                for tier in textgrid.tiers:
                    if tier.name == "AP-medial":
                        new_textgrid.append(tier)
            
            return new_textgrid
        except Exception as e:
            logger.error(f"Error in create_new_textgrid_with_updated_tiers: {str(e)}")
            raise

    def process_ip_tier(self, praat: Praat, pitch: list, textgrid: TextGrid) -> PointTier:
        """
        주어진 pitch와 TextGrid를 사용하여 IP 티어를 처리합니다.

        Args:
            praat (Praat): Praat 객체.
            pitch (list): 피치 값 리스트.
            textgrid (TextGrid): TextGrid 객체.

        Returns:
            PointTier: 처리된 IP 티어.
        """
        try:
            point_tier = PointTier(name="Int")
            if len(pitch) > 0:
                end_time = len(pitch) * 0.01
                point_tier.add(end_time, "H%")
            return point_tier
        except Exception as e:
            logger.error(f"Error in process_ip_tier: {str(e)}")
            raise

    def process_ap_tiers_using_align(self, praat: Praat, pitch_times: list, pitch_values: list, textgrid: TextGrid, transcription_data: dict, speaker_id: str) -> tuple:
        """
        정렬 데이터를 사용하여 AP 및 AP-medial 티어를 처리합니다.

        Args:
            praat (Praat): Praat 객체.
            pitch_times (list): 피치 시간 리스트.
            pitch_values (list): 피치 값 리스트.
            textgrid (TextGrid): TextGrid 객체.
            transcription_data (dict): 전사 데이터.
            speaker_id (str): 화자 ID.

        Returns:
            tuple: AP 티어와 AP-medial 티어.
        """
        try:
            ap_tier = PointTier(name="AP")
            ap_medial_tier = PointTier(name="AP-medial")

            speaker_avg = self.speaker_avg_pitch[speaker_id]
            prev_mean_pitch = None
            prev_tagged_h = False

            fricatives_and_tense = {'ㅊ', 'ㅉ', 'ㅌ', 'ㄸ', 'ㅍ', 'ㅃ', 'ㅅ', 'ㅆ'}
            num_syllables = len(transcription_data['transcript'].replace(' ', ''))

            syllable_count = 0
            space_count = 0
            speech_first = False

            ori_chars = transcription_data['chars']
            phonemes = transcription_data['phonemes']

            if len(ori_chars) == 0:
                return ap_tier, None
            
            # 마지막 음절은 억양구 Tier이기 때문에 여기서 레이블링하지 않는다.
            if len(ori_chars) > 1 and (not is_hangul(ori_chars[-1]['text']) or not ori_chars[-1].get('start')):
                chars = ori_chars[:-2]
            elif len(ori_chars) > 1 and is_hangul(ori_chars[-1]['text']) and ori_chars[-1].get('start'):
                chars = ori_chars[:-1]

            for char_idx, char_info in enumerate(chars):
                if 'start' in char_info and 'end' in char_info:
                    char_start = char_info['start']
                    char_end = char_info['end']
                    char_text = char_info.get('text', '')

                    if char_text == ' ':
                        space_count += 1
                        syllable_count = 0
                        continue
                    
                    # 어절의 음절 수를 계산
                    current_syllable_count = 0
                    for temp_char_info in chars[char_idx:]:
                        if temp_char_info.get('text') == ' ':
                            break
                        current_syllable_count += 1
                    
                    is_last_word = False
                    # 마지막 어절은 억양구 음절이 생략된 것이므로 음절 수 +1로 계산
                    if char_idx + current_syllable_count == len(chars):
                        current_syllable_count += 1
                        is_last_word = True

                    chosung_index, jungseong_index, jongseong_index = decompose_hangul(char_text)
                    chosung = CHOSUNG_LIST[chosung_index] if chosung_index is not None else ''
                    jungseong = JUNGSEONG_LIST[jungseong_index] if jungseong_index is not None else ''

                    # Find the corresponding vowel (jungseong) start time
                    phoneme_start = None

                    for i, phoneme in enumerate(phonemes):
                        if phoneme['text'] in jungseong and phoneme['start'] >= char_start and phoneme['end'] <= char_end:
                            phoneme_start = (phoneme['start'] + phoneme['end']) / 2
                            phonemes = phonemes[i+1:]  # Remove processed phonemes up to this point
                            break

                    if not phoneme_start:
                        phoneme_start = (char_start + char_end) / 2

                    surrounding_pitch_values = [p for t, p in zip(pitch_times, pitch_values)
                                                if t >= phoneme_start - 0.1 and t <= phoneme_start + 0.1 and p > 0]
                    
                    if surrounding_pitch_values and any(surrounding_pitch_values):
                        current_mean_pitch = sum(surrounding_pitch_values) / len(surrounding_pitch_values)

                        if prev_mean_pitch is not None:
                            if current_mean_pitch > prev_mean_pitch:
                                if syllable_count == 0:
                                    ap_tier.add(phoneme_start, "H")
                                    prev_tagged_h = True
                                else:
                                    # 4음절인 경우 2번째 음절은 H로 레이블링하되, 이전 피치평균보다 낮으면 L를 적용
                                    if (current_syllable_count == 4 or is_last_word) and syllable_count == 1:
                                        if current_mean_pitch < prev_mean_pitch:
                                            ap_medial_tier.add(phoneme_start, "H(L)")
                                        else:
                                            ap_medial_tier.add(phoneme_start, "H")
                                    else:
                                        ap_medial_tier.add(phoneme_start, "H")
                                    prev_tagged_h = True
                            else:
                                if syllable_count == 0:
                                    ap_tier.add(phoneme_start, "L")
                                    prev_tagged_h = False
                                else:
                                    ap_medial_tier.add(phoneme_start, "L")
                                    prev_tagged_h = False
                        else: 
                            if not speech_first:
                                speech_first = True
                                if chosung in fricatives_and_tense:
                                    ap_tier.add(phoneme_start, "H")
                                else:
                                    if current_mean_pitch > speaker_avg:
                                        ap_tier.add(phoneme_start, "H")
                                    else:
                                        ap_tier.add(phoneme_start, "L")
                                        prev_tagged_h = False
                            else: 
                                if chosung in fricatives_and_tense:
                                    if syllable_count != 0:
                                        ap_medial_tier.add(phoneme_start, 'H')
                                    else:
                                        ap_tier.add(phoneme_start, "H")
                                    prev_tagged_h = True
                                else:
                                    if prev_tagged_h and syllable_count == 1:
                                        ap_medial_tier.add(phoneme_start, "L")
                                        prev_tagged_h = False
                                    else:
                                        ap_medial_tier.add(phoneme_start, "H")
                                        prev_tagged_h = True

                        prev_mean_pitch = current_mean_pitch
                    else:
                        if not speech_first:
                            speech_first = True
                        if chosung in fricatives_and_tense:
                            if syllable_count == 0:
                                ap_tier.add(char_end, "H")
                                prev_tagged_h = True
                            else:
                                ap_medial_tier.add(char_end, "H")
                                prev_tagged_h = True
                        else:
                            if syllable_count == 0:
                                ap_tier.add(char_end, "L")
                                prev_tagged_h = False
                            else:
                                ap_medial_tier.add(char_end, "L")
                                prev_tagged_h = False

                    syllable_count += 1

            if num_syllables <= 2:
                return ap_tier, None

            return ap_tier, ap_medial_tier
        except Exception as e:
            logger.error(f"Error in process_ap_tiers_using_align: {str(e)}")
            raise
        
    def process_ap_tiers_using_stt(self, praat: Praat, pitch_times: list, pitch_values: list, textgrid: TextGrid, transcription_data: dict, speaker_id: str) -> tuple:
        """
        STT 데이터를 사용하여 AP 및 AP-medial 티어를 처리합니다.

        Args:
            praat (Praat): Praat 객체.
            pitch_times (list): 피치 시간 리스트.
            pitch_values (list): 피치 값 리스트.
            textgrid (TextGrid): TextGrid 객체.
            transcription_data (dict): 전사 데이터.
            speaker_id (str): 화자 ID.

        Returns:
            tuple: AP 티어와 AP-medial 티어.
        """
        try:
            ap_tier = PointTier(name="AP")
            ap_medial_tier = PointTier(name="AP-medial")

            speaker_avg = self.speaker_avg_pitch[speaker_id]
            prev_mean_pitch = None
            prev_tagged_h = False

            fricatives_and_tense = {'ㅊ', 'ㅉ', 'ㅌ', 'ㄸ', 'ㅍ', 'ㅃ', 'ㅅ', 'ㅆ'}
            num_syllables = sum(len(segment['text'].strip()) for segment in transcription_data['segments'])

            syllable_count = 0
            space_count = 0
            speech_first = False

            for segment in transcription_data['segments']:
                chars = segment['chars']
                if len(chars) == 0:
                    return ap_tier, None
                
                if len(chars) > 1 and (not is_hangul(chars[-1]['char']) or not chars[-1].get('start')):
                    chars = chars[:-2]
                elif len(chars) > 1 and is_hangul(chars[-1]['char']) and chars[-1].get('start'):
                    chars = chars[:-1]

                for char_idx, char_info in enumerate(chars):
                    if 'start' in char_info and 'end' in char_info:
                        char_start = char_info['start']
                        char_end = char_info['end']
                        char_text = char_info.get('char', '')

                        if char_text == ' ':
                            space_count += 1
                            syllable_count = 0
                            continue

                        chosung_index, _, _ = decompose_hangul(char_text)
                        chosung = CHOSUNG_LIST[chosung_index] if chosung_index is not None else ''

                        surrounding_pitch_values = [p for t, p in zip(pitch_times, pitch_values)
                                                    if t >= char_start - 0.05 and t <= char_start + 0.05 and p > 0]

                        if surrounding_pitch_values and any(surrounding_pitch_values):
                            current_mean_pitch = sum(surrounding_pitch_values) / len(surrounding_pitch_values)

                            if prev_mean_pitch is not None:
                                if current_mean_pitch > prev_mean_pitch:
                                    if syllable_count == 0:
                                        ap_tier.add(char_start, "H")
                                        prev_tagged_h = True
                                    else:
                                        ap_medial_tier.add(char_start, "H")
                                        prev_tagged_h = True
                                else:
                                    if syllable_count == 0:
                                        ap_tier.add(char_start, "L")
                                        prev_tagged_h = False
                                    else:
                                        ap_medial_tier.add(char_start, "L")
                                        prev_tagged_h = False
                            else: 
                                if not speech_first:
                                    speech_first = True
                                    if chosung in fricatives_and_tense:
                                        ap_tier.add(char_start, "H")
                                    else:
                                        if current_mean_pitch > speaker_avg:
                                            ap_tier.add(char_start, "H")
                                        else:
                                            ap_tier.add(char_start, "L")
                                            prev_tagged_h = False
                                else: 
                                    if chosung in fricatives_and_tense:
                                        if syllable_count != 0:
                                            ap_medial_tier.add(char_start, 'H')
                                        else:
                                            ap_tier.add(char_start, "H")
                                        prev_tagged_h = True
                                    else:
                                        if prev_tagged_h and syllable_count == 1:
                                            ap_medial_tier.add(char_start, "L")
                                            prev_tagged_h = False
                                        else:
                                            ap_medial_tier.add(char_start, "H")
                                            prev_tagged_h = True

                            prev_mean_pitch = current_mean_pitch
                        else:
                            if not speech_first:
                                speech_first = True
                            if chosung in fricatives_and_tense:
                                if syllable_count == 0:
                                    ap_tier.add(char_end, "H")
                                    prev_tagged_h = True
                                else:
                                    ap_medial_tier.add(char_end, "H")
                                    prev_tagged_h = True
                            else:
                                if syllable_count == 0:
                                    ap_tier.add(char_end, "L")
                                    prev_tagged_h = False
                                else:
                                    ap_medial_tier.add(char_end, "L")
                                    prev_tagged_h = False

                        syllable_count += 1

            if num_syllables <= 2:
                return ap_tier, None

            return ap_tier, ap_medial_tier
        except Exception as e:
            logger.error(f"Error in process_ap_tiers: {str(e)}")
            raise

if __name__ == "__main__":
    source_path = "data/sample"
    transcriber = Transcriber(source_path, 
                              out_align_dir='out/align-out', 
                              out_tg_dir='out/textgrid',
                              csv_file='data/texts/sample.csv',
                              align=True)
    transcriber.transcribe()
