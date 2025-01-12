import os
import gc 
import glob
import json
import torch
import whisperx

from tqdm import tqdm

class WhisperSTT:
    """
    음절을 찾는 용도의 Whisper STT Aligner
    """
    def __init__(self, input_folder, out_stt_dir='out/stt-out'):
        self.input_folder = input_folder
        self.out_stt_dir = out_stt_dir
        if not os.path.exists(self.out_stt_dir):
            os.makedirs(self.out_stt_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)

    def stt_files(self):
        wave_files = glob.glob(os.path.join(self.input_folder, '*.wav'))
        wave_files.sort()
        for wave_file in tqdm(wave_files, desc='Transcribing files'):
            try:
                base_name = os.path.splitext(os.path.basename(wave_file))[0]
                json_path = os.path.join(self.out_stt_dir, f"{base_name}.json")

                if os.path.exists(json_path):
                    print(f"Skipping {wave_file}, already transcribed.")
                    continue

                self.transcribe_and_save_json(wave_file)
                print(f"Transcription completed: {wave_file}")
            except Exception as e:
                print(f"Error transcribing {wave_file}: {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    def transcribe_and_save_json(self, wave_file, is_text=False):
        audio = whisperx.load_audio(wave_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size, language='ko')
        print(result["segments"])  # before alignment

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code='ko', device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=True)
        print(result["segments"])  # after alignment

        # JSON으로 저장
        base_name = os.path.splitext(os.path.basename(wave_file))[0]
        json_path = os.path.join(self.out_stt_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_folder = 'data/sample'  # 입력 폴더 경로
    out_stt_dir = 'out/stt-out'  # 출력 폴더 경로
    transcriber = WhisperSTT(input_folder, out_stt_dir)
    transcriber.stt_files()
