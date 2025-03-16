import os
import csv
import glob
import json
import traceback
import soundfile as sf

import grpc
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from lib.baikal.speech import recognition_config_pb2 as config_pb
from lib.baikal.speech import stt_service_pb2_grpc as service_grpcpb
from lib.baikal.speech import forced_align_service_pb2 as align_service_pb
from lib.baikal.speech import forced_align_service_pb2_grpc as align_service_grpcpb

from utils.logger import main_logger

logger = main_logger.getChild('aligner')


MAX_MESSAGE_LENGTH = 500*1024*1024

class BaikalSTTClient:
    """
    텍스트와 음절의 강제 정렬기능을 가진 BaikalSTT Aligner
    텍스트와 음성이 준비되어 있어야 합니다.
    """
    stub = None
    
    def __init__(self, remote='ml-service-vm.baikal.ai.:9082'):
        if remote.endswith("443"):
            channel = grpc.secure_channel( # insecure_channel
                remote,
                grpc.ssl_channel_credentials(),
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ])
        else:
            channel = grpc.insecure_channel( # insecure_channel
                remote,
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ])

        self.stt_stub = service_grpcpb.RecognizeServiceStub(channel)
        self.align_stub = align_service_grpcpb.ForcedAlignServiceStub(channel)

    def align(self, wav_file:str, transcript:str):
        try:
            req = self.generate_align_request(wav_file, transcript)
            ret = self.align_stub.ForcedAlign(req)
        except Exception as e:
            logger.error(f'Error request align: {e}')
            logger.error(f'{traceback.print_exc()}')
        return MessageToDict(ret)
    
    def generate_align_request(self, wav_file:str, transcript:str):
        try:
            wav_bytes = self.read_wav_file_as_bytes(wav_file)
            if wav_bytes is None:
                raise ValueError(f"Failed to read wav file: {wav_file}")

            # Read the wav file with soundfile
            pcm_data, sample_rate = sf.read(wav_file, dtype='int16')
            channel_count = pcm_data.shape[1] if pcm_data.ndim > 1 else 1

            # Flatten pcm_data to a 1D array if it's multi-channel
            if channel_count > 1:
                pcm_data = pcm_data.flatten()

            pcm_data_bytes = pcm_data.tobytes()

            config = config_pb.RecognitionConfig(encoding=config_pb.AudioEncoding.LINEAR16,
                                                sample_rate=sample_rate,
                                                channel_count=channel_count,
                                                enable_confidence=True,
                                                enable_word_time_offsets=True,
                                                enable_char_time_offsets=True,
                                                enable_phoneme_time_offsets=True)
            req = align_service_pb.ForcedAlignRequest(wave=pcm_data_bytes, transcript=transcript, config=config)
            return req
        except Exception as e:
            logger.error(f'Error generate_align: {e}')
            logger.error(traceback.print_exc())
            
    def read_wav_file_as_bytes(self, wav_file_path):
        """
        Read a wav file and return its content as bytes.

        Parameters:
        wav_file_path (str): The path to the wav file.

        Returns:
        bytes: The content of the wav file as bytes.
        """
        try:
            with open(wav_file_path, 'rb') as wav_file:
                wav_bytes = wav_file.read()
            return wav_bytes
        except Exception as e:
            logger.error(f"Error reading wav file {wav_file_path}: {str(e)}")
            return None
        
    def align_files(self):
        wave_files = glob.glob(os.path.join(self.input_folder, '*.wav'))
        wave_files.sort()
        for wave_file in tqdm(wave_files, desc='align files'):
            try:
                base_name = os.path.splitext(os.path.basename(wave_file))[0]
                json_path = os.path.join(self.out_align_dir, f"{base_name}.json")

                if os.path.exists(json_path):
                    print(f"Skipping {wave_file}, already aligned.")
                    continue

                # Retrieve transcript from dictionary
                base_label = "_".join(base_name.split('_')[:5])
                transcript = self.transcript_dict.get(base_label, "")
                if not transcript:
                    logger.warning(f"No transcript found for {base_name}")
                    continue

                self.align_and_save_json(wave_file, transcript)
                logger.info(f"Transcription completed: {wave_file}")
            except Exception as e:
                logger.error(f"Error transcribing {wave_file}: {e}")
                logger.error(traceback.format_exc())
                
    def align_and_save_json(self, wave_file, transcript):
        res = self.align(wav_file=wave_file, transcript=transcript)
        
        # JSON으로 저장
        base_name = os.path.splitext(os.path.basename(wave_file))[0]
        json_path = os.path.join(self.out_align_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
            
    @staticmethod
    def load_transcript_dict(csv_file):
        transcript_dict = {}
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript_dict[row['label']] = row['meaning']
        return transcript_dict
        


if __name__ in '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_file", nargs='?', default='data/sample/sample_sound.wav', help="wav file", type=str)
    parser.add_argument("-host", dest="host", help="remote server host",
                        default="localhost",
                        type=str)
    parser.add_argument("-p", dest="port", help="remote server port",
                        default=9080,
                        type=int)
    parser.set_defaults(wav_file='data/sample/sample_sound.wav')
    args = parser.parse_args()
        
    transcript = "테스트 텍스트"
    client = BaikalSTTClient(f'{args.host}:{args.port}')

    res = client.align(wav_file=args.wav_file, transcript=transcript)
    res_dict = MessageToDict(res)
    print(res_dict)
    
    
