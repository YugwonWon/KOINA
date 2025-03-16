import os
import json
import logging
import time
from threading import Thread, Event
import gradio as gr
from transcribe.transcriber import process_files

# 상수 정의
LOG_FILE_PATH = "out/logs/main.log"  # 로그 파일 경로
CONFIG_FILE = "out/config.json"  # Config 파일 경로
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# 로거 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=LOG_FILE_PATH, filemode='a')
logger = logging.getLogger("front")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

class TranscriptionRunner:
    """억양 전사 프론트 메인 객체"""

    def __init__(self):
        self.thread = None
        self.running = False
        self.stop_flag = Event()
        self.log_lines = []

    def start_transcription(self, tsv_file, output_dir, momel_path="src/lib/momel/momel_linux"):
        if not os.path.exists(tsv_file):
            return "TSV 파일이 존재하지 않습니다."
        if self.thread and self.thread.is_alive():
            return "이미 작업이 진행 중입니다. 잠시만 기다려주세요."
        
        self.stop_flag.clear()  # 작업 중단 플래그 초기화

        def run():
            try:
                logger.info("작업 시작!")
                process_files(tsv_file, output_dir, momel_path, self.stop_flag)
                logger.info("작업 완료!")
            except Exception as e:
                logger.error(f"오류 발생: {e}")
            finally:
                self.running = False

        self.running = True
        self.thread = Thread(target=run)
        self.thread.start()
        return "작업이 시작되었습니다."

    def stop_transcription(self):
        if not self.running:
            return "진행 중인 작업이 없습니다."
        self.stop_flag.set()
        logger.info("작업 중단 요청이 접수되었습니다.")
        return "작업 중단 요청이 접수되었습니다."

    def start_log_stream(self):
        """로그 파일 실시간 스트리밍"""
        if self.running:
            logger.info("이미 로그 스트리밍이 실행 중입니다.")
            return
        self.running = True
        logger.info("로그 스트리밍을 시작합니다.")
        def stream_logs():
            try:
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as log_file:
                    log_file.seek(0, os.SEEK_END)  # 파일 끝으로 이동
                    while self.running:
                        line = log_file.readline()
                        if line:
                            self.log_lines.append(line.strip())
                            if len(self.log_lines) > 100:
                                self.log_lines.pop(0)
                        else:
                            time.sleep(0.1)
            except Exception as e:
                logger.error(f"로그 스트리밍 중 오류 발생: {e}")

        log_thread = Thread(target=stream_logs, daemon=True)
        log_thread.start()

    def stop_log_stream(self):
        """로그 읽기 중단"""
        self.running = False

    def get_logs(self):
        """실시간 로그 반환"""
        filtered_lines = []
        for line in self.log_lines[-20:]:
            if "HTTP Request" in line:
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines) if filtered_lines else "로그가 없습니다."


# Config 관련 함수
def save_config(config):
    """Config 파일 저장"""
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    logger.info(f"설정이 {CONFIG_FILE}에 저장되었습니다.")

def load_config():
    """Config 파일 로드"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("Config 파일이 없습니다. 기본값을 사용합니다.")
    return {}

def toggle_gender_range(use_gender_range_val):
    if use_gender_range_val:
        # True이면 visible = True
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    else:
        # False이면 visible = False
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    transcription_runner = TranscriptionRunner()
    transcription_runner.start_log_stream()

    # 기본값 정의
    default_config = {
        "output_dir": "out/outputs",
        "min_pitch": 75,
        "min_pitch_male": 75,
        "min_pitch_female": 100,
        "max_pitch": 600,
        "max_pitch_male": 500,
        "max_pitch_female": 600,
        "time_step": 0.01,
        "silence_threshold": 0.03,
        "voicing_threshold": 0.5,
        "octave_cost": 0.05,
        "octave_jump_cost": 0.5,
        "voice_unvoiced_cost": 0.2,
        "number_of_candidates": 15,
        "very_accurate": 1,
        "show_spline": False,
        "fixed_y_range": 600
    }

    with gr.Blocks(css=".custom-box {margin-top: 20px;}") as main:
        # 프로그램 제목과 설명
        with gr.Column():
            gr.Label(value="✨ 한국어 억양 주석기 클라이언트✨", label="")
            gr.Textbox(
                value=(
                    "📂 이 프로그램은 음성 파일 경로와 음성 전사 텍스트 칼럼으로 구성된 CSV(TSV)파일을 입력으로 사용하여"
                    "발화에서 억양을 분석하고 자동으로 전사합니다.\n\n"
                    "👉 사용 방법:\n"
                    "1️⃣ CSV(TSV) 파일을 선택하세요.(칼럼 구성 예: wav_filename(*.wav), sex(M,F), text)\n"
                    "2️⃣ 매개변수(parameter) 값을 설정하고 작업 시작 버튼을 누르세요.\n"
                    "3️⃣ 작업을 중단하려면 중단 버튼을 누르세요. 재시작시 해당 음성의 모든 산출물이 존재한다면 건너뜁니다."
                ),
                lines=6,
                interactive=False,
                elem_classes="custom-box",
                label="Readme"
            )

        # TSV 파일 선택
        tsv_file = gr.File(label="📁 CSV(TSV) 파일 선택", type="filepath")
        
        # Pitch Parameters
        gr.Markdown("### 🎛️ Pitch Parameters")  # 섹션 제목 추가

        with gr.Row():
            # 좌측 4개 배치
            with gr.Column():
                min_pitch = gr.Textbox(label="Min Pitch", placeholder="기본값: 75")
                time_step = gr.Textbox(label="Pitch Step", placeholder="기본값: 0.01")
                silence_threshold = gr.Textbox(label="Silence Threshold", placeholder="기본값: 0.03")
                octave_cost = gr.Textbox(label="Octave Cost", placeholder="기본값: 0.05")
                very_accurate = gr.Textbox(label="Very Accurate", placeholder="기본값: 0")

            # 우측 5개 배치
            with gr.Column():
                max_pitch = gr.Textbox(label="Max Pitch", placeholder="기본값: 600")
                number_of_candidates = gr.Textbox(label="Number of Candidates", placeholder="기본값: 15")
                voicing_threshold = gr.Textbox(label="Voicing Threshold", placeholder="기본값: 0.5")
                octave_jump_cost = gr.Textbox(label="Octave Jump Cost", placeholder="기본값: 0.5")
                voice_unvoiced_cost = gr.Textbox(label="Voice Unvoiced Cost", placeholder="기본값: 0.2")

        use_gender_range = gr.Checkbox(
            label="성별에 따라 다른 Pitch 범위 적용",
            value=False
        )

        with gr.Row():
            with gr.Column():
                M_min = gr.Textbox(label="남성 Min Pitch", placeholder="기본값: 75", visible=False, interactive=True)
                M_max = gr.Textbox(label="남성 Max Pitch", placeholder="기본값: 500", visible=False, interactive=True)

            with gr.Column():
                F_min = gr.Textbox(label="여성 Min Pitch", placeholder="기본값: 100", visible=False, interactive=True)
                F_max = gr.Textbox(label="여성 Max Pitch", placeholder="기본값: 600", visible=False, interactive=True)
        use_gender_range.change(
            fn=toggle_gender_range,
            inputs=[use_gender_range],
            outputs=[M_min, M_max, F_min, F_max],
        )

        # 스플라인 옵션 선택
        gr.Markdown("### 📐 Spline Option")
        show_spline = gr.Checkbox(label="스플라인 윤곽 그래프 출력", value=default_config["show_spline"])
        
        # 정규화 옵션 선택
        gr.Markdown("### 📐 Normalize Option")
        fixed_y_range = gr.Textbox(label="Fixed Y axis Range", placeholder="기본값: 600")

        # 버튼 및 상태 출력
        start_button = gr.Button("작업 시작")
        stop_button = gr.Button("작업 중단", visible=False)
        log_output = gr.Textbox(label="🖥️ 상태/로그", lines=5, interactive=False)

        # 입력값 검증 및 변환 함수
        def validate_and_convert(value, default, value_type, option):
            try:
                return value_type(value) if value else default
            except ValueError:
                raise ValueError(f"숫자 값을 입력해야 합니다. {option} 기본값: {default}")

        # 작업 시작 함수
        def start_transcription(tsv_file, min_pitch, max_pitch, time_step,
                        silence_threshold, voicing_threshold, octave_cost, octave_jump_cost, voice_unvoiced_cost,
                        number_of_candidates, very_accurate,
                        show_spline, fixed_y_range,
                        use_gender_range_val,
                        M_min_val, M_max_val, F_min_val, F_max_val):
            try:
                if not tsv_file:
                    return gr.update(), gr.update(), "", "❌ 오류: TSV/CSV 파일이 선택되지 않았습니다."
                # 값 검증 및 변환
                config = {
                    "tsv_file": tsv_file,
                    "output_dir": default_config["output_dir"],
                    "min_pitch": validate_and_convert(min_pitch, default_config["min_pitch"], float, "Min Pitch"),
                    "max_pitch": validate_and_convert(max_pitch, default_config["max_pitch"], float, "Max Pitch"),
                    "time_step": validate_and_convert(time_step, default_config["time_step"], float, "Pitch Step"),
                    "number_of_candidates": validate_and_convert(number_of_candidates, default_config["number_of_candidates"], float, "Number of Candidates"),
                    "silence_threshold": validate_and_convert(silence_threshold, default_config["silence_threshold"], float, "Silence Threshold"),
                    "voicing_threshold": validate_and_convert(voicing_threshold, default_config["voicing_threshold"], float, "Voicing Threshold"),
                    "octave_cost": validate_and_convert(octave_cost, default_config["octave_cost"], float, "Octave Cost"),
                    "octave_jump_cost": validate_and_convert(octave_jump_cost, default_config["octave_jump_cost"], float, "Octave Jump Cost"),
                    "voice_unvoiced_cost": validate_and_convert(voice_unvoiced_cost, default_config["voice_unvoiced_cost"], float, "Voice Unvoiced Cost"),
                    "very_accurate": validate_and_convert(very_accurate, default_config["very_accurate"], float, "Very Accurate"),
                    "use_gender_range": use_gender_range_val,
                    "show_spline": show_spline,
                    "fixed_y_range":validate_and_convert(fixed_y_range, default_config["fixed_y_range"], float, "Fixed Y axis Range")
                }
                if use_gender_range_val:
                    config["min_pitch_male"] = float(M_min_val) if M_min_val else 75.0
                    config["max_pitch_male"] = float(M_max_val) if M_max_val else 500.0
                    config["min_pitch_female"] = float(F_min_val) if F_min_val else 100.0
                    config["max_pitch_female"] = float(F_max_val) if F_max_val else 600.0

                save_config(config)  # 설정 저장
                transcription_runner.start_transcription(tsv_file, config["output_dir"])
                return gr.update(visible=False), gr.update(visible=True), "작업이 시작되었습니다.", ""

            except ValueError as e:
                logger.error(e)
                return gr.update(), gr.update(), f"❌ 오류: {e}" , ""

        def stop_transcription():
            transcription_runner.stop_transcription()
            return gr.update(visible=True), gr.update(visible=False), "작업이 중단되었습니다.", ""
        
        status_output = gr.Textbox(
            label="🖥️ 에러/알림 메시지",
            interactive=False
        )

        # (2) start_transcription()에서는 status_output 쪽에 에러를 표시
        if not tsv_file:
            return (
                gr.update(),  # start_button
                gr.update(),  # stop_button
                "",           # log_output은 건드리지 않음
                "❌ 오류: TSV/CSV 파일이 선택되지 않았습니다."  # status_output
            )

        # 버튼 동작
        start_button.click(
            fn=start_transcription,
            inputs=[
                tsv_file, min_pitch, max_pitch, time_step,
                silence_threshold, voicing_threshold, octave_cost, octave_jump_cost, voice_unvoiced_cost,
                number_of_candidates, very_accurate,
                show_spline,
                fixed_y_range,
                use_gender_range,
                M_min, M_max, F_min, F_max
            ],
            outputs=[start_button, stop_button, log_output, status_output]
        )

        stop_button.click(
            fn=stop_transcription,
            outputs=[start_button, stop_button, log_output, status_output]
        )

        # 실시간 로그 갱신
        def update_logs():
            while True:
                yield transcription_runner.get_logs()
                time.sleep(1)

        main.load(update_logs, outputs=log_output)

    return main

if __name__ == '__main__':
    main = create_gradio_interface()
    main.launch(server_name="0.0.0.0", server_port=7861)
