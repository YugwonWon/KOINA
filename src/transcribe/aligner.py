# src/transcribe/aligner.py
from __future__ import annotations

import json
import os, signal, shutil
from pathlib import Path
import subprocess, tempfile, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf
from textgrid import TextGrid
from utils.ipa2kr import ipa2kr

from utils.logger import main_logger
logger = main_logger.getChild('aligner')

PCM_ARGS = ['-ar', '16000', '-ac', '1', '-sample_fmt', 's16']

__all__ = ["MFAAligner"]

class MFAAligner:
    """
    MFA (Montreal Forced Aligner) wrapper for aligning audio with text.
    Usage:
        aligner = MFAAligner(dict_path="path/to/dict.dict", model="korean_mfa", njobs=8)
        result = aligner.align(wav="path/to/audio.wav", text="transcription text")
        # result will contain aligned words and phonemes as dictionaries.
    Parameters:
        - dict_path: Path to the pronunciation dictionary file (.dict).
        - model: Name or path of the acoustic model to use.
        - njobs: Number of parallel jobs to run for alignment.
        - single_spk: If True, assumes a single speaker in the audio.
    Returns:
        A dictionary with keys "words" and "phonemes", each containing a list of dictionaries
        with "start", "end", and "text" keys for each aligned segment.
    """

    def __init__(self, dict_path: str = "korean_mfa", model: str = "korean_mfa"):
        self.proc = None
        self.dict_path = dict_path
        self.model = model
        self.config = None
        self.single_spk = False
        self.mfa_path = "mfa"  # default: assume mfa is in PATH
        self.njobs = 4  # 일반 PC 기준 기본값
        
        # load config from file
        self._load_config()
        
    def _load_config(self, config_path="out/config.json"):
        """Load configuration from a file"""
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.njobs = self.config.get("alignment_njobs", 4)
        self.single_spk = self.config.get("alignment_single_spk", False)
        self.mfa_path = self.config.get("mfa_path", "mfa")
        self.dict_path = self.config.get("mfa_dictionary", "korean_mfa")
        self.model = self.config.get("mfa_model", "korean_mfa")
        # MFA conda 환경의 bin 경로 (fstcompile 등 의존성 포함)
        self.mfa_env_bin = self.config.get("mfa_env_bin", "/home/yugwon/miniconda3/envs/mfa/bin")
    
    
    def _safe_wav(self, src: Path, dst: Path):
        """libsndfile 로 열리지 않는 WAV 는 ffmpeg 로 변환, 정상이면 복사"""
        try:
            with sf.SoundFile(src) as _:
                # 정상 WAV → 단순 복사 (symlink 대신)
                shutil.copy2(src, dst)
        except Exception:
            logger.warning(f"[ALIGNER] wav 파일 libsndfile 오류, [convert] {src.name} -> PCM 16 kHz")
            cmd = ['ffmpeg', '-y', '-i', str(src), *PCM_ARGS, str(dst)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=True)
    
    def terminate(self, timeout=2):
        """실행 중인 MFA 프로세스를 종료한다(SigTERM → SigKILL)."""
        if not self.proc or self.proc.poll() is not None:
            return  # 이미 끝났거나 실행 안 됨
        try:
            pgid = os.getpgid(self.proc.pid)     # 프로세스 그룹 ID
            os.killpg(pgid, signal.SIGTERM)      # 부드럽게 종료
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)  # 확실히 종료
        except ProcessLookupError:
            pass
        
    def _prepare_single_file(self, args):
        """단일 파일 준비 (멀티스레딩용 워커)"""
        wav, txt, corpus = args
        try:
            src = Path(wav).resolve()
            speaker = src.parent.name
            rel_path = Path(speaker) / src.name
            dst = corpus / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            self._safe_wav(src, dst)
            (dst.with_suffix(".lab")).write_text(txt, 'utf-8')
            return (True, src.name, None)
        except Exception as e:
            return (False, Path(wav).name, str(e))
    
    def align_batch(self, pairs, stop_flag=None):
        """
        Aligns a batch of audio files with their transcriptions.
        Args:
            pairs (list of tuples): Each tuple contains (wav_path, transcription_text).
        Returns:
            dict: A dictionary with aligned words and phonemes for each audio file.
        """
        sid = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix=f"mfa_{sid}_") as tmp:
            corpus = Path(tmp) / "corpus"; corpus.mkdir()
            out    = Path(tmp) / "out"   ; out.mkdir()

            # ───── 멀티스레딩으로 파일 준비 (I/O 바운드 작업) ─────
            logger.info(f"[ALIGNER] {len(pairs)}개의 파일을 병렬로 준비합니다... (workers={self.njobs})")
            tasks = [(wav, txt, corpus) for wav, txt in pairs]
            
            success_count = 0
            error_count = 0
            with ThreadPoolExecutor(max_workers=self.njobs) as executor:
                futures = {executor.submit(self._prepare_single_file, task): task for task in tasks}
                for future in as_completed(futures):
                    if stop_flag is not None and stop_flag.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError("File preparation cancelled by user")
                    
                    success, filename, error = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.warning(f"[ALIGNER] 파일 준비 실패: {filename} - {error}")
            
            logger.info(f"[ALIGNER] 파일 준비 완료: 성공 {success_count}개, 실패 {error_count}개")
            logger.info(f"[ALIGNER] MFA 배치 정렬을 시작합니다... (njobs={self.njobs}, single_spk={self.single_spk})")
            cmd = [
                self.mfa_path, "align", str(corpus), self.dict_path, self.model, str(out),
                "-j", str(self.njobs), "--clean",
                "--verbose", "--disable_tqdm"          # ← 진행 단계 텍스트 출력
            ]
            if self.single_spk:
                cmd += ["--single_speaker", "--no_fmllr"]

            logger.info("[ALIGNER] MFA command: %s", " ".join(map(str, cmd)))

            # MFA conda 환경의 PATH를 포함한 환경변수 설정
            env = os.environ.copy()
            env["PATH"] = f"{self.mfa_env_bin}:{env.get('PATH', '')}"

            # ───── subprocess.run → Popen 스트리밍 ─────
            self.proc = subprocess.Popen(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                preexec_fn=os.setsid,
                env=env
            )
            with self.proc.stdout:
                for line in self.proc.stdout:
                    logger.info("[ALIGNER] %s", line.rstrip())
                    # ----- STOP 버튼을 눌렀다면 즉시 종료 -----
                    if stop_flag is not None and stop_flag.is_set():
                        self.terminate()
                        raise RuntimeError("Alignment cancelled by user")
            
            self.proc.wait()                                   # 프로세스 종료 대기
            
            # TextGrid 수집
            grids = {tg.stem: TextGrid.fromFile(str(tg)) for tg in out.rglob("*.TextGrid")}
            return grids
    
    def align(self, wav: str | Path, text: str) -> dict:
        """ Aligns a single audio file with its transcription text.     
        Args:
            wav (str | Path): Path to the audio file.
            text (str): Transcription text for the audio.
        Returns:
            dict: A dictionary containing aligned words and phonemes.
        """
        wav = Path(wav).expanduser().resolve()
        sid = uuid.uuid4().hex  # isolate each call in its own temp dir
        with tempfile.TemporaryDirectory(prefix=f"mfa_{sid}_") as tmp:
            corpus_dir = Path(tmp) / "corpus"
            out_dir    = Path(tmp) / "out"
            corpus_dir.mkdir()

            # 1. create <utt>.wav symlink + .lab
            wav_dst = corpus_dir / wav.name
            wav_dst.symlink_to(wav)
            (wav_dst.with_suffix(".lab")).write_text(text, encoding="utf-8")

            # 2. run MFA
            subprocess.run([
                self.mfa_path, "align", corpus_dir, self.dict_path, self.model, out_dir,
                "-j", str(self.njobs), "--clean", "--quiet"
            ], check=True)

            # 3. parse TextGrid → word/phoneme list (Praat indexing is 1‑based)
            tg_path = next(out_dir.rglob("*.TextGrid"))
            tg = TextGrid.fromFile(str(tg_path))
            words, phonemes = [], []
            for tier in tg.tiers:
                if tier.name.lower() == "word":
                    words = [
                        {"start": iv.minTime, "end": iv.maxTime, "text": iv.mark}
                        for iv in tier.intervals if iv.mark.strip()
                    ]
                elif tier.name.lower() in {"phone", "phoneme"}:
                    phonemes = [
                        {"start": iv.minTime, "end": iv.maxTime, "text": iv.mark}
                        for iv in tier.intervals if iv.mark.strip()
                    ]

            return {"words": words, "phonemes": phonemes}

def restore_eojeols(word_ivls, transcript):
    """
    MFA word intervals → transcript 의 공백 토큰 순서에 맞춰 ‘어절’ 복원
    * 침묵 interval(text=="")은 start/end 계산에는 포함하지만
      토큰 결합·비교에는 완전히 무시
    """
    tokens = transcript.strip().split()        # ['나는', '실력있는', …]
    tok_idx = 0                                # 다음에 만들어야 할 어절 index
    buf_txt, buf_start, buf_end = "", None, None
    out = []

    for iv in word_ivls:
        # 침묵 interval → 길이만 end 로 확장, 내용 결합은 skip
        if iv["text"] == "":
            if buf_start is not None:
                buf_end = iv["end"]
            continue

        # 첫 실음절 interval이면 start 기록
        if buf_start is None:
            buf_start = iv["start"]

        buf_end = iv["end"]
        buf_txt += iv["text"]                  # 공백 없는 순수 음절 / 어절 이어붙이기

        # 현재 버퍼가 목표 token 과 일치하면 flush
        if tok_idx < len(tokens) and buf_txt == tokens[tok_idx]:
            out.append({"start": buf_start, "end": buf_end, "text": buf_txt})
            tok_idx += 1
            buf_txt, buf_start = "", None      # 버퍼 초기화

    # 남은 버퍼(끝이 무음으로 끝나는 경우) 처리
    if buf_start is not None:
        out.append({"start": buf_start, "end": buf_end, "text": buf_txt})

    return out

def tg_to_alignment(tg: TextGrid, transcript: str) -> dict:
    """ Converts a TextGrid object to an alignment dictionary.
    Args:
        tg (TextGrid): The TextGrid object containing alignment data.
        transcript (str): The original transcript text for reference.
    Returns:
        dict: A dictionary with keys "words", "phonemes", and "phonemes_kr",
    """
    # TODO: phonemes_kr tier는 현재 MFA에서의 G2P 모델을 개량할 수 없으므로 
    # IPA를 한글 자모로 바꿔줄 수 있는 정교한 후처리가 필요하다. e.g. ɲʌ -> 녀(현재는 ㄴ(니), ㅓ로 변환됨)
    words, phones, phones_kr = [], [], []

    for tier in tg.tiers:
        name = tier.name.lower()
        if name in {"word", "words"}:
            # mark 가 공백(None)이어도 포함
            for iv in tier.intervals:
                words.append({"start": iv.minTime,
                              "end"  : iv.maxTime,
                              "text" : iv.mark or ""})
        elif name in {"phone", "phoneme", "phones"}:
            for iv in tier.intervals:
                phon_dict = {"start": iv.minTime,
                             "end"  : iv.maxTime,
                             "text" : iv.mark}
                phones.append(phon_dict)
                phones_kr.append({"start": iv.minTime,
                                  "end"  : iv.maxTime,
                                  "text" : ipa2kr(iv.mark)})

    # 형태소 분석된 raw timestamp에서 어절 복원된 것, 이것을 기존 words로 대체하고, raw words를 words_token으로 저장
    words_restore = restore_eojeols(words, transcript)
    
    return {"words": words_restore,
            "phonemes": phones,
            "phonemes_kr": phones_kr,
            "words_token": words}
