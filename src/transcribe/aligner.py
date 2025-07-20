# src/transcribe/aligner.py
from __future__ import annotations
from pathlib import Path
import subprocess, tempfile, uuid, re
import soundfile as sf
from textgrid import TextGrid
from utils.logger import main_logger
from utils.ipa2kr import ipa2kr

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
    Returns:
        A dictionary with keys "words" and "phonemes", each containing a list of dictionaries
        with "start", "end", and "text" keys for each aligned segment.
    """

    def __init__(self, dict_path: str = "korean_mfa", model: str = "korean_mfa", njobs: int = 8):
        self.dict_path = dict_path
        self.model = model
        self.njobs = njobs

    def _safe_wav(self, src: Path, dst: Path):
        """libsndfile 로 열리지 않는 WAV 는 ffmpeg 로 변환"""
        try:
            with sf.SoundFile(src) as _:
                dst.symlink_to(src)       # 통과 → 심볼릭 링크 유지
        except Exception:
            logger.warning(f"wav 파일 libsndfile 오류, [convert] {src.name} -> PCM 16 kHz")
            cmd = ['ffmpeg', '-y', '-i', str(src), *PCM_ARGS, str(dst)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=True)

    def align_batch(self, pairs, *, njobs=8, single_spk=True):
        """
        Aligns a batch of audio files with their transcriptions.
        Args:
            pairs (list of tuples): Each tuple contains (wav_path, transcription_text).
            njobs (int): Number of parallel jobs to run.
            single_spk (bool): If True, assumes a single speaker for all files.
        Returns:
            dict: A dictionary with aligned words and phonemes for each audio file.
        """
        sid = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix=f"mfa_{sid}_") as tmp:
            corpus = Path(tmp) / "corpus"; corpus.mkdir()
            out    = Path(tmp) / "out"   ; out.mkdir()

            for wav, txt in pairs:
                src = Path(wav).resolve()
                dst = corpus / src.name          # 이름 충돌 주의!
                self._safe_wav(src, dst)         # ← NEW
                (dst.with_suffix(".lab")).write_text(txt, 'utf-8')

            cmd = [
                "mfa", "align",
                corpus,                     # ① CORPUS_DIRECTORY
                self.dict_path,             # ② DICTIONARY_PATH
                self.model,                 # ③ ACOUSTIC_MODEL_PATH
                out,                        # ④ OUTPUT_DIRECTORY
                "-j", str(njobs),           # 이하 옵션
                "--clean", "--quiet"
            ]
            if single_spk:
                cmd += ["--single_speaker", "--no_fmllr"]

            try:
                subprocess.run(list(map(str, cmd)), check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MFA 배치 정렬 실패: {e}") from None

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
                "mfa", "align", corpus_dir, self.dict_path, self.model, out_dir,
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
