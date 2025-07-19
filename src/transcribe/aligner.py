# src/transcribe/aligner.py
from __future__ import annotations
from pathlib import Path
import subprocess, shutil, tempfile, json, uuid
import soundfile as sf
from textgrid import TextGrid
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
            logger.warning(f"[convert] {src.name} -> PCM 16 kHz")
            cmd = ['ffmpeg', '-y', '-i', str(src), *PCM_ARGS, str(dst)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=True)

    def align_batch(self, pairs, *, njobs=8, single_spk=True):
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

def tg_to_alignment(tg: TextGrid) -> dict:
    words, phones = [], []
    for tier in tg.tiers:
        name = tier.name.lower()
        if name == "words":
            words.extend(
                {"start": iv.minTime, "end": iv.maxTime, "text": iv.mark}
                for iv in tier.intervals if iv.mark.strip()
            )
        elif name =="phones":
            phones.extend(
                {"start": iv.minTime, "end": iv.maxTime, "text": iv.mark}
                for iv in tier.intervals if iv.mark.strip()
            )
    return {"words": words, "phonemes": phones}