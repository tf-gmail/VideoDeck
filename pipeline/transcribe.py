"""
pipeline/transcribe.py — Transcribe a WAV file using faster-whisper on CUDA.
"""

from __future__ import annotations

import asyncio
import os
import site
import sys
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

# Module-level model cache — loaded once, reused across jobs
_model: WhisperModel | None = None
_loaded_model_size: str = ""


class TranscriptionCancelled(Exception):
    """Raised when transcription is stopped by the user."""


def _configure_windows_cuda_dll_paths() -> None:
    """
    Make CUDA runtime DLLs discoverable when installed via pip packages.

    Required on Windows when ctranslate2 reports missing cublas/cudnn DLLs.
    """
    if sys.platform != "win32":
        return

    cuda_bin_dirs: list[Path] = []
    for sp in site.getsitepackages():
        nvidia_root = Path(sp) / "nvidia"
        cuda_bin_dirs.extend(
            [
                nvidia_root / "cublas" / "bin",
                nvidia_root / "cudnn" / "bin",
            ]
        )

    existing_dirs = [p for p in cuda_bin_dirs if p and p.exists()]
    if not existing_dirs:
        return

    # Add to DLL search path for current process (preferred on modern Python).
    for dll_dir in existing_dirs:
        try:
            os.add_dll_directory(str(dll_dir))
        except (AttributeError, OSError):
            # Fall back to PATH update below.
            pass

    # Also prepend to PATH for libraries that still rely on PATH lookup.
    current_path = os.environ.get("PATH", "")
    prefix = os.pathsep.join(str(p) for p in existing_dirs)
    if prefix and prefix not in current_path:
        os.environ["PATH"] = prefix + os.pathsep + current_path


def load_model(model_size: str = "large-v3") -> WhisperModel:
    """Load (or return cached) the Whisper model on CUDA."""
    global _model, _loaded_model_size
    if _model is None or _loaded_model_size != model_size:
        _configure_windows_cuda_dll_paths()
        _model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",   # fp16 is fastest on Ampere/Ada
        )
        _loaded_model_size = model_size
    return _model


def transcribe_wav(
    wav_path: Path,
    model_size: str = "large-v3",
    language: str | None = None,
    progress_cb: Callable[[int], bool | None] | None = None,
    segment_cb: Callable[[dict], None] | None = None,
    time_offset: float = 0.0,
) -> dict:
    """
    Transcribe *wav_path* and return a dict with:
      - text:     full plain-text transcript
      - segments: list of {start, end, text} dicts
      - language: detected language code
    """
    model = load_model(model_size)

    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=True,   # skip silent sections — faster & cleaner
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = []
    full_text_parts = []
    duration = info.duration or 1.0

    for seg in segments_iter:
        seg_dict = {
            "start": round(seg.start + time_offset, 2),
            "end": round(seg.end + time_offset, 2),
            "text": seg.text.strip(),
        }
        segments.append(seg_dict)
        full_text_parts.append(seg_dict["text"])
        if segment_cb:
            segment_cb(seg_dict)
        if progress_cb:
            pct = min(int((seg.end / duration) * 100), 99)
            should_continue = progress_cb(pct)
            if should_continue is False:
                raise TranscriptionCancelled("Transcription stopped by user")

    return {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "language": info.language,
    }


async def transcribe_async(
    wav_path: Path,
    model_size: str = "large-v3",
    language: str | None = None,
    progress_cb: Callable[[int], bool | None] | None = None,
    segment_cb: Callable[[dict], None] | None = None,
    time_offset: float = 0.0,
) -> dict:
    """Run transcription in a thread pool so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: transcribe_wav(wav_path, model_size, language, progress_cb, segment_cb, time_offset),
    )
