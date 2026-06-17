"""
pipeline/ingest.py — Extract a 16 kHz mono WAV from any video/audio file.

Uses imageio-ffmpeg which ships a bundled ffmpeg binary — no system
installation required.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import imageio_ffmpeg


SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
}


def _ffmpeg_bin() -> str:
    """Return the path to the bundled ffmpeg binary."""
    return imageio_ffmpeg.get_ffmpeg_exe()


async def extract_audio(source: Path, dest_dir: Path) -> Path:
    """
    Extract audio from *source* and write a 16 kHz mono WAV to *dest_dir*.
    Returns the path to the WAV file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    wav_path = dest_dir / (source.stem + ".wav")

    cmd = [
        _ffmpeg_bin(),
        "-y",               # overwrite without asking
        "-i", str(source),
        "-vn",              # drop video stream
        "-ar", "16000",     # 16 kHz sample rate (Whisper optimum)
        "-ac", "1",         # mono
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (code {proc.returncode}):\n{stderr.decode(errors='replace')}"
        )

    return wav_path


async def trim_audio(source_wav: Path, dest_wav: Path, start_sec: float) -> Path:
    """
    Trim *source_wav* from *start_sec* to end and save to *dest_wav*.
    Returns *dest_wav*.
    """
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-ss", f"{max(0.0, start_sec):.3f}",
        "-i", str(source_wav),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dest_wav),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg trim failed (code {proc.returncode}):\n{stderr.decode(errors='replace')}"
        )

    return dest_wav
