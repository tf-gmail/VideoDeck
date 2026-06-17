"""
main.py — VideoDeck FastAPI application.

Start with:
    python main.py
Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from db.jobs import create_job, delete_job, get_job, init_db, list_jobs, update_job
from pipeline.ingest import SUPPORTED_EXTENSIONS, extract_audio, trim_audio
from pipeline.summarize import list_models, summarize
from pipeline.transcribe import TranscriptionCancelled, transcribe_async

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "output"
TEMP_DIR    = BASE_DIR / "temp"

for _d in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR):
    _d.mkdir(exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="VideoDeck", version="1.0.0")

# Concurrency guard — process one job at a time to stay within VRAM budget
_job_semaphore = asyncio.Semaphore(1)
_job_tasks: dict[str, asyncio.Task] = {}
_cancel_requested: set[str] = set()


class JobCancelled(Exception):
    """Raised when a user stops a job."""


@app.on_event("startup")
async def startup() -> None:
    await init_db()


# ── UI ─────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(BASE_DIR / "ui" / "index.html")


# ── Models endpoint ────────────────────────────────────────────────────────────
@app.get("/api/models")
async def get_models():
    models = await list_models()
    return {"models": models}


# ── Job list ───────────────────────────────────────────────────────────────────
@app.get("/api/jobs")
async def api_list_jobs():
    jobs = await list_jobs()
    # Don't send the full transcript/summary text in the list view
    for j in jobs:
        j.pop("transcript", None)
        j.pop("summary", None)
        full_transcript = OUTPUT_DIR / f"{j['id']}_transcript.txt"
        partial_transcript = OUTPUT_DIR / f"{j['id']}_transcript.partial.txt"
        checkpoint_file = OUTPUT_DIR / f"{j['id']}_transcript.checkpoint.json"
        j["has_transcript"] = full_transcript.exists() or partial_transcript.exists()
        j["has_full_transcript"] = full_transcript.exists()
        j["has_partial_transcript"] = partial_transcript.exists() and checkpoint_file.exists()
        j["has_summary"] = (OUTPUT_DIR / f"{j['id']}_summary.md").exists()
    return jobs


# ── Job detail ─────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ── Create / upload job ────────────────────────────────────────────────────────
@app.post("/api/jobs")
async def api_create_job(
    file: UploadFile = File(...),
    whisper_model: str = Form("large-v3"),
    llm_model: str = Form("qwen2.5-coder:14b"),
    prompt_style: str = Form("study_notes"),
    language: str = Form(""),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Validate that prompt_style maps to an existing prompt file
    prompt_path = BASE_DIR / "prompts" / f"{prompt_style}.txt"
    if not prompt_path.exists():
        raise HTTPException(400, f"Unknown prompt style: {prompt_style}")

    job_id   = str(uuid.uuid4())
    dest     = UPLOAD_DIR / f"{job_id}{suffix}"

    # Save uploaded file
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            await f.write(chunk)

    await create_job(
        job_id,
        file.filename,
        str(dest),
        whisper_model,
        llm_model,
        prompt_style,
        language,
    )

    # Do not auto-start. User must explicitly press Start in UI.
    await update_job(job_id, status="queued", stage="Ready to start", progress=0)

    return {"id": job_id}


@app.post("/api/jobs/{job_id}/start")
async def api_start_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] != "queued":
        raise HTTPException(400, "Only queued jobs can be started")

    source = Path(job["file_path"])
    if not source.exists():
        raise HTTPException(404, "Original source file not found")

    _start_job_task(
        job_id=job_id,
        source=source,
        whisper_model=job.get("whisper_model") or "large-v3",
        llm_model=job.get("llm_model") or "qwen2.5-coder:14b",
        prompt_style=job.get("prompt_style") or "study_notes",
        language=(job.get("language") or "") or None,
        reuse_existing_transcript=False,
        continue_from_checkpoint=False,
    )
    return {"ok": True}


# ── Stop job ───────────────────────────────────────────────────────────────────
@app.post("/api/jobs/{job_id}/stop")
async def api_stop_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] in {"done", "error", "stopped"}:
        return {"ok": True, "message": "Job already finished"}

    _cancel_requested.add(job_id)
    task = _job_tasks.get(job_id)
    if task and not task.done():
        task.cancel()

    await update_job(job_id, status="stopped", stage="Stopping…", error=None)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/restart")
async def api_restart_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] not in {"stopped", "error", "done"}:
        raise HTTPException(400, "Job can only be restarted from stopped, error, or done")

    source = Path(job["file_path"])
    if not source.exists():
        raise HTTPException(404, "Original source file not found")

    await update_job(job_id, status="queued", stage="queued", progress=0, error=None)
    _cancel_requested.discard(job_id)

    _start_job_task(
        job_id=job_id,
        source=source,
        whisper_model=job.get("whisper_model") or "large-v3",
        llm_model=job.get("llm_model") or "qwen2.5-coder:14b",
        prompt_style=job.get("prompt_style") or "study_notes",
        language=(job.get("language") or "") or None,
        reuse_existing_transcript=True,
        continue_from_checkpoint=False,
    )

    return {"ok": True}


@app.post("/api/jobs/{job_id}/continue")
async def api_continue_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] not in {"stopped", "error"}:
        raise HTTPException(400, "Continue is available only for stopped or failed jobs")

    source = Path(job["file_path"])
    if not source.exists():
        raise HTTPException(404, "Original source file not found")

    checkpoint_file = OUTPUT_DIR / f"{job_id}_transcript.checkpoint.json"
    partial_file = OUTPUT_DIR / f"{job_id}_transcript.partial.txt"
    if not checkpoint_file.exists() or not partial_file.exists():
        raise HTTPException(404, "No transcript checkpoint found to continue from")

    await update_job(job_id, status="queued", stage="queued", progress=max(1, job.get("progress") or 1), error=None)
    _cancel_requested.discard(job_id)

    _start_job_task(
        job_id=job_id,
        source=source,
        whisper_model=job.get("whisper_model") or "large-v3",
        llm_model=job.get("llm_model") or "qwen2.5-coder:14b",
        prompt_style=job.get("prompt_style") or "study_notes",
        language=(job.get("language") or "") or None,
        reuse_existing_transcript=False,
        continue_from_checkpoint=True,
    )
    return {"ok": True}


@app.post("/api/jobs/{job_id}/resummarize")
async def api_resummarize_job(
    job_id: str,
    llm_model: str = Form("qwen2.5-coder:14b"),
    prompt_style: str = Form("study_notes"),
):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] == "running":
        raise HTTPException(400, "Cannot re-summarize while job is running")

    prompt_path = BASE_DIR / "prompts" / f"{prompt_style}.txt"
    if not prompt_path.exists():
        raise HTTPException(400, f"Unknown prompt style: {prompt_style}")

    transcript_txt = OUTPUT_DIR / f"{job_id}_transcript.txt"
    if not transcript_txt.exists():
        raise HTTPException(404, "Transcript file not found")

    transcript_text = transcript_txt.read_text(encoding="utf-8")
    if not transcript_text.strip():
        raise HTTPException(400, "Transcript is empty")

    detected_lang = OUTPUT_DIR / f"{job_id}_transcript.json"
    lang_value = job.get("language") or "auto"
    if detected_lang.exists():
        try:
            lang_value = json.loads(detected_lang.read_text(encoding="utf-8")).get("language") or lang_value
        except Exception:
            pass

    logger.info(
        "Job %s: re-summarize requested (model=%s, prompt=%s, language=%s)",
        job_id,
        llm_model,
        prompt_style,
        lang_value,
    )

    await update_job(job_id, status="running", stage="Re-summarizing…", progress=66, error=None)
    await update_job(job_id, stage="Starting summary generation…", progress=67)

    async def _summary_progress(pct: int, stage: str) -> None:
        # Re-summarize occupies 66..99 before final completion.
        overall = min(99, 66 + int((max(0, min(100, pct)) * 33) / 100))
        await update_job(job_id, stage=stage, progress=overall)

    summary_md = await summarize(
        transcript_text,
        llm_model,
        prompt_style,
        progress_cb=_summary_progress,
        output_language=lang_value,
    )

    source = Path(job["file_path"])
    summary_file = OUTPUT_DIR / f"{job_id}_summary.md"

    header = (
        f"# {source.stem}\n\n"
        f"> **Source:** {source.name}  \n"
        f"> **Language:** {lang_value}  \n"
        f"> **Whisper model:** {job.get('whisper_model') or 'large-v3'}  \n"
        f"> **LLM model:** {llm_model}  \n\n"
        "---\n\n"
    )
    summary_file.write_text(header + summary_md, encoding="utf-8")

    await update_job(
        job_id,
        status="done",
        stage="Complete",
        progress=100,
        llm_model=llm_model,
        prompt_style=prompt_style,
        summary=summary_md[:2000],
    )
    return {"ok": True}


# ── Open output folder ─────────────────────────────────────────────────────────
@app.post("/api/jobs/{job_id}/open")
async def api_open_output(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Open the shared output folder even if this job has no summary yet.
    # This avoids false errors for stopped/failed jobs.
    _open_in_explorer(OUTPUT_DIR)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download/transcript")
async def api_download_transcript(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    transcript_file = OUTPUT_DIR / f"{job_id}_transcript.txt"
    partial_file = OUTPUT_DIR / f"{job_id}_transcript.partial.txt"
    if transcript_file.exists():
        download_name = f"{Path(job['filename']).stem}_transcript.txt"
        return FileResponse(transcript_file, media_type="text/plain", filename=download_name)
    if partial_file.exists():
        download_name = f"{Path(job['filename']).stem}_transcript.partial.txt"
        return FileResponse(partial_file, media_type="text/plain", filename=download_name)
    else:
        raise HTTPException(404, "Transcript file not found")


@app.get("/api/jobs/{job_id}/download/summary")
async def api_download_summary(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    summary_file = OUTPUT_DIR / f"{job_id}_summary.md"
    if not summary_file.exists():
        raise HTTPException(404, "Summary file not found")
    download_name = f"{Path(job['filename']).stem}_summary.md"
    return FileResponse(summary_file, media_type="text/markdown", filename=download_name)


# ── Delete job ─────────────────────────────────────────────────────────────────
@app.delete("/api/jobs/{job_id}")
async def api_delete_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Stop active execution before deleting job metadata.
    _cancel_requested.add(job_id)
    task = _job_tasks.get(job_id)
    if task and not task.done():
        task.cancel()

    # Best-effort cleanup of files belonging to this job.
    source = Path(job.get("file_path") or "")
    if source.exists():
        try:
            source.unlink()
        except Exception:
            pass

    for suffix in (
        "_summary.md",
        "_transcript.txt",
        "_transcript.json",
        "_transcript.partial.txt",
        "_transcript.partial.segments.jsonl",
        "_transcript.checkpoint.json",
    ):
        p = OUTPUT_DIR / f"{job_id}{suffix}"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    for p in (TEMP_DIR / f"{job_id}.wav", TEMP_DIR / f"{job_id}.resume.wav"):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    await delete_job(job_id)
    _cancel_requested.discard(job_id)
    return {"ok": True}


# ── Pipeline ───────────────────────────────────────────────────────────────────
async def _run_pipeline(
    job_id: str,
    source: Path,
    whisper_model: str,
    llm_model: str,
    prompt_style: str,
    language: str | None,
    reuse_existing_transcript: bool,
    continue_from_checkpoint: bool,
) -> None:
    try:
        async with _job_semaphore:
            _raise_if_cancelled(job_id)
            await _pipeline_steps(
                job_id,
                source,
                whisper_model,
                llm_model,
                prompt_style,
                language,
                reuse_existing_transcript,
                continue_from_checkpoint,
            )
    except (JobCancelled, TranscriptionCancelled, asyncio.CancelledError):
        await update_job(job_id, status="stopped", stage="Stopped", error=None)
    except Exception as exc:
        await update_job(job_id, status="error", stage="Error", error=str(exc), progress=0)
    finally:
        _cancel_requested.discard(job_id)
        # Clean up temporary WAV
        wav = TEMP_DIR / f"{job_id}.wav"
        if wav.exists():
            wav.unlink()


async def _pipeline_steps(
    job_id: str,
    source: Path,
    whisper_model: str,
    llm_model: str,
    prompt_style: str,
    language: str | None,
    reuse_existing_transcript: bool,
    continue_from_checkpoint: bool,
) -> None:
    loop = asyncio.get_running_loop()
    _raise_if_cancelled(job_id)

    transcript_json = OUTPUT_DIR / f"{job_id}_transcript.json"
    transcript_txt = OUTPUT_DIR / f"{job_id}_transcript.txt"
    partial_transcript_txt = OUTPUT_DIR / f"{job_id}_transcript.partial.txt"
    partial_segments_jsonl = OUTPUT_DIR / f"{job_id}_transcript.partial.segments.jsonl"
    checkpoint_file = OUTPUT_DIR / f"{job_id}_transcript.checkpoint.json"
    resume_offset_sec = 0.0
    resume_language = language
    prior_partial_text = ""
    prior_segments: list[dict] = []
    prior_progress = 5

    can_reuse_full_transcript = (
        reuse_existing_transcript
        and transcript_txt.exists()
        and transcript_txt.stat().st_size > 0
    )

    if can_reuse_full_transcript:
        await update_job(job_id, status="running", stage="Using saved transcript…", progress=65)
        transcript_text = transcript_txt.read_text(encoding="utf-8")
        detected_lang = language or "auto"
        segments: list[dict] = []
        if transcript_json.exists():
            try:
                j = json.loads(transcript_json.read_text(encoding="utf-8"))
                detected_lang = j.get("language") or detected_lang
                segments = j.get("segments") or []
            except Exception:
                pass
    else:
        if continue_from_checkpoint and checkpoint_file.exists() and partial_transcript_txt.exists():
            try:
                ckpt = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                resume_offset_sec = float(ckpt.get("last_end_sec") or 0.0)
                resume_language = ckpt.get("language") or language
                prior_progress = int(ckpt.get("progress") or 5)
                prior_partial_text = partial_transcript_txt.read_text(encoding="utf-8").strip()
                if partial_segments_jsonl.exists():
                    with partial_segments_jsonl.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                prior_segments.append(json.loads(line))
            except Exception:
                resume_offset_sec = 0.0
                prior_partial_text = ""
                prior_segments = []

        # ── Stage 1: Extract audio ────────────────────────────────────────────────
        await update_job(
            job_id,
            status="running",
            stage="Extracting audio…" if resume_offset_sec <= 0 else "Preparing resume checkpoint…",
            progress=2 if resume_offset_sec <= 0 else max(2, prior_progress),
        )
        wav_path = await extract_audio(source, TEMP_DIR)
        # Rename to use job_id so cleanup is reliable
        final_wav = TEMP_DIR / f"{job_id}.wav"
        wav_path.rename(final_wav)
        _raise_if_cancelled(job_id)

        transcribe_wav_path = final_wav
        time_offset = 0.0
        if resume_offset_sec > 0:
            await update_job(job_id, stage="Continuing from checkpoint…", progress=max(5, prior_progress))
            transcribe_wav_path = TEMP_DIR / f"{job_id}.resume.wav"
            await trim_audio(final_wav, transcribe_wav_path, resume_offset_sec)
            time_offset = resume_offset_sec

        # ── Stage 2: Transcribe ───────────────────────────────────────────────────
        await update_job(job_id, stage="Transcribing…", progress=max(5, prior_progress))

        # Reset partial transcript file for this run
        if resume_offset_sec <= 0:
            partial_transcript_txt.write_text("", encoding="utf-8")
            partial_segments_jsonl.write_text("", encoding="utf-8")

        def _progress(pct: int) -> bool:
            if job_id in _cancel_requested:
                return False
            # Transcription covers 5 → 65 % of overall progress
            base = max(5, prior_progress)
            overall = base + int(pct * max(1, (65 - base)) / 100)
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    update_job(job_id, progress=overall, stage="Transcribing…")
                )
            )
            return True

        def _on_segment(seg: dict) -> None:
            # Persist progress to disk so stopped jobs still have readable transcript progress.
            with partial_transcript_txt.open("a", encoding="utf-8") as f:
                f.write(seg["text"] + "\n")
            with partial_segments_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
            checkpoint_file.write_text(
                json.dumps(
                    {
                        "last_end_sec": seg["end"],
                        "language": resume_language or language or "",
                        "progress": None,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        result = await transcribe_async(
            transcribe_wav_path,
            whisper_model,
            resume_language,
            _progress,
            _on_segment,
            time_offset=time_offset,
        )
        _raise_if_cancelled(job_id)
        if prior_partial_text:
            transcript_text = (prior_partial_text + "\n" + result["text"]).strip()
        else:
            transcript_text = result["text"]
        detected_lang = result["language"]
        segments = prior_segments + result["segments"]

        # Save transcript
        transcript_json.write_text(
            json.dumps({"language": detected_lang, "segments": segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        transcript_txt.write_text(transcript_text, encoding="utf-8")
        if partial_transcript_txt.exists():
            partial_transcript_txt.unlink()
        if partial_segments_jsonl.exists():
            partial_segments_jsonl.unlink()
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    await update_job(
        job_id,
        stage="Summarizing…",
        progress=66,
        whisper_model=whisper_model,
        llm_model=llm_model,
        prompt_style=prompt_style,
        language=language or "",
        transcript=transcript_text[:4000],  # store a preview only
    )
    _raise_if_cancelled(job_id)

    # ── Stage 3: Summarize ────────────────────────────────────────────────────
    async def _summary_progress(pct: int, stage: str) -> None:
        # Summarization occupies 66..99 before final completion.
        overall = min(99, 66 + int((max(0, min(100, pct)) * 33) / 100))
        await update_job(job_id, stage=stage, progress=overall)

    summary_md = await summarize(
        transcript_text,
        llm_model,
        prompt_style,
        progress_cb=_summary_progress,
        output_language=detected_lang,
    )
    _raise_if_cancelled(job_id)

    # Save summary
    summary_file = OUTPUT_DIR / f"{job_id}_summary.md"
    header = (
        f"# {source.stem}\n\n"
        f"> **Source:** {source.name}  \n"
        f"> **Language:** {detected_lang}  \n"
        f"> **Whisper model:** {whisper_model}  \n"
        f"> **LLM model:** {llm_model}  \n\n"
        "---\n\n"
    )
    summary_file.write_text(header + summary_md, encoding="utf-8")

    await update_job(
        job_id,
        status="done",
        stage="Complete",
        progress=100,
        summary=summary_md[:2000],  # store a preview
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _raise_if_cancelled(job_id: str) -> None:
    if job_id in _cancel_requested:
        raise JobCancelled("Job stopped by user")


def _start_job_task(
    job_id: str,
    source: Path,
    whisper_model: str,
    llm_model: str,
    prompt_style: str,
    language: str | None,
    reuse_existing_transcript: bool,
    continue_from_checkpoint: bool,
) -> None:
    task = asyncio.create_task(
        _run_pipeline(
            job_id=job_id,
            source=source,
            whisper_model=whisper_model,
            llm_model=llm_model,
            prompt_style=prompt_style,
            language=language,
            reuse_existing_transcript=reuse_existing_transcript,
            continue_from_checkpoint=continue_from_checkpoint,
        )
    )
    _job_tasks[job_id] = task
    task.add_done_callback(lambda _t, jid=job_id: _job_tasks.pop(jid, None))


def _open_in_explorer(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
