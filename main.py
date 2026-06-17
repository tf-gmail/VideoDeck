"""
main.py — VideoDeck FastAPI application.

Start with:
    python main.py
Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from db.jobs import create_job, delete_job, get_job, init_db, list_jobs, update_job
from pipeline.ingest import SUPPORTED_EXTENSIONS, extract_audio
from pipeline.summarize import list_models, summarize
from pipeline.transcribe import TranscriptionCancelled, transcribe_async

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
        j["has_transcript"] = (OUTPUT_DIR / f"{j['id']}_transcript.txt").exists()
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

    # Kick off the pipeline in the background
    task = asyncio.create_task(
        _run_pipeline(
            job_id=job_id,
            source=dest,
            whisper_model=whisper_model,
            llm_model=llm_model,
            prompt_style=prompt_style,
            language=language or None,
        )
    )
    _job_tasks[job_id] = task
    task.add_done_callback(lambda _t, jid=job_id: _job_tasks.pop(jid, None))

    return {"id": job_id}


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

    task = asyncio.create_task(
        _run_pipeline(
            job_id=job_id,
            source=source,
            whisper_model=job.get("whisper_model") or "large-v3",
            llm_model=job.get("llm_model") or "qwen2.5-coder:14b",
            prompt_style=job.get("prompt_style") or "study_notes",
            language=(job.get("language") or "") or None,
        )
    )
    _job_tasks[job_id] = task
    task.add_done_callback(lambda _t, jid=job_id: _job_tasks.pop(jid, None))

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

    await update_job(job_id, status="running", stage="Re-summarizing…", progress=70, error=None)
    summary_md = await summarize(transcript_text, llm_model, prompt_style)

    source = Path(job["file_path"])
    summary_file = OUTPUT_DIR / f"{job_id}_summary.md"
    detected_lang = (OUTPUT_DIR / f"{job_id}_transcript.json")
    lang_value = job.get("language") or "auto"
    if detected_lang.exists():
        try:
            lang_value = json.loads(detected_lang.read_text(encoding="utf-8")).get("language") or lang_value
        except Exception:
            pass

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
    output_file = OUTPUT_DIR / f"{job_id}_summary.md"
    if not output_file.exists():
        raise HTTPException(404, "Output file not found")
    # Open the folder containing the output file in the system file explorer
    _open_in_explorer(output_file.parent)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download/transcript")
async def api_download_transcript(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    transcript_file = OUTPUT_DIR / f"{job_id}_transcript.txt"
    if not transcript_file.exists():
        raise HTTPException(404, "Transcript file not found")
    download_name = f"{Path(job['filename']).stem}_transcript.txt"
    return FileResponse(transcript_file, media_type="text/plain", filename=download_name)


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
    await delete_job(job_id)
    return {"ok": True}


# ── Pipeline ───────────────────────────────────────────────────────────────────
async def _run_pipeline(
    job_id: str,
    source: Path,
    whisper_model: str,
    llm_model: str,
    prompt_style: str,
    language: str | None,
) -> None:
    try:
        async with _job_semaphore:
            _raise_if_cancelled(job_id)
            await _pipeline_steps(
                job_id, source, whisper_model, llm_model, prompt_style, language
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
) -> None:
    loop = asyncio.get_running_loop()
    _raise_if_cancelled(job_id)

    # ── Stage 1: Extract audio ────────────────────────────────────────────────
    await update_job(job_id, status="running", stage="Extracting audio…", progress=2)
    wav_path = await extract_audio(source, TEMP_DIR)
    # Rename to use job_id so cleanup is reliable
    final_wav = TEMP_DIR / f"{job_id}.wav"
    wav_path.rename(final_wav)
    _raise_if_cancelled(job_id)

    # ── Stage 2: Transcribe ───────────────────────────────────────────────────
    await update_job(job_id, stage="Transcribing…", progress=5)

    def _progress(pct: int) -> bool:
        if job_id in _cancel_requested:
            return False
        # Transcription covers 5 → 65 % of overall progress
        overall = 5 + int(pct * 0.60)
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(
                update_job(job_id, progress=overall, stage="Transcribing…")
            )
        )
        return True

    result = await transcribe_async(final_wav, whisper_model, language, _progress)
    _raise_if_cancelled(job_id)
    transcript_text = result["text"]
    detected_lang   = result["language"]
    segments        = result["segments"]

    # Save transcript
    transcript_json = OUTPUT_DIR / f"{job_id}_transcript.json"
    transcript_txt  = OUTPUT_DIR / f"{job_id}_transcript.txt"
    transcript_json.write_text(
        json.dumps({"language": detected_lang, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    transcript_txt.write_text(transcript_text, encoding="utf-8")

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
    summary_md = await summarize(transcript_text, llm_model, prompt_style)
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
