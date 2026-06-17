# VideoDeck — Design Document

## Overview

VideoDeck is a local-first desktop tool that ingests video or audio files, transcribes the spoken content, and then uses a locally hosted LLM to distill the transcript into structured, easy-to-learn study material. All processing stays on-device, powered by an NVIDIA RTX 4070 Ti Super (16 GB VRAM).

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        VideoDeck UI                        │
│  (Minimal web UI served by a local FastAPI backend)        │
└───────────────┬────────────────────────────┬───────────────┘
                │                            │
       ┌────────▼────────┐          ┌────────▼────────┐
       │  Transcription  │          │  Summarization  │
       │  Engine         │          │  Engine         │
       │  (Whisper.cpp / │          │  (Ollama +      │
       │   faster-whisper)│         │   local LLM)    │
       └────────┬────────┘          └────────┬────────┘
                │                            │
       ┌────────▼────────────────────────────▼────────┐
       │           File / Job Manager                  │
       │   SQLite job queue · transcript cache         │
       └───────────────────────────────────────────────┘
```

---

## Processing Pipeline

### Step 1 — Ingest
- The user drops one or more **video** (`.mp4`, `.mkv`, `.webm`, `.mov`) or **audio** (`.mp3`, `.wav`, `.m4a`, `.flac`) files onto the UI.
- VideoDeck extracts the audio track using **FFmpeg** (shipped as a sidecar binary) and writes a temporary `.wav` at 16 kHz mono — the optimal format for Whisper.

### Step 2 — Transcription
- **Engine:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — a CTranslate2-optimised port of OpenAI Whisper that runs entirely on the GPU.
- **Recommended model:** `large-v3` (fits comfortably in 16 GB VRAM alongside the LLM when quantised).
- Output: a timestamped transcript segmented by sentence, saved as JSON and plain text.

| Whisper Model | VRAM  | WER    | Speed (RTX 4070 Ti Super) |
|---------------|-------|--------|---------------------------|
| `base`        | ~1 GB | high   | ~100× real-time           |
| `medium`      | ~5 GB | medium | ~60× real-time            |
| `large-v3`    | ~10 GB| low    | ~25× real-time            |

### Step 3 — Summarization / Key-Point Extraction
- **Engine:** [Ollama](https://ollama.com) — a local model server that exposes an OpenAI-compatible HTTP API on `localhost:11434`.
- **Recommended models** (all run in Q4_K_M quantisation on a 4070 Ti Super):

| Model                | VRAM   | Best for                         |
|----------------------|--------|----------------------------------|
| `llama3.2:3b`        | ~3 GB  | Fast, lightweight summaries      |
| `mistral:7b`         | ~5 GB  | Balanced quality / speed         |
| `qwen2.5:14b`        | ~10 GB | High-quality study material      |
| `deepseek-r1:14b`    | ~10 GB | Reasoning-heavy content          |

- Whisper and the LLM share the 16 GB VRAM budget. The recommended pairing is **`large-v3` + `mistral:7b`** or **`medium` + `qwen2.5:14b`**.

### Step 4 — Output Generation
The LLM receives the full transcript and a structured prompt that instructs it to produce:

1. **TL;DR** — 3–5 sentence summary of the entire video.
2. **Key Points** — bulleted list of the most important concepts.
3. **Topic Breakdown** — timestamped sections with a heading and short explanation.
4. **Glossary** — domain-specific terms mentioned, with one-line definitions.
5. **Quiz Questions** — 5–10 questions to test understanding (with answers).

All output is written to a Markdown file alongside the source video.

---

## Local Model Setup (RTX 4070 Ti Super)

### Prerequisites
```
NVIDIA Driver  ≥ 555
CUDA Toolkit   ≥ 12.4
cuDNN          ≥ 9.x
```

### Install Ollama
```bash
# Windows — download and run the installer from https://ollama.com/download
ollama serve                        # starts the API server on port 11434
ollama pull mistral:7b              # download a model
ollama pull qwen2.5:14b
```

### Install faster-whisper
```bash
pip install faster-whisper
# CUDA acceleration is enabled automatically when CUDA is present
```

### Environment variable (force GPU for Ollama)
```
OLLAMA_NUM_GPU=999   # use all available VRAM layers
```

---

## UI — Minimal Web Frontend

The UI is a single-page application served by the Python backend. No external CDN; all assets are bundled.

```
┌─────────────────────────────────────────────────────┐
│  VideoDeck                              ⚙ Settings  │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌─────────────────────────────────────────────┐  │
│   │   Drop video / audio files here             │  │
│   │   or click to browse                        │  │
│   └─────────────────────────────────────────────┘  │
│                                                     │
│   Jobs                                              │
│   ┌──────────────────────────────┬──────┬───────┐  │
│   │ lecture_ml_week3.mp4         │ 87%  │  ●    │  │
│   │ podcast_ep12.mp3             │ Done │  ✓    │  │
│   │ tutorial_python.mkv          │ Queue│  …    │  │
│   └──────────────────────────────┴──────┴───────┘  │
│                                                     │
│   [ Open output folder ]                            │
└─────────────────────────────────────────────────────┘
```

### Settings Panel
- Whisper model selection (base / medium / large-v3)
- Ollama model selection (populated by querying `ollama list`)
- Output language (auto-detect or forced)
- Output format (Markdown / PDF / plain text)
- Prompt style (Study Notes / Executive Summary / Lecture Notes)

---

## Technology Stack

| Layer          | Technology                          |
|----------------|-------------------------------------|
| Backend        | Python 3.11 · FastAPI · Uvicorn     |
| Task queue     | Python `asyncio` + SQLite           |
| Transcription  | faster-whisper (CTranslate2 + CUDA) |
| Audio extract  | FFmpeg (sidecar)                    |
| LLM inference  | Ollama (local HTTP API)             |
| Frontend       | Vanilla HTML/CSS/JS (single file)   |
| Output         | Markdown → optionally PDF via `weasyprint` |

---

## Project Structure

```
VideoDeck/
├── main.py                  # FastAPI app entry point
├── pipeline/
│   ├── ingest.py            # FFmpeg audio extraction
│   ├── transcribe.py        # faster-whisper wrapper
│   └── summarize.py         # Ollama API client + prompt builder
├── db/
│   └── jobs.py              # SQLite job queue (aiosqlite)
├── ui/
│   └── index.html           # Single-file minimal frontend
├── prompts/
│   └── study_notes.txt      # LLM prompt templates
├── output/                  # Generated transcripts + summaries
├── requirements.txt
└── DESIGN.md
```

---

## Data Flow (End-to-End Example)

```
User drops "lecture.mp4"
        │
        ▼
FFmpeg extracts audio → lecture_audio.wav (16 kHz mono)
        │
        ▼
faster-whisper (GPU, large-v3)
  → lecture_transcript.json   (segments with timestamps)
  → lecture_transcript.txt    (plain text)
        │
        ▼
Ollama (mistral:7b) receives:
  SYSTEM: "You are an expert study assistant..."
  USER:   "<full transcript>"
  → lecture_summary.md
        │
        ▼
UI shows "Done ✓" — user clicks to open lecture_summary.md
```

---

## GPU Memory Budget (RTX 4070 Ti Super — 16 GB)

```
faster-whisper large-v3  ≈  10 GB  (loaded only during transcription)
Ollama mistral:7b Q4     ≈   5 GB  (loaded only during summarization)
OS + CUDA overhead       ≈   1 GB
─────────────────────────────────
Peak simultaneous        ≈  16 GB  (pipeline is sequential, not parallel)
```

Because transcription and summarization run sequentially, both models
fit within the 16 GB VRAM budget without offloading to system RAM.

---

## Performance Estimates (RTX 4070 Ti Super)

| Content              | Duration | Transcription | Summarization | Total   |
|----------------------|----------|---------------|---------------|---------|
| Short lecture        | 10 min   | ~25 sec       | ~20 sec       | ~45 sec |
| University lecture   | 90 min   | ~3.5 min      | ~45 sec       | ~4 min  |
| Full-day workshop    | 8 hrs    | ~20 min       | ~2 min        | ~22 min |

---

## Security & Privacy

- All processing is **100% local** — no audio, transcript, or summary ever leaves the machine.
- No internet connection is required after initial model downloads.
- Temporary audio files are deleted after transcription completes.

---

## Getting Started (Quick Start)

```bash
# 1. Clone the repo
git clone https://github.com/tf-gmail/VideoDeck.git
cd VideoDeck

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install and start Ollama, then pull a model
ollama pull mistral:7b

# 5. Start VideoDeck
python main.py
# → Open http://localhost:8000 in your browser
```

---

## Future Enhancements (Backlog)

- Speaker diarization (who said what) via `pyannote.audio`
- Chapter markers exported to `.m4v` metadata
- Flashcard export (Anki `.apkg` format)
- Batch processing of entire course folders
- Vector search across all past transcripts (local RAG)
