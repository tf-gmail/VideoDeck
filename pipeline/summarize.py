"""
pipeline/summarize.py — Send a transcript to a local Ollama model and
return structured study material.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

OLLAMA_BASE = "http://localhost:11434"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt(name: str) -> str:
    if name not in _PROMPT_CACHE:
        path = PROMPTS_DIR / f"{name}.txt"
        _PROMPT_CACHE[name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


async def list_models() -> list[str]:
    """Return model names available in the local Ollama instance."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


async def summarize(
    transcript: str,
    model: str = "qwen2.5-coder:14b",
    prompt_style: str = "study_notes",
) -> str:
    """
    Send *transcript* to Ollama and return the generated study material.

    The prompt template at prompts/<prompt_style>.txt is used.
    The placeholder {{TRANSCRIPT}} in the template is replaced with
    the actual transcript text.
    """
    system_prompt = _load_prompt(prompt_style)
    user_message = f"<TRANSCRIPT>\n{transcript}\n</TRANSCRIPT>"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "num_ctx": 32768,    # large context for long transcripts
            "temperature": 0.2,  # low temperature → factual, consistent output
        },
    }

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
