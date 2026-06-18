"""
pipeline/summarize.py — Send a transcript to a local Ollama model and
return structured study material.
"""

from __future__ import annotations

from pathlib import Path
import logging
import json
import re
from typing import Awaitable, Callable

import httpx

OLLAMA_BASE = "http://localhost:11434"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_PROMPT_CACHE: dict[str, str] = {}
MIN_SUMMARY_LEN = 180
TARGET_SECTION_COUNT = 50
MIN_SECTION_COUNT = 45
MAX_SECTION_COUNT = 55
MIN_CHUNK_SIZE = 2_200
MAX_CHUNK_SIZE = 4_200
CHUNK_OVERLAP = 280
ProgressCallback = Callable[[int, str], Awaitable[None]]

logger = logging.getLogger(__name__)


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
    progress_cb: ProgressCallback | None = None,
    output_language: str | None = None,
) -> str:
    """
    Send *transcript* to Ollama and return the generated study material.

    The prompt template at prompts/<prompt_style>.txt is used.
    The placeholder {{TRANSCRIPT}} in the template is replaced with
    the actual transcript text.
    """
    cleaned = transcript.strip()
    if not cleaned:
        raise ValueError("Transcript is empty; cannot summarize")

    async with httpx.AsyncClient(timeout=600) as client:
        system_prompt = _load_prompt(prompt_style)
        language_instruction = _language_instruction(output_language)
        if language_instruction:
            system_prompt = f"{system_prompt}\n\n{language_instruction}"
        if progress_cb:
            await progress_cb(5, "Preparing summarization…")
        logger.info(
            "Summarization started (model=%s, transcript_chars=%d, prompt=%s, language=%s)",
            model,
            len(cleaned),
            prompt_style,
            output_language or "auto",
        )

        chunk_size = _choose_chunk_size(cleaned)
        chunks = _split_text(cleaned, chunk_size, CHUNK_OVERLAP)
        total_chunks = max(1, len(chunks))
        section_summaries: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            if progress_cb:
                pct = 5 + int((idx - 1) * 90 / total_chunks)
                await progress_cb(pct, f"Summarizing section {idx}/{total_chunks}…")

            chunk_for_summary = _clean_section_text(chunk)
            section_summary = await _summarize_section(
                client=client,
                model=model,
                system_prompt=system_prompt,
                chunk=chunk_for_summary,
                section_index=idx,
                total_sections=total_chunks,
                output_language=output_language,
            )
            section_summaries.append(section_summary.strip())

        summary = "\n\n---\n\n".join(section_summaries)
        if progress_cb:
            await progress_cb(100, "Summary generated")
        return summary


async def _chat(
    client: httpx.AsyncClient,
    model: str,
    system: str,
    user: str,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "num_ctx": 32768,
            "temperature": temperature,
        },
    }
    logger.debug("Ollama chat request: model=%s, user_chars=%d", model, len(user))
    resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _choose_chunk_size(text: str) -> int:
    n = max(1, len(text))
    desired = max(1, n // TARGET_SECTION_COUNT)
    size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, desired))

    # Keep expected section count in the preferred range (16-20) when possible.
    estimated = (n + max(1, size - CHUNK_OVERLAP) - 1) // max(1, size - CHUNK_OVERLAP)
    if estimated < MIN_SECTION_COUNT:
        size = max(MIN_CHUNK_SIZE, n // MIN_SECTION_COUNT)
    elif estimated > MAX_SECTION_COUNT:
        size = min(MAX_CHUNK_SIZE, max(MIN_CHUNK_SIZE, n // MAX_SECTION_COUNT))
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, size))


def _clean_section_text(text: str) -> str:
    filler_words = {
        "aeh", "aehm", "hm", "hmm", "mhm", "so", "also", "ja", "ne", "ok", "okay",
    }
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kept: list[str] = []
    for line in lines:
        normalized = _normalize_text(line)
        tokens = normalized.split()
        if not tokens:
            continue

        # Drop lines that are mostly filler/interjections from noisy transcripts.
        content_tokens = [t for t in tokens if t not in filler_words]
        if len(tokens) <= 4 and len(content_tokens) <= 1:
            continue
        if len(content_tokens) == 0:
            continue

        kept.append(line)

    cleaned = "\n".join(kept).strip()
    return cleaned or text


def _language_instruction(output_language: str | None) -> str:
    if not output_language:
        return ""
    lang = output_language.lower().strip()
    if lang.startswith("de"):
        return "IMPORTANT: Write the entire response in German (Deutsch)."
    if lang.startswith("en"):
        return "IMPORTANT: Write the entire response in English."
    return ""


async def _summarize_section(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    chunk: str,
    section_index: int,
    total_sections: int,
    output_language: str | None,
) -> str:
    section_label = f"Abschnitt {section_index}/{total_sections}"
    prompt = (
        f"You are summarizing one contiguous lecture section for study notes. "
        f"Write in the language requested by the system prompt. "
        f"Focus only on this section and do not reference other sections.\n\n"
        f"REQUIREMENTS:\n"
        f"- Produce a mini study note for {section_label}.\n"
        f"- Keep all important details, examples, definitions, numbers, names, warnings, and comparisons.\n"
        f"- Use a compact but complete structure:\n"
        f"  ## {section_label}\n"
        f"  ### TL;DR\n"
        f"  2-3 sentences.\n"
        f"  ### Key Points\n"
        f"  5-10 bullets.\n"
        f"  ### Topic Breakdown\n"
        f"  2-4 short paragraphs or bullets.\n"
        f"  ### Glossary\n"
        f"  3-8 terms if present.\n"
        f"  ### Quiz Questions\n"
        f"  2-4 questions with answers.\n"
        f"- Do not invent facts.\n"
        f"- Keep the section self-contained and detailed.\n"
    )
    logger.info("Sending section summary request %s to Ollama (model=%s)", section_label, model)
    summary = await _chat(
        client,
        model=model,
        system=system_prompt,
        user=f"{prompt}\n<TRANSCRIPT_SECTION>\n{chunk}\n</TRANSCRIPT_SECTION>",
        temperature=0.2,
    )
    return summary


async def _extract_key_terms(client: httpx.AsyncClient, model: str, text: str) -> list[str]:
    sample = text[:45_000]
    raw = await _chat(
        client,
        model=model,
        system=(
            "Extract exam-relevant key terms from transcripts. Return ONLY a JSON array of strings,"
            " max 20 terms, no prose."
        ),
        user=f"<TEXT>\n{sample}\n</TEXT>",
        temperature=0.0,
    )

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()][:20]
    except Exception:
        pass

    # Fallback if model did not return clean JSON.
    terms = [m.strip() for m in re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\- ]{4,}", raw)]
    deduped: list[str] = []
    seen: set[str] = set()
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(t)
        if len(deduped) >= 20:
            break
    return deduped


async def _extract_missing_terms(
    client: httpx.AsyncClient,
    model: str,
    transcript: str,
    summary: str,
) -> list[str]:
    transcript_terms = await _extract_key_terms(client, model, transcript)
    return [term for term in transcript_terms if term and not _term_covered(term, summary)]


async def _find_missing_coverage(
    client: httpx.AsyncClient,
    model: str,
    summary: str,
    coverage_items: list[str],
) -> list[str]:
    if not coverage_items:
        return []

    raw = await _chat(
        client,
        model=model,
        system=(
            "You check if a summary covers required checklist items. Return ONLY a JSON array of "
            "missing checklist item strings. If nothing is missing, return []."
        ),
        user=(
            "Checklist:\n"
            + "\n".join(f"- {item}" for item in coverage_items)
            + "\n\nSummary:\n"
            + summary
        ),
        temperature=0.0,
    )

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Fallback heuristic: normalized string / token overlap check.
    missing = []
    for item in coverage_items:
        anchor = item.split(":", 1)[-1].strip().lower()
        if anchor and not _term_covered(anchor, summary):
            missing.append(item)
    return missing


def _term_covered(term: str, text: str) -> bool:
    normalized_term = _normalize_text(term)
    normalized_text = _normalize_text(text)
    if not normalized_term:
        return True
    if normalized_term in normalized_text:
        return True

    term_tokens = _meaningful_tokens(term)
    if not term_tokens:
        return False

    text_tokens = set(_meaningful_tokens(text))
    if not text_tokens:
        return False

    overlap = sum(1 for token in term_tokens if token in text_tokens)
    if len(term_tokens) == 1:
        token = term_tokens[0]
        return any(token in candidate or candidate in token for candidate in text_tokens)

    # Treat the term as covered when most meaningful tokens are present.
    required = max(2, (len(term_tokens) + 1) // 2)
    return overlap >= required


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _meaningful_tokens(text: str) -> list[str]:
    stopwords = {
        "der", "die", "das", "und", "oder", "von", "im", "in", "am", "an", "zu",
        "the", "a", "an", "of", "for", "to", "and", "or", "with", "is", "are", "be",
    }
    tokens = [t for t in _normalize_text(text).split() if len(t) > 2 and t not in stopwords]
    return tokens
