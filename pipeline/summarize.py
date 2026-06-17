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
DIRECT_TRANSCRIPT_LIMIT = 70_000
CHUNK_SIZE = 18_000
CHUNK_OVERLAP = 1_000
OUTLINE_CHUNK_SIZE = 12_000
OUTLINE_OVERLAP = 800
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

        content_map = await _build_content_map(client, model, cleaned, progress_cb)

        # For very long transcripts, summarize in chunks first, then synthesize.
        source_for_final = cleaned
        coverage_items: list[str] = []
        if len(cleaned) > DIRECT_TRANSCRIPT_LIMIT:
            chunk_summaries = []
            chunks = _split_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
            total_chunks = max(1, len(chunks))
            for idx, chunk in enumerate(chunks, start=1):
                chunk_summary = await _chat(
                    client,
                    model=model,
                    system=(
                        "You summarize transcript chunks. Return concise bullets of key facts,"
                        " concepts, and decisions. Do not add facts not in the text."
                    ),
                    user=(
                        f"Chunk {idx}:\n<TRANSCRIPT_CHUNK>\n{chunk}\n</TRANSCRIPT_CHUNK>\n"
                        "Return 8-15 bullet points."
                    ),
                    temperature=0.1,
                )
                if chunk_summary.strip():
                    chunk_summaries.append(f"## Chunk {idx}\n{chunk_summary.strip()}")
                    coverage_items.append(f"Chunk {idx}: {_anchor_from_chunk_summary(chunk_summary)}")
                if progress_cb:
                    pct = 35 + int((idx / total_chunks) * 35)
                    await progress_cb(min(70, pct), f"Chunk summarization {idx}/{total_chunks}…")

            if chunk_summaries:
                source_for_final = "\n\n".join(chunk_summaries)
        elif progress_cb:
            await progress_cb(35, "Generating summary…")

        if progress_cb:
            await progress_cb(75, "Final summary synthesis…")
        coverage_directive = ""
        if coverage_items:
            checklist = "\n".join(f"- {item}" for item in coverage_items)
            coverage_directive = (
                "MANDATORY COVERAGE CHECKLIST:\n"
                "The final study notes must include all major content represented by every checklist "
                "item below, in order. Do not skip any item.\n"
                f"{checklist}\n\n"
            )
        logger.info("Sending first summary request to Ollama (model=%s)", model)
        summary = await _chat(
            client,
            model=model,
            system=system_prompt,
            user=(
                f"{coverage_directive}"
                f"CONTENT MAP:\n{content_map}\n\n"
                f"<TRANSCRIPT>\n{source_for_final}\n</TRANSCRIPT>"
            ),
            temperature=0.2,
        )

        # Retry once if model returned an unhelpfully short answer.
        if len(summary.strip()) < MIN_SUMMARY_LEN:
            if progress_cb:
                await progress_cb(92, "Refining short summary…")
            summary = await _chat(
                client,
                model=model,
                system=system_prompt,
                user=(
                    "The previous answer was too short. Produce the full required structure "
                    "with substantial detail.\n"
                    f"<TRANSCRIPT>\n{source_for_final}\n</TRANSCRIPT>"
                ),
                temperature=0.2,
            )

        if len(summary.strip()) < MIN_SUMMARY_LEN:
            raise RuntimeError(
                "LLM returned an unexpectedly short summary. "
                "Try a stronger general-purpose model (e.g. qwen2.5:14b or mistral:7b)."
            )

        # Coverage pass: if important terms are present in transcript but missing in summary,
        # ask the model to integrate them explicitly.
        key_terms = await _extract_key_terms(client, model, source_for_final)
        missing_terms = [t for t in key_terms if t and t.lower() not in summary.lower()]
        if missing_terms:
            if progress_cb:
                await progress_cb(96, "Integrating missing key terms…")
            logger.info("Refining summary with missing key terms: %s", ", ".join(missing_terms[:12]))
            summary = await _chat(
                client,
                model=model,
                system=system_prompt,
                user=(
                    "Revise the summary below and ensure all listed key terms are explicitly "
                    "covered at least once where contextually relevant.\n\n"
                    f"Missing key terms: {', '.join(missing_terms[:12])}\n\n"
                    f"Current summary draft:\n{summary}"
                ),
                temperature=0.2,
            )

        if coverage_items:
            missing_coverage = await _find_missing_coverage(client, model, summary, coverage_items)
            if missing_coverage:
                if progress_cb:
                    await progress_cb(98, "Adding missing section coverage…")
                logger.info("Refining summary with missing coverage items: %s", ", ".join(missing_coverage[:16]))
                summary = await _chat(
                    client,
                    model=model,
                    system=system_prompt,
                    user=(
                        "Revise the summary so that each missing checklist item is explicitly covered "
                        "in the Topic Breakdown and reflected in Key Points. Preserve the required markdown structure.\n\n"
                        "Missing checklist items:\n"
                        + "\n".join(f"- {item}" for item in missing_coverage[:16])
                        + "\n\nCurrent summary draft:\n"
                        + summary
                    ),
                    temperature=0.2,
                )

        final_missing_terms = await _extract_missing_terms(client, model, source_for_final, summary)
        final_missing_coverage = await _find_missing_coverage(client, model, summary, coverage_items) if coverage_items else []
        if final_missing_terms or final_missing_coverage:
            logger.warning(
                "Summary still missing coverage after repair; missing_terms=%s missing_coverage=%s",
                ", ".join(final_missing_terms[:12]),
                ", ".join(final_missing_coverage[:16]),
            )
            if progress_cb:
                await progress_cb(99, "Final quality repair…")
            summary = await _chat(
                client,
                model=model,
                system=(
                    system_prompt
                    + "\n\nFINAL QUALITY GATE: You must rewrite the entire response from scratch. "
                    + "The new answer must include every required topic, key term, and checklist item. "
                    + "Do not preserve omissions from the draft."
                ),
                user=(
                    "Required structure: TL;DR, Key Points, Topic Breakdown, Glossary, Quiz Questions.\n\n"
                    f"CONTENT MAP:\n{content_map}\n\n"
                    f"Missing key terms: {', '.join(final_missing_terms[:12]) or 'none'}\n"
                    f"Missing coverage items: {', '.join(final_missing_coverage[:16]) or 'none'}\n\n"
                    f"Current draft to improve or replace:\n{summary}"
                ),
                temperature=0.15,
            )

            final_missing_terms = await _extract_missing_terms(client, model, source_for_final, summary)
            final_missing_coverage = await _find_missing_coverage(client, model, summary, coverage_items) if coverage_items else []
            if final_missing_terms or final_missing_coverage:
                raise RuntimeError(
                    "Summary quality gate failed after repair. "
                    f"Missing terms: {', '.join(final_missing_terms[:12]) or 'none'}. "
                    f"Missing coverage: {', '.join(final_missing_coverage[:16]) or 'none'}."
                )

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


def _language_instruction(output_language: str | None) -> str:
    if not output_language:
        return ""
    lang = output_language.lower().strip()
    if lang.startswith("de"):
        return "IMPORTANT: Write the entire response in German (Deutsch)."
    if lang.startswith("en"):
        return "IMPORTANT: Write the entire response in English."
    return ""


def _anchor_from_chunk_summary(chunk_summary: str) -> str:
    for line in chunk_summary.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-*").strip()
        if line:
            return line[:140]
    return "major points from this section"


async def _build_content_map(
    client: httpx.AsyncClient,
    model: str,
    transcript: str,
    progress_cb: ProgressCallback | None,
) -> str:
    chunks = _split_text(transcript, OUTLINE_CHUNK_SIZE, OUTLINE_OVERLAP)
    total_chunks = max(1, len(chunks))
    map_parts: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        outline = await _chat(
            client,
            model=model,
            system=(
                "Extract a dense study-note scaffold from this lecture chunk. Return 8-12 bullets only. "
                "Include every distinct topic, example, definition, number, warning, named concept, and comparison. "
                "Do not merge unrelated points or compress away details. Prefer specific facts over generalities."
            ),
            user=(
                f"Chunk {idx} of {total_chunks}:\n"
                f"<TRANSCRIPT_CHUNK>\n{chunk}\n</TRANSCRIPT_CHUNK>"
            ),
            temperature=0.0,
        )
        if outline.strip():
            map_parts.append(
                f"## Chunk {idx}\n"
                f"{outline.strip()}"
            )
        if progress_cb:
            pct = 8 + int((idx / total_chunks) * 22)
            await progress_cb(min(30, pct), f"Building content map {idx}/{total_chunks}…")

    return "\n\n".join(map_parts) if map_parts else transcript[:20_000]


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
    summary_lower = summary.lower()
    return [term for term in transcript_terms if term and term.lower() not in summary_lower]


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

    # Fallback heuristic: basic string containment check.
    lowered_summary = summary.lower()
    missing = []
    for item in coverage_items:
        anchor = item.split(":", 1)[-1].strip().lower()
        if anchor and anchor not in lowered_summary:
            missing.append(item)
    return missing
