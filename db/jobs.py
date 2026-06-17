"""
db/jobs.py — SQLite-backed job queue using aiosqlite.
"""

from __future__ import annotations

import time
import aiosqlite
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "videodeck.db"


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'queued',
                progress    INTEGER NOT NULL DEFAULT 0,
                stage       TEXT NOT NULL DEFAULT '',
                error       TEXT,
                transcript  TEXT,
                summary     TEXT,
                whisper_model TEXT,
                llm_model   TEXT,
                prompt_style TEXT,
                language    TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
            """
        )
        await _ensure_columns(
            db,
            "jobs",
            {
                "whisper_model": "TEXT",
                "llm_model": "TEXT",
                "prompt_style": "TEXT",
                "language": "TEXT",
            },
        )
        await db.commit()


async def _ensure_columns(
    db: aiosqlite.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing: set[str] = set()
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        async for row in cur:
            existing.add(row[1])

    for name, col_type in columns.items():
        if name not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


async def create_job(
    job_id: str,
    filename: str,
    file_path: str,
    whisper_model: str,
    llm_model: str,
    prompt_style: str,
    language: str,
) -> dict[str, Any]:
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO jobs (
                id, filename, file_path, status, progress, stage,
                whisper_model, llm_model, prompt_style, language,
                created_at, updated_at
            )
            VALUES (?, ?, ?, 'queued', 0, 'queued', ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                filename,
                file_path,
                whisper_model,
                llm_model,
                prompt_style,
                language,
                now,
                now,
            ),
        )
        await db.commit()
    return await get_job(job_id)


async def get_job(job_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def list_jobs() -> list[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def update_job(job_id: str, **kwargs: Any) -> None:
    kwargs["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [job_id]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", values)
        await db.commit()


async def delete_job(job_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await db.commit()
