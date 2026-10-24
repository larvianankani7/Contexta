from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

from backend.app.config import Settings

logger = logging.getLogger(__name__)


class MemoryRecord(TypedDict):
    id: str
    content: str
    facts: list[dict[str, str]]
    source: str | None
    metadata: dict[str, str | int | float | bool | None]
    artifact_path: str
    metadata_path: str
    created_at: str


class Database:
    def __init__(self, settings: Settings) -> None:
        self.path = settings.project_path(settings.database_path)
        self.artifact_dir = settings.project_path(settings.artifact_dir)
        self.fts_enabled = False

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    facts_json TEXT NOT NULL,
                    source TEXT,
                    metadata_json TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS memories_created_at ON memories(created_at)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    query TEXT NOT NULL,
                    retrieval_count INTEGER NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    verification_json TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    retrieval_ms REAL NOT NULL,
                    inference_ms REAL NOT NULL,
                    total_ms REAL NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    query_hash TEXT
                )
                """
            )
            trace_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(execution_traces)")
            }
            if "prompt_tokens" not in trace_columns:
                connection.execute("ALTER TABLE execution_traces ADD COLUMN prompt_tokens INTEGER")
            if "completion_tokens" not in trace_columns:
                connection.execute("ALTER TABLE execution_traces ADD COLUMN completion_tokens INTEGER")
            if "query_hash" not in trace_columns:
                connection.execute("ALTER TABLE execution_traces ADD COLUMN query_hash TEXT")
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        memory_id UNINDEXED, content, facts
                    )
                    """
                )
                self.fts_enabled = True
                fts_count = connection.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
                memory_count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                if fts_count == 0 and memory_count:
                    connection.execute(
                        """
                        INSERT INTO memory_fts(memory_id, content, facts)
                        SELECT id, content, facts_json FROM memories
                        """
                    )
            except sqlite3.OperationalError as error:
                logger.warning("SQLite FTS5 unavailable; using LIKE retrieval: %s", error)

    def save(
        self,
        *,
        memory_id: str,
        content: str,
        facts: list[dict[str, str]],
        source: str | None,
        metadata: dict[str, str | int | float | bool | None],
        created_at: str,
        png_bytes: bytes,
        artifact_metadata: dict[str, object],
    ) -> tuple[Path, Path]:
        image_path = self.artifact_dir / f"{memory_id}.png"
        metadata_path = self.artifact_dir / f"{memory_id}.json"
        image_temp = image_path.with_suffix(".png.tmp")
        metadata_temp = metadata_path.with_suffix(".json.tmp")
        try:
            image_temp.write_bytes(png_bytes)
            metadata_temp.write_text(
                json.dumps(artifact_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            image_temp.replace(image_path)
            metadata_temp.replace(metadata_path)
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO memories
                    (id, content, facts_json, source, metadata_json, artifact_path, metadata_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        content,
                        json.dumps(facts, ensure_ascii=False),
                        source,
                        json.dumps(metadata, ensure_ascii=False),
                        str(image_path),
                        str(metadata_path),
                        created_at,
                    ),
                )
                if self.fts_enabled:
                    connection.execute(
                        "INSERT INTO memory_fts(memory_id, content, facts) VALUES (?, ?, ?)",
                        (memory_id, content, " ".join(fact["text"] for fact in facts)),
                    )
        except Exception:
            image_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
        return image_path, metadata_path

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._decode(row) if row else None

    def list(self, *, limit: int, offset: int) -> tuple[list[MemoryRecord], int]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return [self._decode(row) for row in rows], int(total)

    def search_fts(self, expression: str, *, limit: int) -> list[MemoryRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT memories.*
                FROM memory_fts JOIN memories ON memories.id = memory_fts.memory_id
                WHERE memory_fts MATCH ?
                ORDER BY bm25(memory_fts)
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def search_like(self, terms: list[str], *, limit: int) -> list[MemoryRecord]:
        clauses = " OR ".join("(content LIKE ? OR facts_json LIKE ?)" for _ in terms)
        parameters = [value for term in terms for value in (f"%{term}%", f"%{term}%")]
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM memories WHERE {clauses} ORDER BY created_at DESC LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def save_trace(self, trace: dict[str, object]) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO execution_traces
                (created_at, query, retrieval_count, evidence_ids_json, verification_json,
                 retry_count, status, retrieval_ms, inference_ms, total_ms, prompt_tokens,
                 completion_tokens, query_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace["created_at"],
                    trace["query"],
                    trace["retrieval_count"],
                    json.dumps(trace["evidence_ids"]),
                    json.dumps(trace["verification"]),
                    trace["retry_count"],
                    trace["status"],
                    trace["retrieval_ms"],
                    trace["inference_ms"],
                    trace["total_ms"],
                    trace.get("prompt_tokens"),
                    trace.get("completion_tokens"),
                    trace.get("query_hash"),
                ),
            )

    def latest_trace(self) -> dict[str, object] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM execution_traces ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        trace = dict(row)
        trace["evidence_ids"] = json.loads(str(trace.pop("evidence_ids_json")))
        trace["verification"] = json.loads(str(trace.pop("verification_json")))
        return trace

    @staticmethod
    def _decode(row: sqlite3.Row) -> MemoryRecord:
        return {
            "id": row["id"],
            "content": row["content"],
            "facts": json.loads(str(row["facts_json"])),
            "source": row["source"],
            "metadata": json.loads(str(row["metadata_json"])),
            "artifact_path": row["artifact_path"],
            "metadata_path": row["metadata_path"],
            "created_at": row["created_at"],
        }
