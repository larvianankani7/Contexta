
import re
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.conversation import _compact, _is_save
from backend.app.schemas import Evidence

router = APIRouter()

PROJECT_CODE = re.compile(
    r"\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+\b"
)

PROJECT_NAME = re.compile(
    r"\b(?:for|in|on)\s+"
    r"(?:(?:my|our|the)\s+)?"
    r"(?:project\s+)?"
    r"([A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,5})",
    re.I,
)

DECISION = re.compile(
    r"\b(?:"
    r"i(?:'ve| have)\s+(?:selected|chosen|decided|picked|finalized)|"
    r"we(?:'ve| have)\s+(?:selected|chosen|decided|picked|finalized)|"
    r"i(?:'m| am)\s+(?:using|going to use|gonna use)|"
    r"we(?:'re| are)\s+(?:using|going to use|gonna use)|"
    r"i\s+(?:will use|chose|selected|decided|use)|"
    r"we\s+(?:will use|chose|selected|decided|use)|"
    r"(?:my|our)\s+.{1,80}?\s+(?:uses|will use|runs on)|"
    r"(?:final|confirmed|locked)\s+"
    r"(?:architecture\s+)?(?:decisions?|choices?|settings?)|"
    r"(?:decisions?|choices?|settings?)\s+"
    r"(?:are|were)\s+(?:final|confirmed|locked)"
    r")\b",
    re.I,
)

UNCERTAIN = re.compile(
    r"\b(?:maybe|might|could|possibly|perhaps|"
    r"considering|hypothetical|suppose|imagine|"
    r"what if|should we|instead of)\b",
    re.I,
)

SECRET = re.compile(
    r"\b(?:password|passphrase|api[_ -]?key|"
    r"private[_ -]?key|secret|otp)\s*[:=]",
    re.I,
)

FIELD_TERMS = {
    "database": r"\b(?:database|db)\b",
    "backend port": r"\b(?:backend|service|internal)\s+port\b|\bport\b",
    "cache ttl": r"\b(?:cache|redis|ttl)\b",
    "access token expiration": r"\b(?:access[- ]token|jwt)\b",
    "refresh token expiration": r"\brefresh[- ]token\b",
    "deployment region": r"\b(?:deployment|region|deploy)\b",
    "password hashing": r"\b(?:hashing|hash algorithm)\b",
    "message queue": r"\b(?:message queue|broker)\b",
    "backup schedule": r"\b(?:backup|backups)\b",
    "encryption": r"\bencryption\b",
}


class ObserveInput(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    project: str | None = Field(default=None, max_length=100)
    source: str = Field(default="mcp:user", max_length=200)


class RecallInput(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    project: str | None = Field(default=None, max_length=100)


def project_name(text: str) -> str | None:
    code = PROJECT_CODE.search(text)
    if code:
        return code.group()

    match = PROJECT_NAME.search(text)
    if not match:
        return None

    name = re.split(
        r"\s+(?:uses|will|is|has|and|but|with|"
        r"which|that|where|we're|we've)\b",
        match.group(1),
        maxsplit=1,
        flags=re.I,
    )[0].strip(" .?!,;:")

    if name.casefold() in {
        "project", "application", "app", "system",
        "portal", "website",
    }:
        return None

    return name[:100] or None


def should_save(
    message: str,
    project: str | None,
) -> tuple[bool, str]:
    text = message.replace("\u2019", "'").strip()

    if SECRET.search(text):
        return False, "sensitive_information"

    if text.endswith("?"):
        return False, "question"

    if UNCERTAIN.search(text):
        return False, "unconfirmed_decision"

    if _is_save(text):
        return True, "explicit_memory"

    if not project:
        return False, "no_identifiable_project"

    if DECISION.search(text):
        return True, "confirmed_project_decision"

    return False, "not_a_confirmed_decision"


@router.post("/memory/observe")
async def observe(
    payload: ObserveInput,
    request: Request,
) -> dict:
    started = time.perf_counter()
    message = payload.message.strip()
    project = (
        payload.project or project_name(message) or ""
    ).strip()

    save, reason = should_save(message, project or None)

    if not save:
        return {
            "saved": False,
            "reason": reason,
            "project": project or None,
            "memory_id": None,
            "inference_ms": 0.0,
        }

    compact = (
        _compact(message)
        if reason == "explicit_memory"
        else message
    )

    content = (
        f"Project: {project}\n{compact}"
        if project else compact
    )

    database = request.app.state.database

    try:
        with database.connection() as connection:
            existing = connection.execute(
                "SELECT id FROM memories "
                "WHERE content = ? LIMIT 1",
                (content,),
            ).fetchone()

        if existing is not None:
            return {
                "saved": False,
                "reason": "already_stored",
                "project": project or None,
                "memory_id": str(existing["id"]),
                "inference_ms": 0.0,
            }

        record = await request.app.state.processor.ingest(
            text=content,
            source=payload.source,
            metadata={
                "kind": reason,
                "project": project,
            },
            extract=False,
        )
    except (sqlite3.Error, OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail="Memory could not be saved",
        ) from error

    return {
        "saved": True,
        "reason": reason,
        "project": project or None,
        "memory_id": str(record["id"]),
        "inference_ms": 0.0,
        "total_ms": round(
            (time.perf_counter() - started) * 1000, 3
        ),
    }


@router.post("/memory/recall")
def recall(
    payload: RecallInput,
    request: Request,
) -> dict:
    started = time.perf_counter()
    project = (
        payload.project or project_name(payload.query) or ""
    ).strip()

    if not project:
        return {
            "status": "needs_project",
            "answer": "Specify which project to search.",
            "evidence": [],
            "inference_ms": 0.0,
        }

    database = request.app.state.database
    matches = []
    offset = 0

    try:
        while len(matches) < 30:
            records, total = database.list(
                limit=100,
                offset=offset,
            )

            for record in records:
                stored_project = str(
                    record["metadata"].get("project") or ""
                )
                content = str(record["content"])

                if (
                    stored_project.casefold() == project.casefold()
                    or content.casefold().startswith(
                        f"project: {project}\n".casefold()
                    )
                ):
                    matches.append(record)

            offset += len(records)

            if not records or offset >= total:
                break
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Memory retrieval failed",
        ) from error

    if not matches:
        return {
            "status": "insufficient_evidence",
            "answer": "No matching saved project decisions found.",
            "evidence": [],
            "inference_ms": 0.0,
        }

    evidence = [
        Evidence(
            memory_id=str(record["id"]),
            text=str(record["content"]),
            source=record["source"],
            relevance=1.0,
        )
        for record in matches[:20]
    ]

    requested = [
        field
        for field, pattern in FIELD_TERMS.items()
        if re.search(pattern, payload.query, re.I)
    ]

    relevant = []
    missing = []

    for field in requested:
        pattern = FIELD_TERMS[field]
        found = [
            item for item in evidence
            if re.search(pattern, item.text, re.I)
        ]

        if found:
            for item in found:
                if item.memory_id not in {
                    value.memory_id for value in relevant
                }:
                    relevant.append(item)
        else:
            missing.append(field)

    broad = bool(re.search(
        r"\b(?:all|every|full|complete|decisions)\b",
        payload.query,
        re.I,
    ))

    if not requested and broad:
        relevant = evidence[:10]

    if not requested and not broad:
        relevant = evidence[:10]

    if not relevant:
        status = "insufficient_evidence"
        answer = "No saved evidence supports the requested fields."
    else:
        status = (
            "partial" if missing
            else "source_backed_excerpts"
        )
        answer = "\n".join(
            f"[{item.memory_id}] {item.text}"
            for item in relevant[:10]
        )
        if missing:
            answer += "\nNot found: " + ", ".join(missing)

    return {
        "status": status,
        "project": project,
        "answer": answer,
        "evidence": [
            item.model_dump() for item in relevant[:10]
        ],
        "missing_fields": missing,
        "inference_ms": 0.0,
        "total_ms": round(
            (time.perf_counter() - started) * 1000, 3
        ),
    }
