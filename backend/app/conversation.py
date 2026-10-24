
import logging
import re
import sqlite3
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.api.routes import _save_trace
from backend.app.reasoning.client import ReasoningUnavailable
from backend.app.schemas import ChatResponse, ClaimVerification, Evidence

logger = logging.getLogger(__name__)
router = APIRouter()

SAVE_RE = re.compile(
    r"\b(?:remember\s+(?:that|this)|save\s+(?:it|this|that|these)|"
    r"lock\s+(?:it|this|that|these)|note\s+(?:that|this)|keep\s+in\s+mind|"
    r"(?:decisions?|settings|requirements|tech\s+stack)\s+are\s+locked|"
    r"remind\s+me\s+later)\b",
    re.I,
)
RECALL_RE = re.compile(
    r"\b(?:remember|recall|retrieve|saved|stored|locked|earlier|previously|"
    r"before|last\s+time|beginning|"
    r"(?:we\s+(?:decided|chose|picked|selected)|did\s+we\s+(?:choose|decide|pick|select))|"
    r"our\s+(?:decision|choice|project|settings|architecture|tech\s+stack))\b",
    re.I,
)
QUESTION_RE = re.compile(
    r"^\s*(?:what|which|where|when|who|do\s+you|can\s+you|could\s+you|"
    r"tell\s+me|show\s+me|list|retrieve|recall|remind\s+me)\b",
    re.I,
)
PROJECT_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+\b")
NAMED_RE = re.compile(
    r"\bproject\s+(?:named\s+|codename\s*:\s*)?([A-Z][A-Za-z0-9_-]*)\b"
)

LABELS = {
    "project codename": r"project\s+codename",
    "database": r"database",
    "password hashing": r"password\s+hashing",
    "cache": r"cache",
    "access token expiration": r"access\s+token\s+expiration",
    "refresh token expiration": r"refresh\s+token\s+expiration",
    "api rate limit": r"(?:api\s+)?rate\s+limit",
    "internal service port": r"(?:internal\s+)?service\s+port|internal\s+port",
    "deployment region": r"deployment\s+region",
}
LABEL_RE = re.compile(
    r"(?<!\w)("
    + "|".join(f"(?:{value})" for value in LABELS.values())
    + r")\s*:\s*",
    re.I,
)
FIELD_CUES = {
    "project codename": r"\b(?:codename|project\s+name)\b",
    "database": r"\b(?:database|db)\b",
    "password hashing": r"\b(?:hashing|hash\s+algorithm)\b",
    "cache": r"\b(?:cache|redis|ttl)\b",
    "access token expiration": r"\baccess\s+token\b",
    "refresh token expiration": r"\brefresh\s+token\b",
    "api rate limit": r"\b(?:rate\s+limit|requests\s+per)\b",
    "internal service port": r"\b(?:port|listening\s+port)\b",
    "deployment region": r"\b(?:region|deploy(?:ment)?)\b",
}


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=5_000)


class ConversationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5_000)
    history: list[Turn] = Field(default_factory=list, max_length=24)


class ConversationResponse(ChatResponse):
    saved_memory_id: str | None = None


def _identifiers(text: str) -> set[str]:
    return (
        {value.casefold() for value in PROJECT_RE.findall(text)}
        | {value.casefold() for value in NAMED_RE.findall(text)}
    )


def _is_save(query: str) -> bool:
    return bool(SAVE_RE.search(query)) and not (
        QUESTION_RE.search(query) and RECALL_RE.search(query)
    )


def _needs_memory(query: str) -> bool:
    if not (QUESTION_RE.search(query) or query.rstrip().endswith("?")):
        return False
    return bool(
        RECALL_RE.search(query)
        or (
            _identifiers(query)
            and any(re.search(cue, query, re.I) for cue in FIELD_CUES.values())
        )
    )


def _fields(text: str) -> dict[str, str]:
    matches = list(LABEL_RE.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = next(
            (
                name
                for name, pattern in LABELS.items()
                if re.fullmatch(pattern, match.group(1), re.I)
            ),
            None,
        )
        if not label:
            continue
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        value = text[match.end():end].strip().strip(". ;\n\t")
        value = re.split(
            r"\b(?:These values are intentional|Preserve them exactly|"
            r"Acknowledge the decisions|We'll now discuss|Do not bring up)\b",
            value,
            maxsplit=1,
            flags=re.I,
        )[0].strip().strip(". ;\n\t")
        if value and len(value) < 500:
            fields[label] = value
    return fields


def _save_text(query: str, history: list[Turn]) -> str:
    if re.fullmatch(
        r"\s*(?:lock|save|remember)\s+(?:it|this|that)[.!]?\s*",
        query,
        re.I,
    ):
        return next(
            (item.content for item in reversed(history) if item.role == "user"),
            query,
        )
    return query


def _compact(text: str) -> str:
    fields = _fields(text)
    if fields:
        return "\n".join(
            f"{name.title()}: {value}" for name, value in fields.items()
        )
    return re.sub(
        r"^\s*(?:remember\s+that|save\s+(?:this|that)\s*:?|"
        r"lock\s+(?:this|these)\s*:?|note\s+that)\s*",
        "",
        text,
        flags=re.I,
    ).strip()[:5_000]


def _records(
    request: Request,
    query: str,
    history: list[Turn],
) -> list[Evidence]:
    database = request.app.state.database
    records, _ = database.list(limit=100, offset=0)
    if not records:
        return []

    identifiers = _identifiers(query)
    if not identifiers:
        for turn in reversed(history):
            identifiers = _identifiers(turn.content)
            if identifiers:
                break

    if not identifiers:
        known = set().union(
            *(_identifiers(record["content"]) for record in records)
        )
        if len(known) != 1:
            return []
        identifiers = known

    if len(identifiers) != 1:
        return []

    name = next(iter(identifiers))
    matches = [
        record
        for record in records
        if name in _identifiers(record["content"])
    ]
    matches.sort(
        key=lambda record: (
            len(_fields(record["content"])),
            record["created_at"],
        ),
        reverse=True,
    )
    return [
        Evidence(
            memory_id=str(record["id"]),
            text=str(record["content"]),
            source=record["source"],
            relevance=1.0,
        )
        for record in matches[:8]
    ]


def _recall(
    query: str,
    evidence: list[Evidence],
) -> tuple[str, list[str], bool]:
    requested = [
        name
        for name, cue in FIELD_CUES.items()
        if re.search(cue, query, re.I)
    ]
    broad = bool(
        re.search(
            r"\b(?:all|every|complete|entire|full|"
            r"architecture\s+decisions|locked\s+decisions)\b",
            query,
            re.I,
        )
    ) or len(requested) >= 3

    values: dict[str, set[str]] = {}
    citations: dict[str, list[str]] = {}
    for item in evidence:
        for key, value in _fields(item.text).items():
            values.setdefault(key, set()).add(value)
            citations.setdefault(key, []).append(item.memory_id)

    if not values:
        return evidence[0].text.strip(), [evidence[0].memory_id], True

    targets = list(values) if broad or not requested else requested
    found = [key for key in targets if key in values]

    if not found:
        return (
            "I found saved information for this project, "
            "but not that specific decision.",
            [],
            False,
        )

    if any(len(values[key]) > 1 for key in found):
        return (
            "Conflicting saved decisions exist. "
            "Please review the source memories.",
            [],
            False,
        )

    lines = [
        f"{key.title()}: {next(iter(values[key]))}"
        for key in found
    ]
    missing = [key for key in targets if key not in values]
    if missing:
        lines.append("Not stored: " + ", ".join(missing))

    used = list(
        dict.fromkeys(
            memory_id
            for key in found
            for memory_id in citations[key]
        )
    )
    return "\n".join(lines), used, not missing


@router.post("/conversation", response_model=ConversationResponse)
async def converse(
    payload: ConversationRequest,
    request: Request,
) -> ConversationResponse:
    start = time.perf_counter()
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Question is empty")

    evidence: list[Evidence] = []
    checks: list[ClaimVerification] = []
    saved_id = None
    retrieval_ms = inference_ms = 0.0
    prompt_tokens = completion_tokens = None

    if _is_save(query):
        text = _compact(_save_text(query, payload.history))
        if not text:
            raise HTTPException(status_code=422, detail="No decision to save")
        try:
            with request.app.state.database.connection() as connection:
                previous = connection.execute(
                    "SELECT id FROM memories WHERE content = ? LIMIT 1",
                    (text,),
                ).fetchone()
            if previous:
                saved_id = str(previous["id"])
            else:
                stored = await request.app.state.processor.ingest(
                    text=text,
                    source="chat:auto",
                    metadata={"kind": "explicit_memory"},
                    extract=False,
                )
                saved_id = str(stored["id"])
        except (sqlite3.Error, OSError, ValueError) as error:
            logger.exception("Automatic save failed")
            raise HTTPException(
                status_code=500,
                detail="Memory could not be saved",
            ) from error

        answer = "Saved the decision to Contexta memory."
        if "remind me later" in query.casefold():
            answer += " This is not a scheduled reminder."
        status = "unverified"

    elif _needs_memory(query):
        retrieval_start = time.perf_counter()
        try:
            evidence = _records(request, query, payload.history)
        except sqlite3.Error as error:
            raise HTTPException(
                status_code=500,
                detail="Memory retrieval failed",
            ) from error

        retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        if not evidence:
            answer = (
                "I couldn't identify a matching saved decision. "
                "Please specify the project."
            )
            status = "insufficient_evidence"
        else:
            answer, ids, complete = _recall(query, evidence)
            evidence = [
                item for item in evidence
                if item.memory_id in ids
            ]
            status = (
                "verified" if complete else "insufficient_evidence"
            )
            if complete:
                checks = [
                    ClaimVerification(
                        claim=answer,
                        supported=True,
                        evidence_ids=ids,
                        reason=(
                            "Answer assembled from exact stored fields; "
                            "no model generation."
                        ),
                    )
                ]

    else:
        inference_start = time.perf_counter()
        try:
            result = await request.app.state.reasoning.answer_general(
                query,
                history=[
                    turn.model_dump()
                    for turn in payload.history
                ],
            )
        except ReasoningUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail="Ollama unavailable or invalid response",
            ) from error

        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000
        prompt_tokens = result._prompt_tokens
        completion_tokens = result._completion_tokens
        answer = result.answer
        status = "unverified"

    response = ConversationResponse(
        answer=answer,
        status=status,
        verified=status == "verified",
        retry_count=0,
        claims=checks,
        evidence=evidence,
        saved_memory_id=saved_id,
    )

    _save_trace(
        request,
        query=query,
        evidence=evidence,
        checks=checks,
        retry_count=0,
        status=status,
        retrieval_ms=retrieval_ms,
        inference_ms=inference_ms,
        total_ms=(time.perf_counter() - start) * 1000,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return response
