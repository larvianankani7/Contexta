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
    r"\b(?:remember\s+(?:that|this|the\s+following)|save\s+(?:this|that|these|the\s+following)|"
    r"lock\s+(?:this|it|these|the\s+following)|note\s+(?:that|this)|keep\s+in\s+mind|"
    r"(?:decisions|settings|requirements|tech\s+stack)\s+are\s+locked|"
    r"(?:we|i)\s+(?:have\s+)?(?:decided|chosen)\s+to|remind\s+me\s+later)\b",
    re.I,
)
RECALL_RE = re.compile(
    r"\b(?:what|which|when|where|who|tell me|show me|list|retrieve|recall|remind me|do you remember)\b"
    r".{0,180}\b(?:we|our|my|earlier|previous|before|last\s+time|beginning|stored|saved|locked|discussed|decided|chose)\b"
    r"|\b(?:our|my)\s+(?:project|database|decisions?|settings|tech\s+stack|preferences?)\b"
    r"|\b(?:earlier|previously|last\s+time|at\s+the\s+beginning)\b",
    re.I,
)
PROJECT_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+\b")
NAMED_PROJECT_RE = re.compile(r"\b[Pp]roject\s+(?:named\s+|codename\s*:\s*)?([A-Z][A-Za-z0-9_-]*)\b")
FIELD_RE = {
    "database": re.compile(r"\b(?:database|db)\b", re.I),
    "password hashing": re.compile(r"\b(?:password.hash|hashing)\b", re.I),
    "cache": re.compile(r"\b(?:cache|caching|ttl)\b", re.I),
    "access token expiration": re.compile(r"\baccess\s+token\b", re.I),
    "refresh token expiration": re.compile(r"\brefresh\s+token\b", re.I),
    "api rate limit": re.compile(r"\b(?:rate\s+limit|requests\s+per)\b", re.I),
    "internal service port": re.compile(r"\b(?:internal\s+port|service\s+port|port\s+number)\b", re.I),
    "deployment region": re.compile(r"\b(?:deployment\s+region|region)\b", re.I),
    "project codename": re.compile(r"\b(?:codename|project\s+name)\b", re.I),
}
ANALYSIS_RE = re.compile(r"\b(?:why|how|compare|difference|recommend|explain|design|advantages|disadvantages)\b", re.I)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=5_000)


class ConversationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5_000)
    history: list[Turn] = Field(default_factory=list, max_length=24)


class ConversationResponse(ChatResponse):
    saved_memory_id: str | None = None


def _needs_memory(query: str) -> bool:
    return bool(RECALL_RE.search(query) or PROJECT_RE.search(query) or NAMED_PROJECT_RE.search(query))


def _is_save(query: str) -> bool:
    if query.strip().endswith("?") and RECALL_RE.search(query):
        return False
    return bool(SAVE_RE.search(query))


def _save_text(query: str, history: list[Turn]) -> str:
    if re.fullmatch(r"(?i)\s*(?:lock|save|remember)\s+(?:it|this|that)[.!]?\s*", query):
        return next((turn.content for turn in reversed(history) if turn.content.strip()), query)
    return query


def _project_fallback(request: Request, query: str, history: list[Turn]) -> list[Evidence]:
    tokens = set(PROJECT_RE.findall(query)) | set(NAMED_PROJECT_RE.findall(query))
    if not tokens:
        for turn in reversed(history):
            tokens = set(PROJECT_RE.findall(turn.content)) | set(NAMED_PROJECT_RE.findall(turn.content))
            if tokens:
                break
    database = request.app.state.database
    records, _ = database.list(limit=50, offset=0)
    if not tokens:
        identities = {match for r in records for match in PROJECT_RE.findall(r["content"])} | {match for r in records for match in NAMED_PROJECT_RE.findall(r["content"])}
        if len(identities) != 1:
            return []
        tokens = identities
    if len(tokens) != 1:
        return []
    identifier = next(iter(tokens)).casefold()
    matching = [r for r in records if identifier in r["content"].casefold()]
    matching.sort(key=lambda r: (len(r["content"]), r["created_at"]), reverse=True)
    return [Evidence(memory_id=str(r["id"]), text=str(r["content"]), source=r["source"], relevance=1.0) for r in matching[:3]]


def _source_answer(query: str, evidence: list[Evidence]) -> tuple[str, list[Evidence], bool]:
    unique: list[Evidence] = []
    seen: set[str] = set()
    for item in evidence:
        key = item.text.strip().casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    primary = unique[0]
    fields = [key for key, pattern in FIELD_RE.items() if pattern.search(query)]
    broad = len(fields) >= 3 or re.search(r"\b(?:all|entire|complete|every|full|architecture\s+decisions)\b", query, re.I)
    if fields and not broad:
        labels = tuple(field.casefold() for field in fields)
        lines = [line.strip() for line in primary.text.splitlines() if line.strip() and ":" in line and line.split(":", 1)[0].strip().casefold() in labels]
        if lines:
            conflicts = False
            for field in fields:
                values = {
                    line.split(":", 1)[1].strip().casefold()
                    for item in unique
                    for line in item.text.splitlines()
                    if ":" in line and line.split(":", 1)[0].strip().casefold() == field
                }
                conflicts |= len(values) > 1
            if conflicts:
                return "Conflicting saved values exist for this decision. Review the cited memories.", evidence, False
            return "\n".join(lines), evidence, True
    if len(unique) > 1 and any(item.text != primary.text and len(item.text) >= len(primary.text) * 0.8 for item in unique[1:]):
        return "Multiple different memories may apply. Please specify which decision to use.", evidence, False
    return primary.text[:5_000], evidence, True


@router.post("/conversation", response_model=ConversationResponse)
async def converse(payload: ConversationRequest, request: Request) -> ConversationResponse:
    start = time.perf_counter()
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Question is empty")
    evidence: list[Evidence] = []
    checks: list[ClaimVerification] = []
    saved_id = None
    retrieval_ms = 0.0
    inference_ms = 0.0
    prompt_tokens = completion_tokens = None
    if _is_save(query):
        memory_text = _save_text(query, payload.history)
        try:
            with request.app.state.database.connection() as connection:
                previous = connection.execute("SELECT id FROM memories WHERE content = ? LIMIT 1", (memory_text,)).fetchone()
            if previous:
                saved_id = str(previous["id"])
            else:
                record = await request.app.state.processor.ingest(text=memory_text, source="chat:auto", metadata={"kind": "explicit_memory"}, extract=False)
                saved_id = str(record["id"])
        except (sqlite3.Error, OSError, ValueError) as error:
            logger.exception("Automatic memory save failed")
            raise HTTPException(status_code=500, detail="Memory could not be saved") from error
        answer = "Saved to Contexta memory." + (" This is stored as a note, not a scheduled reminder." if re.search(r"remind\s+me\s+later", query, re.I) else "")
        status = "unverified"
    elif _needs_memory(query):
        try:
            t = time.perf_counter()
            evidence = request.app.state.retriever.retrieve(query, limit=5)
            if not evidence:
                evidence = _project_fallback(request, query, payload.history)
            retrieval_ms = (time.perf_counter() - t) * 1_000
        except sqlite3.Error as error:
            logger.exception("Memory retrieval failed")
            raise HTTPException(status_code=500, detail="Memory could not be retrieved") from error
        if not evidence:
            answer = "I couldn't find matching saved information. Please specify the project or decision."
            status = "insufficient_evidence"
        elif ANALYSIS_RE.search(query):
            t = time.perf_counter()
            try:
                result = await request.app.state.reasoning.answer_general(query, history=[turn.model_dump() for turn in payload.history], memory=evidence)
            except ReasoningUnavailable as error:
                raise HTTPException(status_code=503, detail="Ollama is unavailable") from error
            inference_ms = (time.perf_counter() - t) * 1_000
            prompt_tokens, completion_tokens = result._prompt_tokens, result._completion_tokens
            answer, status = result.answer, "unverified"
        else:
            answer, evidence, confirmed = _source_answer(query, evidence)
            status = "verified" if confirmed else "insufficient_evidence"
            if confirmed:
                checks = [ClaimVerification(claim=answer, supported=True, evidence_ids=[evidence[0].memory_id], reason="Verbatim excerpt from stored source; no generated claim.")]
    else:
        t = time.perf_counter()
        try:
            result = await request.app.state.reasoning.answer_general(query, history=[turn.model_dump() for turn in payload.history])
        except ReasoningUnavailable as error:
            raise HTTPException(status_code=503, detail="Ollama is unavailable") from error
        inference_ms = (time.perf_counter() - t) * 1_000
        prompt_tokens, completion_tokens = result._prompt_tokens, result._completion_tokens
        answer, status = result.answer, "unverified"
    response = ConversationResponse(answer=answer, status=status, verified=status == "verified", retry_count=0, claims=checks, evidence=evidence, saved_memory_id=saved_id)
    _save_trace(request, query=query, evidence=evidence, checks=checks, retry_count=0, status=status, retrieval_ms=retrieval_ms, inference_ms=inference_ms, total_ms=(time.perf_counter() - start) * 1_000, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response
