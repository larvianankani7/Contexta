import logging
import hashlib
import re
import sqlite3
import time
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from backend.app.reasoning.client import ReasoningUnavailable
from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    ClaimVerification,
    ContextCreated,
    ContextInput,
    Evidence,
    MemoryDetail,
    MemoryList,
    MemorySummary,
    RetrieveRequest,
    RetrieveResponse,
    VerifyRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()
MEMORY_CUE_RE = re.compile(
    r"\b(?:my|our|we|previous(?:ly)?|earlier|last time|remember|recall|stored|saved|"
    r"decided|decision|in this project|for project)\b",
    re.IGNORECASE,
)
PROJECT_NAME_RE = re.compile(r"\bproject\s+[A-Z][A-Za-z0-9_-]*\b")
IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+\b")


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    database = request.app.state.database
    try:
        with database.connection() as connection:
            connection.execute("SELECT 1").fetchone()
        database_available = True
    except sqlite3.Error as error:
        logger.error("Database health check failed (%s)", type(error).__name__)
        database_available = False
    settings = request.app.state.settings
    ollama_service = False
    ollama_available = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
        ollama_service = True
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        if isinstance(models, list):
            ollama_available = any(
                isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and (
                    item["name"] == settings.ollama_model
                    or item["name"].startswith(f"{settings.ollama_model}:")
                )
                for item in models
            )
    except (httpx.HTTPError, ValueError):
        logger.warning("Ollama health check failed")
    return {
        "status": "ok" if database_available and ollama_available else "degraded",
        "database": database_available,
        "ollama_service": ollama_service,
        "ollama": ollama_available,
        "model": settings.ollama_model,
    }


@router.post("/context", response_model=ContextCreated, status_code=201)
async def create_context(payload: ContextInput, request: Request) -> dict[str, object]:
    try:
        record = await request.app.state.processor.ingest(
            text=payload.text,
            source=payload.source,
            metadata=payload.metadata,
        )
    except (OSError, sqlite3.Error) as error:
        logger.exception("Context persistence failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Context could not be persisted") from error
    return {
        **_summary(record),
        "artifact_url": f"/artifacts/{record['id']}",
        "metadata_url": f"/artifacts/{record['id']}/metadata",
        "extraction_method": record["extraction_method"],
    }


@router.get("/memory", response_model=MemoryList)
def list_memories(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> MemoryList:
    try:
        records, total = request.app.state.database.list(limit=limit, offset=offset)
    except sqlite3.Error as error:
        logger.exception("Memory listing failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Memories could not be listed") from error
    return MemoryList(
        items=[MemorySummary.model_validate(_summary(record)) for record in records],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.get("/memory/{memory_id}", response_model=MemoryDetail)
def get_memory(memory_id: str, request: Request) -> dict[str, object]:
    try:
        record = request.app.state.database.get(memory_id)
    except sqlite3.Error as error:
        logger.exception("Memory lookup failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Memory could not be retrieved") from error
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        **_summary(record),
        "metadata": record["metadata"],
        "artifact_url": f"/artifacts/{memory_id}",
        "metadata_url": f"/artifacts/{memory_id}/metadata",
    }


@router.get("/artifacts/{memory_id}")
def get_artifact(memory_id: str, request: Request) -> FileResponse:
    try:
        record = request.app.state.database.get(memory_id)
    except sqlite3.Error as error:
        logger.exception("Artifact lookup failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Artifact could not be retrieved") from error
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return FileResponse(str(record["artifact_path"]), media_type="image/png")


@router.get("/artifacts/{memory_id}/metadata")
def get_artifact_metadata(memory_id: str, request: Request) -> FileResponse:
    try:
        record = request.app.state.database.get(memory_id)
    except sqlite3.Error as error:
        logger.exception("Metadata lookup failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Metadata could not be retrieved") from error
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return FileResponse(str(record["metadata_path"]), media_type="application/json")


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    try:
        evidence = request.app.state.retriever.retrieve(payload.query, limit=payload.limit)
    except sqlite3.Error as error:
        logger.exception("Evidence retrieval failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Evidence could not be retrieved") from error
    return RetrieveResponse(query=payload.query, evidence=evidence)


@router.post("/verify")
def verify(payload: VerifyRequest, request: Request) -> dict[str, object]:
    evidence: list[Evidence] = []
    database = request.app.state.database
    for memory_id in dict.fromkeys(payload.evidence_ids):
        try:
            record = database.get(memory_id)
        except sqlite3.Error as error:
            logger.exception("Evidence lookup failed (%s)", type(error).__name__)
            raise HTTPException(status_code=500, detail="Evidence could not be loaded") from error
        if record is None:
            continue
        evidence.append(
            Evidence(
                memory_id=memory_id,
                text=" ".join(str(fact["text"]) for fact in record["facts"]),
                source=record["source"],
                relevance=1.0,
            )
        )
    if not evidence:
        return {
            "status": "insufficient_evidence",
            "supported": False,
            "contradictory": False,
            "verification": None,
        }
    from backend.app.reasoning.client import ReasoningClaim, ReasoningResult

    result = ReasoningResult(
        answer=payload.claim,
        claims=[ReasoningClaim(text=payload.claim, evidence_ids=payload.evidence_ids)],
    )
    try:
        check = request.app.state.verifier.verify(result, evidence)[0]
    except sqlite3.Error as error:
        logger.exception("Evidence verification failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Evidence could not be verified") from error
    return {
        "status": "supported" if check.supported else "contradictory" if check.contradictory else "unsupported",
        "supported": check.supported,
        "contradictory": check.contradictory,
        "verification": check.model_dump(),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    started = time.perf_counter()
    retrieval_started = started
    retriever = request.app.state.retriever
    try:
        evidence_by_id = {
            item.memory_id: item for item in retriever.retrieve(payload.query, limit=5)
        }
    except sqlite3.Error as error:
        logger.exception("Chat context retrieval failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Context could not be retrieved") from error
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1_000
    reasoning = request.app.state.reasoning
    settings = request.app.state.settings
    if not evidence_by_id:
        if _requires_memory(payload.query):
            response = _insufficient(retry_count=0, evidence=[])
            _save_trace(
                request,
                query=payload.query,
                evidence=[],
                checks=[],
                retry_count=0,
                status=response.status,
                retrieval_ms=retrieval_ms,
                inference_ms=0,
                total_ms=(time.perf_counter() - started) * 1_000,
                prompt_tokens=None,
                completion_tokens=None,
            )
            return response
        inference_started = time.perf_counter()
        try:
            result = await reasoning.answer_general(payload.query)
        except ReasoningUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail="The configured Ollama model is unavailable or returned an invalid response",
            ) from error
        inference_ms = (time.perf_counter() - inference_started) * 1_000
        if not result.answer.strip():
            raise HTTPException(status_code=503, detail="Ollama returned an empty response")
        response = ChatResponse(
            answer=result.answer,
            status="unverified",
            verified=False,
            retry_count=0,
            claims=[],
            evidence=[],
        )
        _save_trace(
            request,
            query=payload.query,
            evidence=[],
            checks=[],
            retry_count=0,
            status=response.status,
            retrieval_ms=retrieval_ms,
            inference_ms=inference_ms,
            total_ms=(time.perf_counter() - started) * 1_000,
            prompt_tokens=result._prompt_tokens,
            completion_tokens=result._completion_tokens,
        )
        return response
    verifier = request.app.state.verifier
    last_answer = "I don't have enough verified context to answer that."
    last_checks: list[ClaimVerification] = []
    inference_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    has_usage = False

    retry_count = 0
    for retry_count in range(settings.max_recovery_attempts + 1):
        evidence = list(evidence_by_id.values())
        try:
            inference_started = time.perf_counter()
            result = await reasoning.answer(payload.query, evidence)
            inference_ms += (time.perf_counter() - inference_started) * 1_000
            if result._prompt_tokens is not None:
                prompt_tokens += result._prompt_tokens
                has_usage = True
            if result._completion_tokens is not None:
                completion_tokens += result._completion_tokens
        except ReasoningUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail="The configured Ollama model is unavailable or returned an invalid response",
            ) from error
        try:
            checks = verifier.verify(result, evidence)
        except sqlite3.Error as error:
            logger.exception("Claim verification failed (%s)", type(error).__name__)
            raise HTTPException(status_code=500, detail="Claims could not be verified") from error
        answer_supported = verifier.answer_supported(result.answer, checks)
        if (
            result.answer.strip()
            and result.claims
            and checks
            and all(check.supported for check in checks)
            and answer_supported
            and not result.insufficient_evidence
        ):
            response = ChatResponse(
                answer=result.answer,
                status="verified",
                verified=True,
                retry_count=retry_count,
                claims=checks,
                evidence=evidence,
            )
            _save_trace(
                request,
                query=payload.query,
                evidence=evidence,
                checks=checks,
                retry_count=retry_count,
                status=response.status,
                retrieval_ms=retrieval_ms,
                inference_ms=inference_ms,
                total_ms=(time.perf_counter() - started) * 1_000,
                prompt_tokens=prompt_tokens if has_usage else None,
                completion_tokens=completion_tokens if has_usage else None,
            )
            return response
        last_answer = "I don't have enough verified context to answer that."
        last_checks = checks
        if checks and all(check.supported for check in checks) and not answer_supported:
            last_checks = [
                *checks,
                ClaimVerification(
                    claim=result.answer,
                    supported=False,
                    evidence_ids=[],
                    reason="The answer includes terms not covered by its verified claims.",
                ),
            ]
        if retry_count >= settings.max_recovery_attempts:
            break
        recovery_query = " ".join(
            [payload.query, *(check.claim for check in last_checks if not check.supported)]
        )
        try:
            additional_evidence = retriever.retrieve(recovery_query, limit=10)
        except sqlite3.Error as error:
            logger.exception("Recovery evidence retrieval failed (%s)", type(error).__name__)
            raise HTTPException(status_code=500, detail="Recovery context could not be retrieved") from error
        new_evidence = [
            item for item in additional_evidence if item.memory_id not in evidence_by_id
        ]
        if not new_evidence:
            break
        for item in new_evidence:
            evidence_by_id[item.memory_id] = item
    response = ChatResponse(
        answer=last_answer,
        status="insufficient_evidence",
        verified=False,
        retry_count=min(settings.max_recovery_attempts, retry_count),
        claims=last_checks,
        evidence=list(evidence_by_id.values()),
    )
    _save_trace(
        request,
        query=payload.query,
        evidence=response.evidence,
        checks=response.claims,
        retry_count=response.retry_count,
        status=response.status,
        retrieval_ms=retrieval_ms,
        inference_ms=inference_ms,
        total_ms=(time.perf_counter() - started) * 1_000,
        prompt_tokens=prompt_tokens if has_usage else None,
        completion_tokens=completion_tokens if has_usage else None,
    )
    return response


@router.get("/trace")
def latest_trace(request: Request) -> dict[str, object]:
    try:
        trace = request.app.state.database.latest_trace()
    except sqlite3.Error as error:
        logger.exception("Execution trace lookup failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Execution trace could not be loaded") from error
    if trace is None:
        raise HTTPException(status_code=404, detail="No execution trace available")
    return trace


def _summary(record: dict[str, object]) -> dict[str, object]:
    return {
        "id": record["id"],
        "content": record["content"],
        "facts": record["facts"],
        "source": record["source"],
        "created_at": record["created_at"],
    }


def _insufficient(*, retry_count: int, evidence: list[Evidence]) -> ChatResponse:
    return ChatResponse(
        answer="I don't have enough verified context to answer that.",
        status="insufficient_evidence",
        verified=False,
        retry_count=retry_count,
        claims=[],
        evidence=evidence,
    )


def _requires_memory(query: str) -> bool:
    return bool(
        MEMORY_CUE_RE.search(query)
        or PROJECT_NAME_RE.search(query)
        or IDENTIFIER_RE.search(query)
    )


def _save_trace(
    request: Request,
    *,
    query: str,
    evidence: list[Evidence],
    checks: list[ClaimVerification],
    retry_count: int,
    status: str,
    retrieval_ms: float,
    inference_ms: float,
    total_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    try:
        request.app.state.database.save_trace(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "query": "[redacted]",
                "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "retrieval_count": len(evidence),
                "evidence_ids": [item.memory_id for item in evidence],
                "verification": [
                    {
                        "supported": check.supported,
                        "contradictory": check.contradictory,
                        "evidence_ids": check.evidence_ids,
                        "reason": check.reason,
                    }
                    for check in checks
                ],
                "retry_count": retry_count,
                "status": status,
                "retrieval_ms": round(retrieval_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "total_ms": round(total_ms, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
    except sqlite3.Error as error:
        logger.exception("Execution trace persistence failed (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Execution trace could not be saved") from error
