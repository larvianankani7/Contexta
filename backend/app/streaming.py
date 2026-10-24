
import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.app.api.routes import _save_trace
from backend.app.conversation import (
    ConversationRequest,
    _is_save,
    _needs_memory,
    converse,
)
from backend.app.reasoning.client import ReasoningUnavailable
from backend.app.schemas import ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _event(name: str, data: dict) -> str:
    return json.dumps(
        {"type": name, **data},
        ensure_ascii=False,
    ) + "\n"


@router.post("/conversation/stream")
async def stream_conversation(
    payload: ConversationRequest,
    request: Request,
) -> StreamingResponse:
    query = payload.query.strip()

    async def generate():
        started = time.perf_counter()

        if _is_save(query) or _needs_memory(query):
            try:
                response = await converse(payload, request)
                yield _event(
                    "done",
                    {
                        "response": response.model_dump(),
                        "cache_hit": False,
                        "ttft_ms": None,
                        "generation_ms": 0.0,
                        "inference_ms": 0.0,
                        "total_ms": round(
                            (time.perf_counter() - started) * 1000, 3
                        ),
                    },
                )
            except Exception:
                logger.exception("Memory operation failed")
                yield _event(
                    "error", {"message": "Memory operation failed"}
                )
            return

        inference_started = time.perf_counter()
        first_token_at = None
        completed = None

        try:
            async for event in request.app.state.reasoning.stream_general(
                query,
                history=[
                    turn.model_dump() for turn in payload.history
                ],
            ):
                if await request.is_disconnected():
                    return

                if event["type"] == "token":
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield _event("token", {"text": event["text"]})

                elif event["type"] == "completed":
                    completed = event

            if completed is None:
                raise ReasoningUnavailable(
                    "Stream ended without completion"
                )

            finished = time.perf_counter()
            cache_hit = completed["cache_hit"]

            inference_ms = (
                0.0 if cache_hit
                else (finished - inference_started) * 1000
            )

            ttft_ms = (
                (first_token_at - inference_started) * 1000
                if first_token_at is not None else None
            )

            generation_ms = (
                (finished - first_token_at) * 1000
                if first_token_at is not None else 0.0
            )

            response = ChatResponse(
                answer=completed["answer"],
                status="unverified",
                verified=False,
                retry_count=0,
                claims=[],
                evidence=[],
            )

            _save_trace(
                request,
                query=query,
                evidence=[],
                checks=[],
                retry_count=0,
                status="unverified",
                retrieval_ms=0.0,
                inference_ms=inference_ms,
                total_ms=(finished - started) * 1000,
                prompt_tokens=completed["prompt_tokens"],
                completion_tokens=completed["completion_tokens"],
            )

            yield _event(
                "done",
                {
                    "response": response.model_dump(),
                    "cache_hit": cache_hit,
                    "ttft_ms": (
                        round(ttft_ms, 3)
                        if ttft_ms is not None else None
                    ),
                    "generation_ms": round(generation_ms, 3),
                    "inference_ms": round(inference_ms, 3),
                    "total_ms": round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                    "prompt_tokens": completed["prompt_tokens"],
                    "completion_tokens": completed["completion_tokens"],
                    "ollama_total_ms": completed["ollama_total_ms"],
                    "prompt_eval_ms": completed["prompt_eval_ms"],
                    "eval_ms": completed["eval_ms"],
                },
            )

        except ReasoningUnavailable:
            logger.exception("Streaming inference unavailable")
            yield _event(
                "error",
                {"message": "Ollama streaming inference failed"},
            )
        except Exception:
            logger.exception("Streaming request failed")
            yield _event(
                "error",
                {"message": "Streaming request failed"},
            )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
