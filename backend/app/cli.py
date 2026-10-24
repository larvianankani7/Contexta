import argparse
import asyncio
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import httpx
from ollama import AsyncClient, ResponseError

from backend.app.config import Settings, get_settings
from backend.app.main import create_app


@asynccontextmanager
async def api_client(settings: Settings) -> AsyncGenerator[httpx.AsyncClient, None]:
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://contexta.local") as client:
            yield client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contexta", description="Grounded memory for AI agents")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Start an interactive, evidence-verified chat")
    run.add_argument("model", nargs="?", help="Ollama model (defaults to OLLAMA_MODEL)")
    run.add_argument("--trace", action="store_true", help="Show retrieval and verification trace")
    run.add_argument("--reset-session", action="store_true", help="Start without local chat history")

    ingest = commands.add_parser("ingest", help="Store a UTF-8 text file as context")
    ingest.add_argument("file", type=Path)

    memory = commands.add_parser("memory", help="List stored memories")
    memory.add_argument("--limit", type=int, default=20)
    memory.add_argument("--offset", type=int, default=0)

    retrieve = commands.add_parser("retrieve", help="Search stored evidence")
    retrieve.add_argument("query")
    retrieve.add_argument("--limit", type=int, default=5)

    commands.add_parser("trace", help="Display the most recent execution trace")
    commands.add_parser("benchmark", help="Run the local Ollama memory benchmark")
    return parser


async def dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "run" and args.model:
        settings.ollama_model = args.model
    try:
        async with api_client(settings) as client:
            if args.command == "run":
                return await run_chat(
                    client,
                    settings,
                    trace=args.trace,
                    reset_session=args.reset_session,
                )
            if args.command == "ingest":
                return await ingest_file(client, args.file)
            if args.command == "memory":
                if not 1 <= args.limit <= 100 or args.offset < 0:
                    print("Limit must be 1-100 and offset must not be negative.", file=sys.stderr)
                    return 2
                return await list_memory(client, args.limit, args.offset)
            if args.command == "retrieve":
                if not 1 <= args.limit <= 20:
                    print("Limit must be between 1 and 20.", file=sys.stderr)
                    return 2
                return await retrieve(client, args.query, args.limit)
            if args.command == "trace":
                return await show_trace(client)
            if args.command == "benchmark":
                return await benchmark(client, settings)
    except (httpx.HTTPError, OSError) as error:
        print(f"Contexta could not reach a required local service: {error}", file=sys.stderr)
        return 1
    return 2


async def ingest_file(client: httpx.AsyncClient, path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        print(f"Cannot read UTF-8 context file {path}: {error}", file=sys.stderr)
        return 1
    if not text.strip():
        print("Context file is empty.", file=sys.stderr)
        return 2
    response = await client.post("/context", json={"text": text, "source": path.name})
    if response.is_error:
        print(f"Ingestion failed ({response.status_code}): {response.text}", file=sys.stderr)
        return 1
    record = response.json()
    print(f"Stored memory: {record['id']}")
    print(f"Artifact: {record['artifact_url']}")
    print(f"Metadata: {record['metadata_url']}")
    return 0


async def list_memory(client: httpx.AsyncClient, limit: int, offset: int) -> int:
    response = await client.get("/memory", params={"limit": limit, "offset": offset})
    if response.is_error:
        print(f"Memory listing failed ({response.status_code}): {response.text}", file=sys.stderr)
        return 1
    data = response.json()
    for item in data["items"]:
        summary = "; ".join(fact["text"] for fact in item["facts"])
        print(f"{item['id']}  {item['created_at']}  source={item['source'] or '-'}")
        print(f"  {summary[:240]}")
    print(f"{len(data['items'])} of {data['total']} memories")
    return 0


async def retrieve(client: httpx.AsyncClient, query: str, limit: int) -> int:
    response = await client.post("/retrieve", json={"query": query, "limit": limit})
    if response.is_error:
        print(f"Retrieval failed ({response.status_code}): {response.text}", file=sys.stderr)
        return 1
    evidence = response.json()["evidence"]
    for item in evidence:
        print(f"{item['memory_id']}  relevance={item['relevance']:.3f}  source={item['source'] or '-'}")
        print(f"  {item['text']}")
    if not evidence:
        print("No relevant evidence found.")
    return 0


async def show_trace(client: httpx.AsyncClient) -> int:
    response = await client.get("/trace")
    if response.status_code == 404:
        print("No execution trace is available yet.")
        return 0
    if response.is_error:
        print(f"Trace lookup failed ({response.status_code}): {response.text}", file=sys.stderr)
        return 1
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return 0


async def run_chat(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    trace: bool,
    reset_session: bool = False,
) -> int:
    print("CONTEXTA — Grounded Memory for AI Agents")
    if reset_session:
        print("Starting a fresh chat session; persistent memory is unchanged.")
    print(f"Model: {settings.ollama_model} | Verification: enabled")
    try:
        health = (await client.get("/health")).json()
    except (httpx.HTTPError, ValueError) as error:
        print(f"Could not check Contexta health: {error}", file=sys.stderr)
        return 1
    print(f"Memory: {'Connected' if health['database'] else 'Unavailable'}")
    if not health["ollama"]:
        print(f"Ollama model {settings.ollama_model!r} is unavailable.", file=sys.stderr)
        return 1
    while True:
        try:
            query = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if query.casefold() in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if not query:
            continue
        try:
            response = await client.post("/chat", json={"query": query})
        except httpx.HTTPError as error:
            print(f"Chat request failed: {error}", file=sys.stderr)
            continue
        if response.is_error:
            print(f"Chat unavailable ({response.status_code}): {response.text}", file=sys.stderr)
            continue
        result = response.json()
        print(f"\nContexta > {result['answer']}")
        verification = {
            "verified": "PASS",
            "unverified": "NOT PERFORMED (no relevant memory)",
            "insufficient_evidence": "INSUFFICIENT EVIDENCE",
        }[result["status"]]
        print(f"Verification: {verification}")
        if result["evidence"]:
            print("Evidence: " + ", ".join(item["memory_id"] for item in result["evidence"]))
        print(f"Recovery attempts: {result['retry_count']}")
        if trace:
            await show_trace(client)


async def benchmark(client: httpx.AsyncClient, settings: Settings) -> int:
    run_id = uuid.uuid4().hex[:8]
    dataset = [
        {
            "fact": f"For Contexta run {run_id}, Project Alpha selected PostgreSQL as its database.",
            "question": f"For Contexta run {run_id}, which database did Project Alpha select?",
            "expected": "PostgreSQL",
        },
        {
            "fact": f"For Contexta run {run_id}, Project Beta uses SQLite for local prototyping.",
            "question": f"For Contexta run {run_id}, which database does Project Beta use?",
            "expected": "SQLite",
        },
        {
            "fact": f"For Contexta run {run_id}, the alpha service listens on port 8123.",
            "question": f"For Contexta run {run_id}, which port does the alpha service listen on?",
            "expected": "8123",
        },
        {
            "fact": f"For Contexta run {run_id}, Project Delta selected PostgreSQL as its database.",
            "conflicting_fact": f"For Contexta run {run_id}, Project Delta selected MongoDB as its database.",
            "question": f"For Contexta run {run_id}, which database did Project Delta select?",
            "expected": "unknown",
        },
        {
            "fact": None,
            "question": f"For Contexta run {run_id}, which database engine does Project Gamma use?",
            "expected": "unknown",
        },
    ]
    model = AsyncClient(host=settings.ollama_base_url, timeout=settings.ollama_timeout)
    cases: list[dict[str, object]] = []
    test_b_cases: list[dict[str, object]] = []
    test_c_cases: list[dict[str, object]] = []
    for index, item in enumerate(dataset, start=1):
        memory_id: str | None = None
        memory_ids: list[str] = []
        ingestion_ms = 0.0
        if item["fact"]:
            ingest_started = time.perf_counter()
            created = await client.post(
                "/context",
                json={"text": item["fact"], "source": f"benchmark:{index}"},
            )
            if created.is_error:
                print(f"Benchmark ingestion failed ({created.status_code}).", file=sys.stderr)
                return 1
            created_id = created.json().get("id")
            if not isinstance(created_id, str) or not created_id:
                print("Benchmark ingestion response did not include a memory ID.", file=sys.stderr)
                return 1
            memory_id = created_id
            memory_ids.append(memory_id)
            conflict = item.get("conflicting_fact")
            if conflict:
                second = await client.post(
                    "/context",
                    json={"text": conflict, "source": f"benchmark:{index}:conflict"},
                )
                if second.is_error:
                    print(f"Conflicting benchmark ingestion failed ({second.status_code}).", file=sys.stderr)
                    return 1
                second_id = second.json().get("id")
                if not isinstance(second_id, str) or not second_id:
                    print("Conflicting benchmark response did not include a memory ID.", file=sys.stderr)
                    return 1
                memory_ids.append(second_id)
            ingestion_ms = (time.perf_counter() - ingest_started) * 1_000
        baseline_started = time.perf_counter()
        try:
            baseline_response = await model.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": "Answer briefly and truthfully. Say unknown if not known."},
                    {"role": "user", "content": item["question"]},
                ],
                options={
                    "temperature": 0,
                    "num_ctx": settings.ollama_num_ctx,
                    "num_predict": settings.ollama_num_predict,
                    "seed": settings.ollama_seed,
                },
                think=False,
            )
        except (ResponseError, httpx.HTTPError, OSError, TimeoutError) as error:
            print(f"Baseline Ollama request failed: {type(error).__name__}", file=sys.stderr)
            return 1
        baseline_ms = (time.perf_counter() - baseline_started) * 1_000
        baseline_answer = baseline_response.message.content
        if baseline_answer is None or not baseline_answer.strip():
            print("Baseline Ollama request returned no answer text.", file=sys.stderr)
            return 1
        controlled_started = time.perf_counter()
        try:
            controlled = await model.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": "Answer briefly and truthfully using only the supplied source."},
                    {
                        "role": "user",
                        "content": (
                            "Source: "
                            + "\n".join(
                                part
                                for part in (item["fact"], item.get("conflicting_fact"))
                                if part
                            )
                            + f"\nQuestion: {item['question']}"
                            if item["fact"]
                            else f"No source evidence is available.\nQuestion: {item['question']}"
                        ),
                    },
                ],
                options={
                    "temperature": 0,
                    "num_ctx": settings.ollama_num_ctx,
                    "num_predict": settings.ollama_num_predict,
                    "seed": settings.ollama_seed,
                },
                think=False,
            )
        except (ResponseError, httpx.HTTPError, OSError, TimeoutError) as error:
            print(f"Controlled Ollama request failed: {type(error).__name__}", file=sys.stderr)
            return 1
        controlled_ms = (time.perf_counter() - controlled_started) * 1_000
        controlled_answer = controlled.message.content
        if controlled_answer is None or not controlled_answer.strip():
            print("Controlled Ollama request returned no answer text.", file=sys.stderr)
            return 1
        enhanced_response = await client.post("/chat", json={"query": item["question"]})
        if enhanced_response.is_error:
            print(f"Contexta benchmark request failed ({enhanced_response.status_code}).", file=sys.stderr)
            return 1
        enhanced = enhanced_response.json()
        trace_response = await client.get("/trace")
        trace = trace_response.json() if not trace_response.is_error else {}
        cases.append(
            {
                "question": item["question"],
                "expected": item["expected"],
                "baseline_access": "question only; prior facts unavailable",
                "baseline_answer": baseline_answer,
                "baseline_correct": _answer_correct(baseline_answer, item["expected"]),
                "baseline_latency_ms": round(baseline_ms, 3),
                "baseline_prompt_tokens": getattr(baseline_response, "prompt_eval_count", None),
                "baseline_completion_tokens": getattr(baseline_response, "eval_count", None),
                "contexta_answer": enhanced["answer"],
                "contexta_correct": _answer_correct(
                    enhanced["answer"], item["expected"], insufficient=not enhanced["verified"]
                ),
                "contexta_verified": enhanced["verified"],
                "contexta_retry_count": enhanced["retry_count"],
                "memory_id": memory_id,
                "memory_ids": memory_ids,
                "ingestion_ms": round(ingestion_ms, 3),
                "contexta_retrieval_ms": trace.get("retrieval_ms"),
                "contexta_inference_ms": trace.get("inference_ms"),
                "contexta_total_ms": trace.get("total_ms"),
                "contexta_prompt_tokens": trace.get("prompt_tokens"),
                "contexta_completion_tokens": trace.get("completion_tokens"),
            }
        )
        test_b_cases.append(
            {
                "question": item["question"],
                "expected": item["expected"],
                "baseline_source": "\n".join(
                    part for part in (item["fact"], item.get("conflicting_fact")) if part
                )
                or "no source evidence",
                "baseline_answer": controlled_answer,
                "baseline_correct": _answer_correct(controlled_answer, item["expected"]),
                "baseline_latency_ms": round(controlled_ms, 3),
                "baseline_prompt_tokens": getattr(controlled, "prompt_eval_count", None),
                "baseline_completion_tokens": getattr(controlled, "eval_count", None),
                "contexta_answer": enhanced["answer"],
                "contexta_correct": _answer_correct(
                    enhanced["answer"], item["expected"], insufficient=not enhanced["verified"]
                ),
                "contexta_verified": enhanced["verified"],
                "contexta_retry_count": enhanced["retry_count"],
                "contexta_trace": trace,
            }
        )
        for distractor_count in (50, 250, 1_000):
            history = "\n".join(
                f"Unrelated note {offset}: this is a non-relevant project preference about color {offset}."
                for offset in range(distractor_count)
            )
            source_text = "\n".join(
                part for part in (item["fact"], item.get("conflicting_fact")) if part
            )
            source = f"\nRelevant source: {source_text}" if source_text else "\nNo source evidence is available."
            stress_prompt = f"{history}{source}\nQuestion: {item['question']}"
            stress_started = time.perf_counter()
            try:
                stressed = await model.chat(
                    model=settings.ollama_model,
                    messages=[
                        {"role": "system", "content": "Answer briefly using only the relevant source. Do not invent."},
                        {"role": "user", "content": stress_prompt},
                    ],
                    options={
                        "temperature": 0,
                        "num_ctx": settings.ollama_num_ctx,
                        "num_predict": settings.ollama_num_predict,
                        "seed": settings.ollama_seed,
                    },
                    think=False,
                )
            except (ResponseError, httpx.HTTPError, OSError, TimeoutError) as error:
                print(f"Stress-test Ollama request failed: {type(error).__name__}", file=sys.stderr)
                return 1
            stressed_answer = stressed.message.content
            if stressed_answer is None or not stressed_answer.strip():
                print("Stress-test Ollama request returned no answer text.", file=sys.stderr)
                return 1
            test_c_cases.append(
                {
                    "question": item["question"],
                    "expected": item["expected"],
                    "distractor_count": distractor_count,
                    "history_characters": len(stress_prompt),
                    "configured_context_tokens": settings.ollama_num_ctx,
                    "documented_truncation": "Ollama prompt_eval_count is recorded; no application truncation is applied.",
                    "baseline_answer": stressed_answer,
                    "baseline_correct": _answer_correct(stressed_answer, item["expected"]),
                    "baseline_latency_ms": round((time.perf_counter() - stress_started) * 1_000, 3),
                    "baseline_prompt_tokens": getattr(stressed, "prompt_eval_count", None),
                    "baseline_completion_tokens": getattr(stressed, "eval_count", None),
                    "contexta_answer": enhanced["answer"],
                    "contexta_correct": _answer_correct(
                        enhanced["answer"], item["expected"], insufficient=not enhanced["verified"]
                    ),
                    "contexta_verified": enhanced["verified"],
                    "contexta_retrieval_count": len(enhanced["evidence"]),
                    "contexta_retrieval_ms": trace.get("retrieval_ms"),
                    "contexta_inference_ms": trace.get("inference_ms"),
                    "contexta_prompt_tokens": trace.get("prompt_tokens"),
                    "contexta_completion_tokens": trace.get("completion_tokens"),
                }
            )
        print(f"{index}/{len(dataset)} {item['question']}")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": settings.ollama_model,
        "temperature": 0,
        "seed": settings.ollama_seed,
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": settings.ollama_num_predict,
        "test_a": {
            "label": "Raw baseline versus persistent Contexta memory; information access differs",
            "cases": cases,
        },
        "test_b": {
            "label": "Controlled comparison; both approaches receive the same source evidence",
            "cases": test_b_cases,
        },
        "test_c": {
            "label": "Long-context stress; baseline sees 50, 250, and 1,000 distractor notes",
            "cases": test_c_cases,
        },
        "summary": {
            "test_a_baseline_accuracy": _accuracy(cases, "baseline_correct"),
            "test_a_contexta_accuracy": _accuracy(cases, "contexta_correct"),
            "test_a_baseline_incorrect_rate": _rate(cases, "baseline_correct", invert=True),
            "test_a_contexta_unverified_response_rate": _rate(cases, "contexta_verified", invert=True),
            "test_b_baseline_accuracy": _accuracy(test_b_cases, "baseline_correct"),
            "test_b_contexta_accuracy": _accuracy(test_b_cases, "contexta_correct"),
            "test_b_contexta_unverified_response_rate": _rate(test_b_cases, "contexta_verified", invert=True),
            "test_c_baseline_accuracy": _accuracy(test_c_cases, "baseline_correct"),
            "test_c_contexta_accuracy": _accuracy(test_c_cases, "contexta_correct"),
            "test_c_contexta_unverified_response_rate": _rate(test_c_cases, "contexta_verified", invert=True),
            "test_a_contexta_verified_count": sum(bool(item["contexta_verified"]) for item in cases),
            "test_a_recovery_success_count": sum(
                bool(item["contexta_verified"])
                and isinstance(item["contexta_retry_count"], int)
                and item["contexta_retry_count"] > 0
                for item in cases
            ),
            "test_a_case_count": len(cases),
        },
        "limitations": [
            "Test A gives the baseline only the question; Contexta can retrieve stored facts.",
            "Test B gives both approaches the same source evidence but compares ordinary generation with retrieval plus verification.",
            "Test C reports source-history size and Ollama token counts at three distractor levels; truncation behavior depends on the selected Ollama model.",
            "Prompt-token and completion-token counts are recorded only when Ollama exposes them.",
        ],
    }
    output_dir = settings.project_path(Path("backend/data/benchmarks"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{run_id}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved actual results to {output_path}")
    print(f"Contexta verified: {sum(bool(item['contexta_verified']) for item in cases)}/{len(cases)}")
    return 0


def _accuracy(cases: list[dict[str, object]], key: str) -> float | None:
    return (
        round(sum(bool(case[key]) for case in cases) / len(cases), 4)
        if cases
        else None
    )


def _rate(
    cases: list[dict[str, object]],
    key: str,
    *,
    invert: bool = False,
) -> float | None:
    if not cases:
        return None
    matches = sum(bool(case[key]) != invert for case in cases)
    return round(matches / len(cases), 4)


def _answer_correct(answer: str, expected: str, *, insufficient: bool = False) -> bool:
    if expected == "unknown":
        lowered = answer.casefold()
        return insufficient or any(
            phrase in lowered
            for phrase in ("unknown", "don't know", "do not know", "not enough evidence")
        )
    return expected.casefold() in answer.casefold() and not insufficient


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        return asyncio.run(dispatch(args, settings))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
