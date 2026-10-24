import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Sequence
import httpx
from backend.app.config import Settings, get_settings
from backend.app.main import create_app

@asynccontextmanager
async def api_client(settings: Settings) -> AsyncGenerator[httpx.AsyncClient, None]:
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://contexta.local",
        ) as client:
            yield client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contexta",
        description="Grounded memory for AI agents",
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run = commands.add_parser(
        "run",
        help="Start an interactive Contexta conversation",
    )
    run.add_argument(
        "model",
        nargs="?",
        help="Ollama model",
    )
    run.add_argument(
        "--trace",
        action="store_true",
    )
    run.add_argument(
        "--reset-session",
        action="store_true",
    )

    ingest = commands.add_parser(
        "ingest",
        help="Store a UTF-8 text file",
    )
    ingest.add_argument("file", type=Path)

    memory = commands.add_parser(
        "memory",
        help="List stored memories",
    )
    memory.add_argument("--limit", type=int, default=20)
    memory.add_argument("--offset", type=int, default=0)

    retrieve = commands.add_parser(
        "retrieve",
        help="Retrieve stored evidence",
    )
    retrieve.add_argument("query")
    retrieve.add_argument("--limit", type=int, default=5)

    commands.add_parser(
        "trace",
        help="Display the latest execution trace",
    )

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
                    print("Invalid memory range.", file=sys.stderr)
                    return 2
                return await list_memory(
                    client,
                    args.limit,
                    args.offset,
                )
            if args.command == "retrieve":
                if not 1 <= args.limit <= 20:
                    print("Limit must be 1-20.", file=sys.stderr)
                    return 2
                return await retrieve(
                    client,
                    args.query,
                    args.limit,
                )
            if args.command == "trace":
                return await show_trace(client)
    except (httpx.HTTPError, OSError) as error:
        print(f"Contexta service error: {error}", file=sys.stderr)
        return 1

    return 2


async def ingest_file(client: httpx.AsyncClient, path: Path) -> int:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        print(f"Cannot read file: {error}", file=sys.stderr)
        return 1

    if not content.strip():
        print("Context file is empty.", file=sys.stderr)
        return 2

    response = await client.post(
        "/context",
        json={"text": content, "source": path.name},
    )
    if response.is_error:
        print(f"Ingestion failed: {response.text}", file=sys.stderr)
        return 1

    record = response.json()
    print(f"Stored memory: {record['id']}")
    print(f"Artifact: {record['artifact_url']}")
    print(f"Metadata: {record['metadata_url']}")
    return 0


async def list_memory(
    client: httpx.AsyncClient,
    limit: int,
    offset: int,
) -> int:
    response = await client.get(
        "/memory",
        params={"limit": limit, "offset": offset},
    )
    if response.is_error:
        print(f"Memory lookup failed: {response.text}", file=sys.stderr)
        return 1

    data = response.json()
    for item in data["items"]:
        summary = "; ".join(
            fact["text"] for fact in item["facts"]
        )
        print(
            f"{item['id']}  {item['created_at']}  "
            f"source={item['source'] or '-'}"
        )
        print(f"  {summary[:240]}")

    print(f"{len(data['items'])} of {data['total']} memories")
    return 0


async def retrieve(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
) -> int:
    response = await client.post(
        "/retrieve",
        json={"query": query, "limit": limit},
    )
    if response.is_error:
        print(f"Retrieval failed: {response.text}", file=sys.stderr)
        return 1

    evidence = response.json()["evidence"]
    for item in evidence:
        print(
            f"{item['memory_id']}  "
            f"relevance={item['relevance']:.3f}  "
            f"source={item['source'] or '-'}"
        )
        print(f"  {item['text']}")

    if not evidence:
        print("No relevant evidence found.")

    return 0


async def show_trace(client: httpx.AsyncClient) -> int:
    response = await client.get("/trace")
    if response.status_code == 404:
        print("No execution trace available.")
        return 0
    if response.is_error:
        print(f"Trace lookup failed: {response.text}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            response.json(),
            indent=2,
            ensure_ascii=False,
        )
    )
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
        print(f"Health check failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Memory: {'Connected' if health['database'] else 'Unavailable'}"
    )
    if not health["ollama"]:
        print(
            f"Ollama model {settings.ollama_model!r} is unavailable.",
            file=sys.stderr,
        )
        return 1

    history: list[dict[str, str]] = []

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
            response = await client.post(
                "/conversation",
                json={
                    "query": query,
                    "history": history[-20:],
                },
            )
        except httpx.HTTPError as error:
            print(f"Chat request failed: {error}", file=sys.stderr)
            continue

        if response.is_error:
            print(
                f"Chat unavailable ({response.status_code}): "
                f"{response.text}",
                file=sys.stderr,
            )
            continue

        result = response.json()
        history.extend([
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": result["answer"][:5000],
            },
        ])

        print(f"\nContexta > {result['answer']}")

        verification = {
            "verified": "SOURCE-BACKED (exact saved text)",
            "unverified": "NOT MEMORY-VERIFIED",
            "insufficient_evidence": "INSUFFICIENT EVIDENCE",
        }
        status = verification.get(result["status"]) or result["status"]
        print(f"Verification: {status}")

        if result["evidence"]:
            print(
                "Evidence: "
                + ", ".join(
                    item["memory_id"]
                    for item in result["evidence"]
                )
            )

        if result.get("saved_memory_id"):
            print(f"Saved memory: {result['saved_memory_id']}")

        if trace:
            await show_trace(client)


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
