
import argparse
import asyncio
import json
import sys
import time

import httpx


async def chat(url: str, trace: bool) -> None:
    history = []
    print("CONTEXTA — Streaming Conversation")
    print(f"Server: {url}")
    print("Type exit to quit.")

    async with httpx.AsyncClient(timeout=None) as client:
        while True:
            try:
                query = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return

            if query.casefold() in {"exit", "quit"}:
                print("Goodbye.")
                return

            if not query:
                continue

            started = time.perf_counter()
            first_token = None
            result = None
            printed = False

            try:
                async with client.stream(
                    "POST",
                    f"{url.rstrip('/')}/conversation/stream",
                    json={
                        "query": query,
                        "history": history[-20:],
                    },
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        event = json.loads(line)
                        kind = event.get("type")

                        if kind == "token":
                            if first_token is None:
                                first_token = time.perf_counter()
                            if not printed:
                                print("\nContexta > ", end="", flush=True)
                                printed = True
                            print(
                                event["text"],
                                end="",
                                flush=True,
                            )

                        elif kind == "done":
                            result = event
                            break

                        elif kind == "error":
                            print(
                                "\nError: " + event["message"],
                                file=sys.stderr,
                            )
                            break

            except (httpx.HTTPError, ValueError) as error:
                print(
                    f"\nConnection error: {error}",
                    file=sys.stderr,
                )

            if result is None:
                continue

            finished = time.perf_counter()
            data = result["response"]

            if printed:
                print()
            else:
                print(f"\nContexta > {data['answer']}")

            history.extend([
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": data["answer"][:5000],
                },
            ])

            verification = {
                "verified": "SOURCE-BACKED",
                "unverified": "NOT MEMORY-VERIFIED",
                "insufficient_evidence": "INSUFFICIENT EVIDENCE",
            }
            status = verification.get(data["status"]) or data["status"]
            print(f"Verification: {status}")

            if result.get("cache_hit"):
                print("Response: CACHED")

            if data["evidence"]:
                print(
                    "Evidence: "
                    + ", ".join(
                        item["memory_id"]
                        for item in data["evidence"]
                    )
                )

            saved_id = data.get("saved_memory_id")
            if saved_id:
                print(f"Saved memory: {saved_id}")

            if not trace:
                continue

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    asyncio.run(chat(args.url, args.trace))


if __name__ == "__main__":
    main()
