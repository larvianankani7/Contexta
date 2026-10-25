import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from backend.app import cli
from backend.app.config import Settings
from backend.app.main import create_app


def test_help_lists_supported_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for command in ("run", "ingest", "memory", "retrieve", "trace"):
        assert command in output
    assert "benchmark" not in output


def test_cli_ingests_utf8_file_with_memory_and_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context_file = tmp_path / "notes.txt"
    context_file.write_text("Project Alpha uses PostgreSQL.", encoding="utf-8")
    settings = Settings(
        database_path=tmp_path / "context.db",
        artifact_dir=tmp_path / "artifacts",
    )

    class Extractor:
        async def chat(self, *, messages, **kwargs):
            source = json.loads(messages[-1]["content"].split("\n", 1)[1])
            return SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {"facts": [{"text": source, "category": "decision"}]}
                    )
                )
            )

    app = create_app(settings)
    app.state.database.initialize()
    app.state.processor.client = Extractor()

    async def exercise() -> int:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://contexta.local",
            ) as client:
                return await cli.ingest_file(client, context_file)

    assert asyncio.run(exercise()) == 0
    assert "Stored memory:" in capsys.readouterr().out
    assert list((tmp_path / "artifacts").glob("*.png"))
    assert list((tmp_path / "artifacts").glob("*.json"))


def test_chat_exits_cleanly_on_exit_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        database_path=tmp_path / "context.db",
        artifact_dir=tmp_path / "artifacts",
    )

    class HealthResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"models": [{"name": settings.ollama_model}]}

    class OllamaProbe:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> HealthResponse:
            return HealthResponse()

    import backend.app.api.routes as routes

    monkeypatch.setattr(
        routes,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **kwargs: OllamaProbe()),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")

    app = create_app(settings)
    app.state.database.initialize()

    async def exercise() -> int:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://contexta.local",
            ) as client:
                return await cli.run_chat(client, settings, trace=False, reset_session=True)

    assert asyncio.run(exercise()) == 0
    output = capsys.readouterr().out
    assert "fresh chat session" in output
    assert "Goodbye." in output


def test_invalid_command_returns_argparse_error() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["unknown"])
    assert error.value.code == 2
