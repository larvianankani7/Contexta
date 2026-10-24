import json
from types import SimpleNamespace
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from fastapi import FastAPI
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


class ExtractionStub:
    async def chat(self, *, messages: list[dict[str, str]], **kwargs: object) -> SimpleNamespace:
        source = json.loads(
            messages[-1]["content"].split("Source text (JSON string):\n", 1)[1]
        )
        content = json.dumps({"facts": [{"text": source, "category": "decision"}]})
        return SimpleNamespace(message=SimpleNamespace(content=content))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        database_path=tmp_path / "context.db",
        artifact_dir=tmp_path / "artifacts",
        max_recovery_attempts=2,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        cast(FastAPI, test_client.app).state.processor.client = ExtractionStub()
        yield test_client
