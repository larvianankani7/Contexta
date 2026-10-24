import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from fastapi.testclient import TestClient


def test_health_reports_database_and_ollama(client: TestClient, monkeypatch) -> None:
    class OllamaResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"models": [{"name": "qwen3-vl:4b"}]}

    class OllamaProbe:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str):
            return OllamaResponse()

    monkeypatch.setattr("backend.app.api.routes.httpx.AsyncClient", lambda **kwargs: OllamaProbe())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"] is True
    assert response.json()["ollama"] is True


def test_context_ingestion_creates_persisted_png_and_json(client: TestClient) -> None:
    text = "For Project Alpha, we decided to use PostgreSQL as the database."
    response = client.post("/context", json={"text": text, "source": "planning"})
    assert response.status_code == 201
    created = response.json()
    assert created["content"] == text
    assert created["extraction_method"] == "ollama"

    image_response = client.get(created["artifact_url"])
    image = Image.open(BytesIO(image_response.content))
    assert image.format == "PNG"
    assert image.width > 0

    metadata_response = client.get(created["metadata_url"])
    artifact_metadata = metadata_response.json()
    assert artifact_metadata["id"] == created["id"]
    assert artifact_metadata["facts"][0]["text"] == text
    assert cast(FastAPI, client.app).state.database.get(created["id"]) is not None


def test_empty_context_is_rejected(client: TestClient) -> None:
    response = client.post("/context", json={"text": "   "})
    assert response.status_code == 422


def test_memory_detail_and_missing_memory(client: TestClient) -> None:
    created = client.post("/context", json={"text": "Project Alpha uses PostgreSQL."}).json()
    detail = client.get(f"/memory/{created['id']}")
    assert detail.status_code == 200
    assert detail.json()["metadata"] == {}
    assert client.get("/memory/not-a-memory").status_code == 404


def test_retrieval_ranks_relevant_memory(client: TestClient) -> None:
    client.post("/context", json={"text": "Project Alpha selected PostgreSQL for its database."})
    client.post("/context", json={"text": "Project Beta team prefers dark mode."})
    client.post("/context", json={"text": "Project Beta selected MySQL for its database."})
    response = client.post("/retrieve", json={"query": "What database did Project Alpha choose?"})
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert len(evidence) == 1
    assert "PostgreSQL" in evidence[0]["text"]
    assert evidence[0]["relevance"] >= 0.35


def test_memories_survive_application_restart(client: TestClient, tmp_path: Path) -> None:
    created = client.post("/context", json={"text": "Project Alpha uses PostgreSQL."}).json()
    settings = Settings(
        database_path=tmp_path / "context.db",
        artifact_dir=tmp_path / "artifacts",
    )
    app = create_app(settings)
    with TestClient(app) as restarted:
        assert restarted.get(f"/memory/{created['id']}").status_code == 200


def test_app_shutdown_closes_ollama_clients(tmp_path: Path, monkeypatch) -> None:
    class CloseSpy:
        closed = False

        async def close(self) -> None:
            self.closed = True

    processor_client = CloseSpy()
    reasoning_client = CloseSpy()
    monkeypatch.setattr(
        "backend.app.main.ContextProcessor",
        lambda settings, database: SimpleNamespace(client=processor_client),
    )
    monkeypatch.setattr(
        "backend.app.main.ReasoningClient",
        lambda settings: SimpleNamespace(client=reasoning_client),
    )
    app = create_app(
        Settings(
            database_path=tmp_path / "context.db",
            artifact_dir=tmp_path / "artifacts",
        )
    )

    with TestClient(app):
        pass

    assert processor_client.closed
    assert reasoning_client.closed


def test_retrieval_ignores_query_without_meaningful_terms(client: TestClient) -> None:
    client.post("/context", json={"text": "Project Alpha uses PostgreSQL."})
    response = client.post("/retrieve", json={"query": "What is it?"})
    assert response.status_code == 200
    assert response.json()["evidence"] == []


def test_verify_reports_support_and_conflicting_evidence(client: TestClient) -> None:
    supporting = client.post(
        "/context",
        json={"text": "Project Alpha uses PostgreSQL as its database."},
    ).json()
    conflicting = client.post(
        "/context",
        json={"text": "Project Alpha selected MongoDB, not PostgreSQL."},
    ).json()
    response = client.post(
        "/verify",
        json={
            "claim": "Project Alpha uses PostgreSQL as its database.",
            "evidence_ids": [supporting["id"], conflicting["id"]],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "contradictory"
    assert response.json()["supported"] is False


def test_chat_trace_persists_only_redacted_query_and_summary(client: TestClient) -> None:
    from backend.app.reasoning.client import ReasoningClaim, ReasoningResult

    created = client.post(
        "/context",
        json={"text": "Project Alpha uses PostgreSQL as its database."},
    ).json()

    class GroundedReasoner:
        async def answer(self, query, evidence):
            return ReasoningResult(
                answer="PostgreSQL",
                claims=[
                    ReasoningClaim(text="PostgreSQL", evidence_ids=[created["id"]])
                ],
            )

    cast(FastAPI, client.app).state.reasoning = GroundedReasoner()
    response = client.post(
        "/chat",
        json={"query": "Which database does Project Alpha use?"},
    )
    assert response.json()["verified"] is True
    trace = client.get("/trace").json()
    assert trace["query"] == "[redacted]"
    assert len(trace["query_hash"]) == 64
    assert "claim" not in json.dumps(trace["verification"])
