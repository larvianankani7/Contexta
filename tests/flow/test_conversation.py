from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.reasoning.client import ReasoningResult


class FakeGeneral:
    def __init__(self) -> None:
        self.calls = []

    async def answer_general(self, query: str, history=None, memory=None) -> ReasoningResult:
        self.calls.append((query, history, memory))
        return ReasoningResult(answer="A normal conversational answer.", claims=[])


def test_explicit_lock_is_saved_without_an_extra_model_call(client: TestClient) -> None:
    reasoner = FakeGeneral()
    cast(FastAPI, client.app).state.reasoning = reasoner
    response = client.post("/conversation", json={"query": "Remember that Project AETHER-47 uses MariaDB 11.4."})
    assert response.status_code == 200
    assert response.json()["saved_memory_id"]
    assert reasoner.calls == []
    memory_id = response.json()["saved_memory_id"]
    detail = client.get(f"/memory/{memory_id}")
    assert "MariaDB 11.4" in detail.json()["content"]
    assert detail.json()["metadata"]["kind"] == "explicit_memory"


def test_repeated_lock_does_not_duplicate_memory(client: TestClient) -> None:
    message = "Remember that Project AETHER-47 uses MariaDB 11.4."
    one = client.post("/conversation", json={"query": message}).json()
    two = client.post("/conversation", json={"query": message}).json()
    assert one["saved_memory_id"] == two["saved_memory_id"]
    assert client.get("/memory").json()["total"] == 1


def test_recall_is_source_backed_and_skips_inference(client: TestClient) -> None:
    reasoner = FakeGeneral()
    cast(FastAPI, client.app).state.reasoning = reasoner
    stored = client.post("/conversation", json={"query": "Remember that Project AETHER-47 uses MariaDB 11.4."}).json()
    recalled = client.post("/conversation", json={"query": "What did we decide for Project AETHER-47?"})
    assert recalled.status_code == 200
    result = recalled.json()
    assert result["status"] == "verified"
    assert "MariaDB 11.4" in result["answer"]
    assert stored["saved_memory_id"] in {e["memory_id"] for e in result["evidence"]}
    assert not reasoner.calls


def test_general_chat_uses_history_and_no_evidence(client: TestClient) -> None:
    reasoner = FakeGeneral()
    cast(FastAPI, client.app).state.reasoning = reasoner
    response = client.post("/conversation", json={"query": "Explain hashing.", "history": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]})
    assert response.json()["status"] == "unverified"
    assert reasoner.calls[0][1][0]["content"] == "Hello"
    assert not response.json()["evidence"]


def test_unavailable_private_fact_is_not_invented(client: TestClient) -> None:
    reasoner = FakeGeneral()
    cast(FastAPI, client.app).state.reasoning = reasoner
    response = client.post("/conversation", json={"query": "What did we decide for Project UNKNOWN-58?"})
    assert response.json()["status"] == "insufficient_evidence"
    assert not reasoner.calls


def test_ambiguous_earlier_ref_finds_unique_project(client: TestClient) -> None:
    client.post("/conversation", json={"query": "The following decisions are locked: Project codename: AETHER-47. Database: MariaDB 11.4."})
    result = client.post("/conversation", json={"query": "What did we lock at the beginning?"}).json()
    assert result["status"] == "verified"
    assert "MariaDB 11.4" in result["answer"]


def test_nine_decisions_recalled_without_model_inference(client: TestClient) -> None:
    message = ("The following architecture decisions are locked:\n"
               "Project codename: AETHER-47\nDatabase: MariaDB 11.4\n"
               "Password hashing: Argon2id\nCache: Redis with a TTL of 47 seconds\n"
               "Access token expiration: 13 minutes\nRefresh token expiration: 9 days\n"
               "API rate limit: 73 requests per 11 minutes\nInternal service port: 8463\n"
               "Deployment region: Singapore")
    reasoner = FakeGeneral()
    cast(FastAPI, client.app).state.reasoning = reasoner
    created = client.post("/conversation", json={"query": message})
    assert created.status_code == 200
    recall = client.post("/conversation", json={"query": "What were all our locked architecture decisions for AETHER-47?"})
    assert recall.status_code == 200
    answer = recall.json()["answer"]
    for value in ("MariaDB 11.4", "Argon2id", "47 seconds", "13 minutes", "9 days", "73 requests per 11 minutes", "8463", "Singapore"):
        assert value in answer
    assert recall.json()["status"] == "verified"
    assert reasoner.calls == []
