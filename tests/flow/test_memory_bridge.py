
from fastapi.testclient import TestClient


def test_general_question_is_not_saved(client: TestClient):
    result = client.post(
        "/memory/observe",
        json={"message": "What's caching?"},
    ).json()

    assert result["saved"] is False
    assert result["inference_ms"] == 0
    assert client.get("/memory").json()["total"] == 0


def test_automatic_project_memory_survives(client: TestClient):
    saved = client.post(
        "/memory/observe",
        json={
            "message": "I'm gonna use MongoDB for my Job Listing Portal"
        },
    )

    assert saved.status_code == 200
    assert saved.json()["saved"] is True
    assert saved.json()["inference_ms"] == 0

    recalled = client.post(
        "/memory/recall",
        json={
            "query": "What database did we choose for Job Listing Portal?",
            "project": "Job Listing Portal",
        },
    )

    assert recalled.status_code == 200
    result = recalled.json()
    assert result["status"] == "source_backed_excerpts"
    assert "MongoDB" in result["answer"]
    assert result["evidence"]
    assert result["inference_ms"] == 0


def test_repeated_observation_does_not_duplicate_memory(
    client: TestClient,
):
    data = {
        "message": "I'm gonna use MongoDB for my Job Listing Portal"
    }

    first = client.post("/memory/observe", json=data).json()
    second = client.post("/memory/observe", json=data).json()

    assert first["saved"] is True
    assert second["saved"] is False
    assert second["reason"] == "already_stored"
    assert first["memory_id"] == second["memory_id"]


def test_hypothetical_is_not_stored(client: TestClient):
    result = client.post(
        "/memory/observe",
        json={
            "message": (
                "Maybe we will use PostgreSQL "
                "for my Job Listing Portal"
            )
        },
    ).json()

    assert result["saved"] is False


def test_unknown_project_does_not_invent_memory(
    client: TestClient,
):
    result = client.post(
        "/memory/recall",
        json={
            "query": "What did we choose for project UNKNOWN-79?",
            "project": "UNKNOWN-79",
        },
    ).json()

    assert result["status"] == "insufficient_evidence"
    assert not result["evidence"]
