import json
from types import SimpleNamespace

import httpx


class UnavailableOllama:
    async def chat(self, **kwargs: object) -> None:
        raise httpx.ConnectError("offline")


class UnsupportedExtraction:
    async def chat(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(
                    {
                        "facts": [
                            {"text": "Project Alpha", "category": "preference"},
                            {"text": "PostgreSQL", "category": "decision"},
                            {"text": "database", "category": "decision"},
                        ]
                    }
                )
            )
        )


class StaticExtraction:
    def __init__(self, content: str | None, *, done_reason: str = "stop") -> None:
        self.content = content
        self.done_reason = done_reason
        self.request_options: dict[str, object] | None = None

    async def chat(self, **kwargs: object) -> SimpleNamespace:
        self.request_options = kwargs
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            done_reason=self.done_reason,
        )


def test_ollama_unavailable_preserves_source_without_inventing(client) -> None:
    text = "The team is considering PostgreSQL."
    client.app.state.processor.client = UnavailableOllama()
    response = client.post("/context", json={"text": text})
    assert response.status_code == 201
    assert response.json()["extraction_method"] == "source_text"
    assert response.json()["facts"][0]["text"] == text


def test_incomplete_extracted_facts_fall_back_to_source(client) -> None:
    client.app.state.processor.client = UnsupportedExtraction()
    text = "For Project Alpha, we decided to use PostgreSQL as the database."
    response = client.post("/context", json={"text": text})
    assert response.status_code == 201
    assert response.json()["extraction_method"] == "source_text"
    assert response.json()["facts"][0]["text"] == text


def test_valid_extraction_persists_multiple_supported_facts(client) -> None:
    text = "Project Alpha uses PostgreSQL. The API listens on port 8123."
    client.app.state.processor.client = StaticExtraction(
        json.dumps(
            {
                "facts": [
                    {"text": "Project Alpha uses PostgreSQL.", "category": "decision"},
                    {"text": "The API listens on port 8123.", "category": "fact"},
                ]
            }
        )
    )

    response = client.post("/context", json={"text": text})

    assert response.status_code == 201
    created = response.json()
    assert created["extraction_method"] == "ollama"
    assert [fact["text"] for fact in created["facts"]] == [
        "Project Alpha uses PostgreSQL.",
        "The API listens on port 8123.",
    ]
    record = client.app.state.database.get(created["id"])
    assert record is not None
    assert record["facts"] == created["facts"]
    assert client.get(created["artifact_url"]).headers["content-type"] == "image/png"
    metadata = client.get(created["metadata_url"]).json()
    assert metadata["extraction_method"] == "ollama"
    assert metadata["facts"] == created["facts"]


def test_qwen3_vl_extraction_uses_measured_token_cap(client) -> None:
    extraction = StaticExtraction(
        json.dumps({"facts": [{"text": "Project Alpha uses PostgreSQL.", "category": "fact"}]})
    )
    client.app.state.processor.client = extraction

    response = client.post(
        "/context",
        json={"text": "Project Alpha uses PostgreSQL.", "source": "token-cap-test"},
    )

    assert response.status_code == 201
    assert extraction.request_options is not None
    assert extraction.request_options["options"]["num_predict"] == 4096


def test_empty_extraction_output_falls_back_to_source(client) -> None:
    text = "Project Alpha uses PostgreSQL as its database."
    client.app.state.processor.client = StaticExtraction("   ")

    response = client.post("/context", json={"text": text})

    assert response.status_code == 201
    assert response.json()["extraction_method"] == "source_text"
    assert response.json()["facts"][0]["text"] == text


def test_malformed_extraction_json_falls_back_to_source(client) -> None:
    text = "Project Alpha uses PostgreSQL as its database."
    client.app.state.processor.client = StaticExtraction('{"facts": [')

    response = client.post("/context", json={"text": text})

    assert response.status_code == 201
    assert response.json()["extraction_method"] == "source_text"
    assert response.json()["facts"][0]["text"] == text


def test_truncated_extraction_is_rejected_even_if_json_is_valid(client) -> None:
    text = "Project Alpha uses PostgreSQL as its database."
    client.app.state.processor.client = StaticExtraction(
        json.dumps({"facts": [{"text": text, "category": "decision"}]}),
        done_reason="length",
    )

    response = client.post("/context", json={"text": text})

    assert response.status_code == 201
    assert response.json()["extraction_method"] == "source_text"
    assert response.json()["facts"][0]["text"] == text
