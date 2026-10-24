from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=50_000)
    source: str | None = Field(default=None, max_length=500)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("text must not be blank")
        return value


class Fact(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    category: Literal["fact", "decision", "constraint", "preference", "other"] = "fact"


class MemorySummary(BaseModel):
    id: str
    content: str
    facts: list[Fact]
    source: str | None
    created_at: str


class ContextCreated(MemorySummary):
    artifact_url: str
    metadata_url: str
    extraction_method: Literal["ollama", "source_text"]


class MemoryDetail(MemorySummary):
    metadata: dict[str, str | int | float | bool | None]
    artifact_url: str
    metadata_url: str


class MemoryList(BaseModel):
    items: list[MemorySummary]
    limit: int
    offset: int
    total: int


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=5_000)
    limit: int = Field(default=5, ge=1, le=20)


class Evidence(BaseModel):
    memory_id: str
    text: str
    source: str | None
    relevance: float


class RetrieveResponse(BaseModel):
    query: str
    evidence: list[Evidence]


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=5_000)


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


class ClaimVerification(BaseModel):
    claim: str
    supported: bool
    contradictory: bool = False
    evidence_ids: list[str]
    reason: str


class ChatResponse(BaseModel):
    answer: str
    status: Literal["verified", "unverified", "insufficient_evidence"]
    verified: bool
    retry_count: int
    claims: list[ClaimVerification]
    evidence: list[Evidence]
