import re

from backend.app.memory.retrieval import meaningful_tokens
from backend.app.reasoning.client import ReasoningResult
from backend.app.schemas import ClaimVerification, Evidence
from backend.app.storage.database import Database

ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
NEGATION_RE = re.compile(
    r"\b(?:not|never|no|neither|without|cannot|can't|isn't|aren't|wasn't|weren't|"
    r"don't|doesn't|didn't|won't)\b",
    re.IGNORECASE,
)


class EvidenceVerifier:
    def __init__(self, database: Database) -> None:
        self.database = database

    def verify(self, result: ReasoningResult, evidence: list[Evidence]) -> list[ClaimVerification]:
        available = {item.memory_id: item for item in evidence}
        checks: list[ClaimVerification] = []
        for claim in result.claims:
            cited = [memory_id for memory_id in claim.evidence_ids if memory_id in available]
            missing = [memory_id for memory_id in claim.evidence_ids if memory_id not in available]
            existing = {
                memory_id for memory_id in cited if self.database.get(memory_id) is not None
            }
            missing.extend(memory_id for memory_id in cited if memory_id not in existing)
            cited = [memory_id for memory_id in cited if memory_id in existing]
            claim_tokens = meaningful_tokens(claim.text)
            supporting = [
                memory_id
                for memory_id in cited
                if _supports(claim.text, claim_tokens, available[memory_id].text)
            ]
            contradictions = [
                item.memory_id
                for item in evidence
                if _contradicts(claim.text, claim_tokens, item.text)
            ]
            supported = bool(supporting) and not missing and not contradictions
            if missing:
                reason = "One or more cited memories do not exist in the retrieved evidence."
            elif contradictions:
                reason = "Retrieved evidence contains a conflicting value or negation."
            elif supported:
                reason = "Claim terms are present in the cited source-backed evidence."
            else:
                reason = "The cited evidence does not substantiate the claim."
            checks.append(
                ClaimVerification(
                    claim=claim.text,
                    supported=supported,
                    contradictory=bool(contradictions),
                    evidence_ids=supporting,
                    reason=reason,
                )
            )
        return checks

    @staticmethod
    def answer_supported(answer: str, checks: list[ClaimVerification]) -> bool:
        answer_tokens = meaningful_tokens(answer)
        claim_tokens = {
            token
            for check in checks
            if check.supported
            for token in meaningful_tokens(check.claim)
        }
        return bool(answer_tokens) and answer_tokens.issubset(claim_tokens)


def _supports(claim: str, claim_tokens: set[str], evidence: str) -> bool:
    evidence_tokens = meaningful_tokens(evidence)
    claim_numbers = set(NUMBER_RE.findall(claim))
    evidence_numbers = set(NUMBER_RE.findall(evidence))
    entities = {
        token.casefold() for token in ENTITY_RE.findall(claim)
    }.intersection(claim_tokens)
    if (
        not claim_tokens
        or not entities.issubset(evidence_tokens)
        or not claim_numbers.issubset(evidence_numbers)
        or bool(NEGATION_RE.search(claim)) != bool(NEGATION_RE.search(evidence))
    ):
        return False
    return len(claim_tokens.intersection(evidence_tokens)) / len(claim_tokens) >= 0.7


def _contradicts(claim: str, claim_tokens: set[str], evidence: str) -> bool:
    evidence_tokens = meaningful_tokens(evidence)
    claim_numbers = set(NUMBER_RE.findall(claim))
    evidence_numbers = set(NUMBER_RE.findall(evidence))
    claim_entities = {
        token.casefold() for token in ENTITY_RE.findall(claim)
    }.intersection(claim_tokens)
    evidence_entities = {
        token.casefold() for token in ENTITY_RE.findall(evidence)
    }.intersection(evidence_tokens)
    shared_context = claim_tokens.intersection(evidence_tokens)
    if (
        claim_numbers
        and evidence_numbers
        and not claim_numbers.issubset(evidence_numbers)
        and len(shared_context) / len(claim_tokens) >= 0.5
    ):
        return True
    if (
        not shared_context
        or not claim_entities
        or not claim_entities.intersection(evidence_entities)
    ):
        return False
    if bool(NEGATION_RE.search(claim)) != bool(NEGATION_RE.search(evidence)):
        return len(shared_context) / len(claim_tokens) >= 0.5
    return (
        len(shared_context) / len(claim_tokens) >= 0.7
        and bool(claim_entities - evidence_entities)
        and bool(evidence_entities - claim_entities)
    )
