import re
from backend.app.schemas import Evidence
from backend.app.storage.database import Database, MemoryRecord

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
PROJECT_RE = re.compile(r"\b[Pp]roject\s+(?:named\s+|codename\s*:\s*)?([A-Z][A-Za-z0-9_-]*)\b")
STOP_WORDS = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be", "been",
    "before", "but", "by", "can", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "i", "in", "into", "is", "it", "its", "of", "on",
    "or", "our", "please", "should", "that", "the", "their", "them", "there",
    "these", "this", "those", "to", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your",
}


def meaningful_tokens(text: str) -> set[str]:
    return {t.casefold() for t in TOKEN_RE.findall(text) if t.casefold() not in STOP_WORDS}


class MemoryRetriever:
    def __init__(self, database: Database) -> None:
        self.database = database

    def retrieve(self, query: str, *, limit: int = 5) -> list[Evidence]:
        tokens = meaningful_tokens(query)
        if not tokens or limit < 1:
            return []
        projects = {p.casefold() for p in PROJECT_RE.findall(query)}
        identifiers = {
            t.casefold() for t in TOKEN_RE.findall(query)
            if t.isupper() and ("-" in t or "_" in t or any(c.isdigit() for c in t))
        }
        search_limit = max(limit * 5, 20)
        if self.database.fts_enabled:
            expression = " OR ".join(f'"{t}"' for t in sorted(tokens))
            candidates: list[MemoryRecord] = self.database.search_fts(expression, limit=search_limit)
        else:
            candidates = self.database.search_like(sorted(tokens), limit=search_limit)
        ranked = []
        for memory in candidates:
            facts = memory["facts"]
            content = str(memory["content"]) if memory["metadata"].get("kind") == "explicit_memory" else (" ".join(str(f["text"]) for f in facts) if facts else str(memory["content"]))
            content_tokens = meaningful_tokens(content)
            content_projects = {p.casefold() for p in PROJECT_RE.findall(content)}
            if projects and not projects.issubset(content_projects):
                continue
            if identifiers and not identifiers.issubset(content_tokens):
                continue
            relevance = len(tokens & content_tokens) / len(tokens)
            minimum = 0 if projects or identifiers else (0.35 if len(tokens) >= 4 else 1 / len(tokens))
            if relevance < minimum:
                continue
            ranked.append((relevance, memory, content))
        ranked.sort(key=lambda x: (x[0], str(x[1]["created_at"])), reverse=True)
        return [
            Evidence(memory_id=str(m["id"]), text=text, source=m["source"], relevance=round(score, 4))
            for score, m, text in ranked[:limit]
        ]
