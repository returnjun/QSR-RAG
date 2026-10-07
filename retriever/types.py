from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CorpusDocument:
    doc_id: str
    title: str
    text: str
    source: str | None = None
    timestamp: str | None = None
    entities: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankedHit:
    doc_index: int
    score: float


@dataclass(slots=True)
class RankedDocument:
    doc_index: int
    doc_id: str
    title: str
    text: str
    score: float
    channel_scores: dict[str, float] = field(default_factory=dict)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None


@dataclass(slots=True)
class SentenceWindow:
    doc_index: int
    doc_id: str
    title: str
    sentence: str
    window: str
    sentence_index: int
    covered_sentence_indices: list[int]
    score: float
    rerank_score: float | None = None


@dataclass(slots=True)
class RetrievalResult:
    query: str
    documents: list[RankedDocument]
    sentences: list[SentenceWindow]
    diagnostics: dict[str, Any] = field(default_factory=dict)
