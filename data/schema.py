from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A normalized evidence document used by hop-level RAG experiments."""

    doc_id: str
    title: str
    text: str
    source: str | None = None
    timestamp: str | None = None
    is_supporting: bool | None = None
    is_poisoned: bool = False
    poison_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultiHopExample:
    """A normalized multi-hop QA example across HotpotQA/2Wiki/MuSiQue-like files."""

    example_id: str
    question: str
    answer: str | None
    documents: list[Document]
    supporting_facts: Any = None
    hops: Any = None
    dataset: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetStats:
    total_examples: int
    total_documents: int
    poisoned_documents: int
    supporting_documents: int

    @property
    def avg_documents_per_example(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.total_documents / self.total_examples
