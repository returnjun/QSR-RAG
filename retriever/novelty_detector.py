from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

import numpy as np


EmbeddingFunction = Callable[[list[str]], Any]
QUESTION_SIMILARITY_THRESHOLD = 0.85

QUESTION_TYPE_PATTERNS = {
    "DIRECTOR_LOOKUP": [r"\bwho directed\b", r"\bwho is the director\b"],
    "BIRTH_DATE": [r"\bwhen was .* born\b", r"\bdate of birth\b"],
    "BIRTH_PLACE": [r"\bwhere was .* born\b", r"\bplace of birth\b"],
    "COUNTRY": [r"\bwhat country\b", r"\bwhich country\b", r"\bnationality\b"],
    "PARENT": [r"\bwho is .* father\b", r"\bwho is .* mother\b"],
    "RELEASE_DATE": [r"\bwhen (?:was|did) .* release", r"\brelease date\b"],
    "MEMBER_COUNT": [r"\bhow many members\b"],
}


@dataclass(slots=True)
class QuestionRecord:
    round: int
    question: str
    normalized_question: str
    question_type: str
    anchors: list[str]
    resolution_target: str = ""
    target_span: str = ""
    query_id: str = ""
    local_query_id: str = ""
    question_version: int = 0
    executed: bool = False
    answer: str = ""
    repeat_result: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "question": self.question,
            "normalized_question": self.normalized_question,
            "question_type": self.question_type,
            "anchors": list(self.anchors),
            "resolution_target": self.resolution_target,
            "target_span": self.target_span,
            "query_id": self.query_id,
            "local_query_id": self.local_query_id,
            "question_version": self.question_version,
            "executed": self.executed,
            "answer": self.answer,
            "repeat_result": dict(self.repeat_result),
        }


def build_question_record(
    *,
    question: str,
    round_index: int,
    resolution_target: str = "",
    target_span: str = "",
    query_id: str = "",
    local_query_id: str = "",
    question_version: int = 0,
    embedding_model: EmbeddingFunction | Any | None = None,
) -> QuestionRecord:
    normalized = normalize_question(question)
    embedding = embed_texts([question], embedding_model)[0] if embedding_model is not None else None
    return QuestionRecord(
        round=round_index,
        question=question.strip(),
        normalized_question=normalized,
        question_type=classify_question_type(question),
        anchors=extract_question_anchors(question),
        resolution_target=str(resolution_target or "").strip(),
        target_span=str(target_span or "").strip(),
        query_id=str(query_id or "").strip(),
        local_query_id=str(local_query_id or "").strip(),
        question_version=int(question_version),
        embedding=embedding,
    )


def classify_question_type(question: str) -> str:
    normalized = normalize_question(question)
    for question_type, patterns in QUESTION_TYPE_PATTERNS.items():
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
            return question_type
    return "OTHER"


def extract_question_anchors(question: str) -> list[str]:
    anchors: list[str] = []
    for quoted in re.findall(r'["“”]([^"“”]+)["“”]', question):
        _append_unique(anchors, quoted)
    proper_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9'.-]*|[A-Z])(?:\s+(?:(?:of|the|and|with|in)\s+)?(?:[A-Z][A-Za-z0-9'.-]*|[A-Z]))*\b"
    )
    ignored = {"who", "what", "when", "where", "which", "how", "are", "is", "did"}
    for match in proper_pattern.finditer(question):
        value = match.group(0).strip()
        if value.casefold() not in ignored:
            _append_unique(anchors, value)
    for number in re.findall(r"\b\d{3,4}\b", question):
        _append_unique(anchors, number)
    return anchors


def check_repetition(
    *,
    candidate: QuestionRecord,
    history: list[QuestionRecord],
    threshold: float = QUESTION_SIMILARITY_THRESHOLD,
    embedding_model: EmbeddingFunction | Any | None = None,
    dataset: str = "",
) -> dict[str, Any]:
    dataset_name = str(dataset or "").strip().casefold()
    best_similarity = 0.0
    best_index: int | None = None
    reason: str | None = None
    for index, previous in enumerate(history):
        similarity = question_similarity(candidate, previous, embedding_model)
        if similarity > best_similarity:
            best_similarity, best_index = similarity, index
        overlap = anchor_overlap(candidate.anchors, previous.anchors)
        if candidate.normalized_question == previous.normalized_question:
            return _repeat_result(True, "EXACT_NORMALIZED_MATCH", similarity, index, overlap)
        if dataset_name == "musique":
            if (
                candidate.question_type != "OTHER"
                and previous.question_type == candidate.question_type
                and similarity >= 0.97
                and overlap >= 0.5
            ):
                return _repeat_result(
                    True,
                    "HIGH_SIMILARITY_SAME_RELATION",
                    similarity,
                    index,
                    overlap,
                )
            continue
        if similarity >= threshold and overlap >= 0.5:
            return _repeat_result(True, "HIGH_SIMILARITY_SAME_ANCHOR", similarity, index, overlap)
        if (
            candidate.question_type != "OTHER"
            and previous.question_type == candidate.question_type
            and overlap >= 0.5
            and previous.answer
        ):
            return _repeat_result(True, "SAME_RESOLVED_QUERY_TYPE", similarity, index, overlap)
    return _repeat_result(False, reason, best_similarity, best_index, 0.0)


def question_similarity(
    left: QuestionRecord | str,
    right: QuestionRecord | str,
    embedding_model: EmbeddingFunction | Any | None = None,
) -> float:
    left_text = left.question if isinstance(left, QuestionRecord) else str(left)
    right_text = right.question if isinstance(right, QuestionRecord) else str(right)
    left_vector = left.embedding if isinstance(left, QuestionRecord) else None
    right_vector = right.embedding if isinstance(right, QuestionRecord) else None
    if left_vector is None or right_vector is None:
        if embedding_model is not None:
            vectors = embed_texts([left_text, right_text], embedding_model)
            left_vector, right_vector = vectors[0], vectors[1]
    if left_vector is not None and right_vector is not None:
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator:
            return float(np.dot(left_vector, right_vector) / denominator)
    return SequenceMatcher(None, normalize_question(left_text), normalize_question(right_text)).ratio()


def anchor_overlap(left: list[str], right: list[str]) -> float:
    left_set = {normalize_question(item) for item in left if item.strip()}
    right_set = {normalize_question(item) for item in right if item.strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def rewrite_has_no_progress(
    *,
    old_question: str,
    new_question: str,
    new_facts: list[dict[str, Any]],
    embedding_model: EmbeddingFunction | Any | None = None,
    similarity_threshold: float = 0.98,
) -> tuple[bool, float]:
    similarity = text_similarity(old_question, new_question, embedding_model)
    new_entities = [
        answer_item
        for fact in new_facts
        for answer_item in _fact_answer_items(fact)
    ]
    entity_added = any(
        answer and answer.casefold() in new_question.casefold() and answer.casefold() not in old_question.casefold()
        for answer in new_entities
    )
    return similarity >= similarity_threshold and not entity_added, similarity


def _fact_answer_items(fact: dict[str, Any]) -> list[str]:
    raw_items = fact.get("answer_items")
    if isinstance(raw_items, list):
        items = [str(item or "").strip() for item in raw_items]
        items = [item for item in items if item]
        if items:
            return items
    answer = str(fact.get("answer") or "").strip()
    return [answer] if answer else []


def text_similarity(
    left: str,
    right: str,
    embedding_model: EmbeddingFunction | Any | None = None,
) -> float:
    if embedding_model is not None:
        vectors = embed_texts([left, right], embedding_model)
        denominator = float(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        if denominator:
            return float(np.dot(vectors[0], vectors[1]) / denominator)
    return SequenceMatcher(None, normalize_question(left), normalize_question(right)).ratio()


def embed_texts(texts: list[str], embedding_model: EmbeddingFunction | Any) -> np.ndarray:
    if callable(embedding_model) and not hasattr(embedding_model, "encode_dense"):
        return np.asarray(embedding_model(texts), dtype=np.float32)
    encode_dense = getattr(embedding_model, "encode_dense", None)
    if not callable(encode_dense):
        raise TypeError("embedding_model must be callable or provide encode_dense")
    return np.asarray(
        encode_dense(texts, batch_size=max(1, len(texts)), max_length=512),
        dtype=np.float32,
    )


def normalize_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def _repeat_result(
    is_repeat: bool,
    reason: str | None,
    similarity: float,
    matched_history_index: int | None,
    overlap: float,
) -> dict[str, Any]:
    return {
        "is_repeat": is_repeat,
        "reason": reason,
        "similarity": similarity,
        "anchor_overlap": overlap,
        "matched_history_index": matched_history_index,
    }


def _append_unique(items: list[str], value: str) -> None:
    normalized = normalize_question(value)
    if normalized and all(normalize_question(item) != normalized for item in items):
        items.append(value.strip())
