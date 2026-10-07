from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class EvidencePoolItem:
    evidence_id: str
    stage: str
    source_query: str
    source_query_type: str
    resolution_target: str | None
    resolution_answer_type: str | None
    resolution_answer_subtype: str | None
    doc_id: str | None
    title: str
    text: str
    rank: int
    score: float | None
    focus_sentence: str
    context_before: str
    context_after: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_evidence_pool_items(
    evidence: Iterable[Any],
    *,
    stage: str,
    source_query: str,
    source_query_type: str,
    resolution_target: str | None = None,
    resolution_answer_type: str | None = None,
    resolution_answer_subtype: str | None = None,
    step: int = 0,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for rank, evidence_item in enumerate(evidence, start=1):
        item = EvidencePoolItem(
            evidence_id=str(_item_get(evidence_item, "evidence_id", "") or ""),
            stage=stage,
            source_query=source_query,
            source_query_type=source_query_type,
            resolution_target=resolution_target,
            resolution_answer_type=resolution_answer_type,
            resolution_answer_subtype=resolution_answer_subtype,
            doc_id=_optional_str(_item_get(evidence_item, "doc_id", None)),
            title=str(_item_get(evidence_item, "title", "") or "").strip(),
            text=str(_item_get(evidence_item, "text", "") or "").strip(),
            rank=rank,
            score=_optional_float(
                _item_get(
                    evidence_item,
                    "rerank_score",
                    _item_get(evidence_item, "score", None),
                )
            ),
            focus_sentence=str(
                _item_get(evidence_item, "focus_sentence", "") or ""
            ).strip(),
            context_before=str(
                _item_get(evidence_item, "context_before", "") or ""
            ).strip(),
            context_after=str(
                _item_get(evidence_item, "context_after", "") or ""
            ).strip(),
            metadata={
                "step": step,
                "evidence_aliases": [],
                "source_queries": [source_query],
                "stages": [stage],
                "supporting_evidence_used_by_answerer": False,
                "answer": None,
                "sentence_index": _optional_int(
                    _item_get(evidence_item, "sentence_index", None)
                ),
                "covered_sentence_indices": _int_list(
                    _item_get(evidence_item, "covered_sentence_indices", None)
                ),
            },
        ).to_dict()
        if item["title"] and item["text"]:
            items.append(item)
    return items


def mark_answer_support(
    items: list[dict[str, Any]],
    *,
    supporting_evidence_ids: Iterable[str],
    answer: str,
) -> None:
    supporting_ids = {str(item) for item in supporting_evidence_ids}
    for item in items:
        metadata = item.setdefault("metadata", {})
        is_supporting = item.get("evidence_id") in supporting_ids
        metadata["supporting_evidence_used_by_answerer"] = is_supporting
        metadata["answer"] = answer


def merge_evidence_pool(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}

    for group in groups:
        for original in group:
            item = _copy_item(original)
            key = evidence_key(item)
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                _initialize_merge_metadata(item)
                by_key[key] = item
                merged.append(item)
                continue
            _merge_duplicate(existing, item)

    return merged


def evidence_key(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "").strip().casefold()
    text = str(item.get("text") or "").strip().casefold()
    if not title or not text:
        return ""
    return hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()


def _initialize_merge_metadata(item: dict[str, Any]) -> None:
    metadata = item.setdefault("metadata", {})
    metadata["evidence_aliases"] = _dedupe(
        [*metadata.get("evidence_aliases", []), str(item.get("evidence_id") or "")]
    )
    metadata["source_queries"] = _dedupe(
        [*metadata.get("source_queries", []), str(item.get("source_query") or "")]
    )
    metadata["stages"] = _dedupe(
        [*metadata.get("stages", []), str(item.get("stage") or "")]
    )


def _merge_duplicate(existing: dict[str, Any], duplicate: dict[str, Any]) -> None:
    existing_metadata = existing.setdefault("metadata", {})
    duplicate_metadata = duplicate.get("metadata") or {}
    existing_metadata["evidence_aliases"] = _dedupe(
        [
            *existing_metadata.get("evidence_aliases", []),
            *duplicate_metadata.get("evidence_aliases", []),
            str(duplicate.get("evidence_id") or ""),
        ]
    )
    existing_metadata["source_queries"] = _dedupe(
        [
            *existing_metadata.get("source_queries", []),
            *duplicate_metadata.get("source_queries", []),
            str(duplicate.get("source_query") or ""),
        ]
    )
    existing_metadata["stages"] = _dedupe(
        [
            *existing_metadata.get("stages", []),
            *duplicate_metadata.get("stages", []),
            str(duplicate.get("stage") or ""),
        ]
    )
    existing_metadata["supporting_evidence_used_by_answerer"] = bool(
        existing_metadata.get("supporting_evidence_used_by_answerer")
        or duplicate_metadata.get("supporting_evidence_used_by_answerer")
    )
    if duplicate_metadata.get("answer"):
        existing_metadata["answer"] = duplicate_metadata["answer"]


def _copy_item(item: dict[str, Any]) -> dict[str, Any]:
    copied = dict(item)
    copied["metadata"] = dict(item.get("metadata") or {})
    return copied


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[int] = []
    for item in value:
        parsed = _optional_int(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
