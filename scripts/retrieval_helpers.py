"""Shared retrieval setup and support metrics for the main QSR-RAG experiment."""

from __future__ import annotations

import argparse
from typing import Any

from data.schema import MultiHopExample
from retriever import RetrievalConfig
from retriever.dense import DenseRetriever


DATASET_DEFAULTS = {
    "hotpotqa": {
        "data": "data/raw/hotpotqa/hotpot_dev_distractor_v1.json",
        "index_dir": "indexes/hotpotqa_dev",
    },
    "2wikimultihopqa": {
        "data": "data/raw/2wikimultihopqa/dev.parquet",
        "index_dir": "indexes/2wikimultihopqa_dev",
    },
    "musique": {
        "data": "data/raw/musique/musique_ans_v1.0_dev.jsonl",
        "index_dir": "indexes/musique_dev",
    },
}


def build_retriever(args: argparse.Namespace) -> DenseRetriever:
    return DenseRetriever(
        args.index_dir,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        config=RetrievalConfig(
            retriever_mode="dense_only",
            dense_only_top_k=args.dense_only_top_k,
            candidate_top_k=args.candidate_top_k,
            document_top_k=args.document_top_k,
            sentence_top_k=args.sentence_top_k,
            sentence_window_radius=args.sentence_window_radius,
            sentence_max_per_document=args.sentence_max_per_document,
            sentence_max_per_title=args.sentence_max_per_title,
            sentence_preselect_per_doc=args.sentence_preselect_per_doc,
            sentence_global_candidate_limit=args.sentence_global_candidate_limit,
            sentence_candidate_limit=args.sentence_candidate_limit,
            sentence_document_top_k=args.sentence_document_top_k,
            embedding_batch_size=args.embedding_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            embedding_max_length=args.embedding_max_length,
            reranker_max_length=args.reranker_max_length,
        ),
    )


def support_metric(*, gold_count: int, hit_count: int,
                   gold_items: list[Any], hit_items: list[Any]) -> dict[str, Any]:
    return {
        "gold_count": gold_count,
        "hit_count": hit_count,
        "recall": hit_count / gold_count if gold_count else None,
        "any_recalled": bool(gold_count and hit_count > 0),
        "fully_recalled": bool(gold_count and hit_count == gold_count),
        "gold_items": gold_items,
        "hit_items": hit_items,
    }


def summarize_support_recall(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in metrics if int(item.get("gold_count") or 0) > 0]
    total_gold = sum(int(item.get("gold_count") or 0) for item in eligible)
    total_hits = sum(int(item.get("hit_count") or 0) for item in eligible)
    def average(values):
        items = [float(value) for value in values if value is not None]
        return sum(items) / len(items) if items else None
    return {
        "eligible_questions": len(eligible),
        "average_recall": average(item.get("recall") for item in eligible),
        "micro_recall": total_hits / total_gold if total_gold else None,
        "any_recall_rate": average(1.0 if item.get("any_recalled") else 0.0 for item in eligible),
        "full_recall_rate": average(1.0 if item.get("fully_recalled") else 0.0 for item in eligible),
        "total_gold": total_gold,
        "total_hits": total_hits,
    }


def gold_supporting_titles(example: MultiHopExample) -> set[str]:
    titles = {normalize_title(doc.title) for doc in example.documents if doc.is_supporting is True}
    titles.update(title for title, _ in gold_supporting_facts(example))
    return {title for title in titles if title}


def gold_supporting_facts(example: MultiHopExample) -> set[tuple[str, int]]:
    facts: set[tuple[str, int]] = set()
    if not isinstance(example.supporting_facts, list):
        return facts
    for item in example.supporting_facts:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sentence_index = normalize_title(str(item[0])), _optional_int(item[1])
        elif isinstance(item, dict):
            title = normalize_title(str(item.get("title") or item.get("doc_title") or item.get("name") or ""))
            sentence_index = _optional_int(item.get("sent_id", item.get("sentence_index", item.get("index"))))
        else:
            continue
        if title and sentence_index is not None:
            facts.add((title, sentence_index))
    return facts


def normalize_title(title: str) -> str:
    return " ".join(title.strip().casefold().split())


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
