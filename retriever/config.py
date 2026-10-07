"""Retrieval settings used by the paper's dense main experiment."""

from __future__ import annotations

from dataclasses import dataclass


VALID_RETRIEVER_MODES = {"dense_only"}


@dataclass(slots=True)
class RetrievalConfig:
    retriever_mode: str = "dense_only"
    dense_only_top_k: int = 200
    candidate_top_k: int = 200
    document_top_k: int = 60
    sentence_top_k: int = 20
    embedding_max_length: int = 512
    reranker_max_length: int = 512
    embedding_batch_size: int = 8
    reranker_batch_size: int = 4
    document_rerank_first_sentences: int = 2
    document_rerank_snippets: int = 4
    sentence_window_radius: int = 1
    sentence_max_per_document: int = 3
    sentence_max_per_title: int = 3
    sentence_preselect_per_doc: int = 6
    sentence_global_candidate_limit: int = 240
    sentence_candidate_limit: int = 0
    sentence_document_top_k: int = 40
    enable_document_rerank: bool = True
    enable_sentence_rerank: bool = True

    def __post_init__(self) -> None:
        if self.retriever_mode != "dense_only":
            raise ValueError("The release supports dense_only retrieval")
        if self.dense_only_top_k <= 0 or self.candidate_top_k <= 0:
            raise ValueError("retrieval depth must be positive")
