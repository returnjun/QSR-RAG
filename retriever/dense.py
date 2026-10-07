from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from .config import RetrievalConfig
from .encoders import BGEEncoder, BGEReranker
from .sentence import build_sentence_windows, document_sentences, select_diverse_sentence_windows
from .text import tokenize
from .types import CorpusDocument, RankedDocument, RankedHit, RetrievalResult, SentenceWindow


class DenseRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedding_model: str = "models/bge-m3",
        reranker_model: str = "models/bge-reranker-v2-m3",
        embedding_device: str | None = "auto",
        reranker_device: str | None = "auto",
        config: RetrievalConfig | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.config = config or RetrievalConfig()
        self.manifest = self._load_json(self.index_dir / "manifest.json", default={})
        self.documents = self._load_documents(self.index_dir / "documents.jsonl")
        self.dense_index = faiss.read_index(str(self.index_dir / "dense.faiss"))
        self.encoder = BGEEncoder(embedding_model, device=embedding_device)
        self.reranker = BGEReranker(reranker_model, device=reranker_device)
        self.allowed_doc_indices: set[int] | None = None
        self.scope_example_id: str | None = None

    def set_example_scope(self, example_id: str | None) -> None:
        """Restrict subsequent retrieval calls to one example's LongBench context."""

        if example_id is None:
            self.set_document_scope(None)
            return
        key = str(example_id)
        allowed = {
            index
            for index, document in enumerate(self.documents)
            if str(document.metadata.get("example_id") or "") == key
        }
        if not allowed:
            raise ValueError(f"No indexed documents found for example_id={key}")
        self.set_document_scope(allowed, label=key)

    def set_document_scope(
        self,
        doc_indices: Iterable[int] | None,
        *,
        label: str | None = None,
    ) -> None:
        """Restrict retrieval to an explicit subset of indexed documents."""

        if doc_indices is None:
            self.allowed_doc_indices = None
            self.scope_example_id = None
            return
        allowed = {
            int(index)
            for index in doc_indices
            if 0 <= int(index) < len(self.documents)
        }
        if not allowed:
            raise ValueError("Document scope must contain at least one indexed document")
        self.allowed_doc_indices = allowed
        self.scope_example_id = str(label or "custom_document_scope")


    def retrieve(
        self,
        query: str,
        *,
        query_variants: Iterable[str] | None = None,
    ) -> RetrievalResult:
        queries = [query]
        if query_variants:
            queries.extend(variant for variant in query_variants if variant and variant != query)

        (
            candidates,
            channel_scores,
            channel_ranks,
            candidate_diagnostics,
        ) = self.get_candidates(
            query,
            query_variants=queries[1:],
        )
        reranked_documents = (
            self._rerank_documents(
                query, candidates, channel_scores, channel_ranks
            )
            if self.config.enable_document_rerank
            else self._unreranked_documents(
                candidates, channel_scores, channel_ranks
            )
        )
        sentence_windows = self._rerank_sentences(
            query, candidates, reranked_documents
        )

        return RetrievalResult(
            query=query,
            documents=reranked_documents[: self.config.document_top_k],
            sentences=sentence_windows[: self.config.sentence_top_k],
            diagnostics={
                "index_dir": str(self.index_dir),
                "document_count": len(self.documents),
                **candidate_diagnostics,
                "retriever_mode": self.config.retriever_mode,
                "candidate_top_k": self.config.candidate_top_k,
                "candidate_count": len(candidates),
                "candidate_documents": [
                    {
                        "rank": rank,
                        "doc_index": hit.doc_index,
                        "doc_id": self.documents[hit.doc_index].doc_id,
                        "title": self.documents[hit.doc_index].title,
                        "score": float(hit.score),
                    }
                    for rank, hit in enumerate(candidates, start=1)
                ],
                "document_top_k": self.config.document_top_k,
                "sentence_top_k": self.config.sentence_top_k,
                "sentence_window_radius": self.config.sentence_window_radius,
                "sentence_document_top_k": self.config.sentence_document_top_k,
                "sentence_max_per_title": self.config.sentence_max_per_title,
                "enable_document_rerank": self.config.enable_document_rerank,
                "enable_sentence_rerank": self.config.enable_sentence_rerank,
                "document_reranker_calls": int(
                    self.config.enable_document_rerank and bool(candidates)
                ),
                "sentence_reranker_calls": int(
                    self.config.enable_sentence_rerank and bool(sentence_windows)
                ),
                "scope_example_id": self.scope_example_id,
                "scope_document_count": (
                    len(self.allowed_doc_indices)
                    if self.allowed_doc_indices is not None
                    else len(self.documents)
                ),
            },
        )

    def get_candidates(
        self,
        query: str,
        *,
        query_variants: Iterable[str] | None = None,
    ) -> tuple[
        list[RankedHit],
        dict[int, dict[str, float]],
        dict[int, dict[str, int]],
        dict[str, object],
    ]:
        """Use BGE-M3 dense search; merge multiple query variants by score."""

        queries = [query]
        if query_variants:
            queries.extend(variant for variant in query_variants if variant and variant != query)
        dense_hits = self._merge_variant_hits(
            self._dense_search(variant, self.config.dense_only_top_k)
            for variant in queries
        )[: self.config.candidate_top_k]
        channel_scores = {hit.doc_index: {"dense": hit.score} for hit in dense_hits}
        channel_ranks = {
            hit.doc_index: {"dense": rank}
            for rank, hit in enumerate(dense_hits, start=1)
        }
        return dense_hits, channel_scores, channel_ranks, {
            "channel_counts": {"dense": len(dense_hits)},
            "channel_call_counts": {"dense": len(queries)},
            "fusion_calls": 0,
            "dense_search_top_k": self.config.dense_only_top_k,
        }

    def _dense_search(self, query: str, top_k: int) -> list[RankedHit]:
        if top_k <= 0:
            return []
        vector = self.encoder.encode_dense(
            [query],
            batch_size=1,
            max_length=self.config.embedding_max_length,
        )
        search_k = len(self.documents) if self.allowed_doc_indices is not None else top_k
        scores, indices = self.dense_index.search(vector.astype(np.float32), search_k)
        hits = [
            RankedHit(int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if int(index) >= 0
        ]
        return self._filter_scoped_hits(hits, top_k)



    def _filter_scoped_hits(
        self,
        hits: list[RankedHit],
        top_k: int,
    ) -> list[RankedHit]:
        if self.allowed_doc_indices is None:
            return hits[:top_k]
        return [
            hit for hit in hits if hit.doc_index in self.allowed_doc_indices
        ][:top_k]

    def _rerank_documents(
        self,
        query: str,
        fused: list[RankedHit],
        channel_scores: dict[int, dict[str, float]],
        channel_ranks: dict[int, dict[str, int]],
    ) -> list[RankedDocument]:
        candidates = fused[: self.config.candidate_top_k]
        views = [self._document_rerank_view(query, hit.doc_index) for hit in candidates]
        rerank_scores = self.reranker.score(
            query,
            views,
            batch_size=self.config.reranker_batch_size,
            max_length=self.config.reranker_max_length,
        )
        ranked: list[RankedDocument] = []
        for hit, rerank_score in zip(candidates, rerank_scores):
            doc = self.documents[hit.doc_index]
            ranked.append(
                RankedDocument(
                    doc_index=hit.doc_index,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=float(hit.score),
                    channel_scores=channel_scores.get(hit.doc_index, {}),
                    channel_ranks=channel_ranks.get(hit.doc_index, {}),
                    rerank_score=float(rerank_score),
                )
            )
        ranked.sort(key=lambda item: (item.rerank_score if item.rerank_score is not None else item.score), reverse=True)
        return ranked

    def _unreranked_documents(
        self,
        fused: list[RankedHit],
        channel_scores: dict[int, dict[str, float]],
        channel_ranks: dict[int, dict[str, int]],
    ) -> list[RankedDocument]:
        return [
            RankedDocument(
                doc_index=hit.doc_index,
                doc_id=self.documents[hit.doc_index].doc_id,
                title=self.documents[hit.doc_index].title,
                text=self.documents[hit.doc_index].text,
                score=float(hit.score),
                channel_scores=channel_scores.get(hit.doc_index, {}),
                channel_ranks=channel_ranks.get(hit.doc_index, {}),
                rerank_score=None,
            )
            for hit in fused[: self.config.candidate_top_k]
        ]

    def _document_rerank_view(self, query: str, doc_index: int) -> str:
        document = self.documents[doc_index]
        sentences = document_sentences(document)
        first = sentences[: self.config.document_rerank_first_sentences]
        snippets = self._best_snippets(query, sentences, self.config.document_rerank_snippets)
        parts = [document.title, *first, *snippets]
        deduped: list[str] = []
        seen: set[str] = set()
        for part in parts:
            key = part.casefold()
            if part and key not in seen:
                deduped.append(part)
                seen.add(key)
        return "\n".join(deduped)

    def _best_snippets(self, query: str, sentences: list[str], limit: int) -> list[str]:
        query_terms = set(tokenize(query))
        scored: list[tuple[float, str]] = []
        for sentence in sentences:
            terms = set(tokenize(sentence))
            overlap = len(query_terms & terms)
            if overlap:
                scored.append((float(overlap) / max(len(terms), 1), sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [sentence for _, sentence in scored[:limit]]

    def _rerank_sentences(
        self,
        query: str,
        fused: list[RankedHit],
        reranked_documents: list[RankedDocument],
    ) -> list[SentenceWindow]:
        doc_order = [hit.doc_index for hit in fused]
        rerank_position = {doc.doc_index: index for index, doc in enumerate(reranked_documents)}
        doc_order.sort(key=lambda doc_index: rerank_position.get(doc_index, len(reranked_documents)))
        if self.config.sentence_document_top_k > 0:
            doc_order = doc_order[: self.config.sentence_document_top_k]

        candidates: list[SentenceWindow] = []
        for doc_index in doc_order:
            document = self.documents[doc_index]
            sentences = document_sentences(document)
            windows = build_sentence_windows(
                doc_index,
                sentences,
                radius=self.config.sentence_window_radius,
                doc_id=document.doc_id,
                title=document.title,
            )
            preselected = self._preselect_windows(query, windows, self.config.sentence_preselect_per_doc)
            candidates.extend(preselected)
            if (
                self.config.sentence_global_candidate_limit > 0
                and len(candidates) >= self.config.sentence_global_candidate_limit
            ):
                candidates = candidates[: self.config.sentence_global_candidate_limit]
                break

        if self.config.sentence_candidate_limit > 0:
            candidates = candidates[: self.config.sentence_candidate_limit]
        if not candidates:
            return []

        scores = (
            self.reranker.score(
                query,
                [window.window for window in candidates],
                batch_size=self.config.reranker_batch_size,
                max_length=self.config.reranker_max_length,
            )
            if self.config.enable_sentence_rerank
            else [window.score for window in candidates]
        )
        ranked: list[SentenceWindow] = []
        for window, score in zip(candidates, scores):
            window.score = float(score)
            window.rerank_score = float(score)
            ranked.append(window)
        ranked.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else item.score, reverse=True)
        return select_diverse_sentence_windows(
            ranked,
            self.documents,
            top_k=self.config.sentence_top_k,
            max_per_document=self.config.sentence_max_per_document,
            max_per_title=self.config.sentence_max_per_title,
        )

    def _preselect_windows(
        self,
        query: str,
        windows: list[SentenceWindow],
        limit: int,
    ) -> list[SentenceWindow]:
        if limit <= 0:
            return windows
        query_terms = set(tokenize(query))
        scored: list[SentenceWindow] = []
        for window in windows:
            terms = set(tokenize(window.window))
            overlap = len(query_terms & terms)
            window.score = float(overlap) / max(len(terms), 1)
            scored.append(window)
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    @staticmethod
    def _merge_variant_hits(hit_lists: Iterable[list[RankedHit]]) -> list[RankedHit]:
        merged: dict[int, float] = {}
        for hits in hit_lists:
            for hit in hits:
                merged[hit.doc_index] = max(merged.get(hit.doc_index, float("-inf")), hit.score)
        return [
            RankedHit(doc_index, score)
            for doc_index, score in sorted(merged.items(), key=lambda item: item[1], reverse=True)
        ]


    @staticmethod
    def _load_json(path: Path, *, default):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_documents(path: Path) -> list[CorpusDocument]:
        documents: list[CorpusDocument] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                documents.append(
                    CorpusDocument(
                        doc_id=str(row.get("doc_id", "")),
                        title=str(row.get("title", "")),
                        text=str(row.get("text", "")),
                        source=row.get("source"),
                        timestamp=row.get("timestamp"),
                        entities=list(row.get("entities") or []),
                        sentences=list(row.get("sentences") or []),
                        metadata=dict(row.get("metadata") or {}),
                    )
                )
        return documents
