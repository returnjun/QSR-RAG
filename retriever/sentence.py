from __future__ import annotations

from collections import defaultdict

from .text import split_sentences
from .types import CorpusDocument, SentenceWindow


def document_sentences(document: CorpusDocument) -> list[str]:
    if document.sentences:
        return [sentence.strip() for sentence in document.sentences if sentence.strip()]
    return split_sentences(document.text)


def build_sentence_windows(
    doc_index: int,
    sentences: list[str],
    *,
    radius: int = 1,
    doc_id: str = "",
    title: str = "",
) -> list[SentenceWindow]:
    windows: list[SentenceWindow] = []
    for index, sentence in enumerate(sentences):
        start = max(0, index - radius)
        end = min(len(sentences), index + radius + 1)
        covered = list(range(start, end))
        windows.append(
            SentenceWindow(
                doc_index=doc_index,
                doc_id=doc_id,
                title=title,
                sentence=sentence,
                window=" ".join(sentences[start:end]).strip(),
                sentence_index=index,
                covered_sentence_indices=covered,
                score=0.0,
            )
        )
    return windows


def select_diverse_sentence_windows(
    ranked: list[SentenceWindow],
    documents: list[CorpusDocument],
    *,
    top_k: int,
    max_per_document: int = 3,
    max_per_title: int = 3,
) -> list[SentenceWindow]:
    del documents
    selected: list[SentenceWindow] = []
    per_doc: dict[int, int] = defaultdict(int)
    per_title: dict[str, int] = defaultdict(int)
    seen_windows: set[tuple[str, str]] = set()
    for window in ranked:
        key = (window.doc_id, window.window.casefold())
        if key in seen_windows:
            continue
        if per_doc[window.doc_index] >= max_per_document:
            continue
        title_key = window.title.strip().casefold()
        if max_per_title > 0 and per_title[title_key] >= max_per_title:
            continue
        selected.append(window)
        seen_windows.add(key)
        per_doc[window.doc_index] += 1
        per_title[title_key] += 1
        if len(selected) >= top_k:
            break
    return selected
