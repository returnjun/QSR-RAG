from __future__ import annotations

import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"[\w-]+", flags=re.UNICODE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def normalize_exact(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def split_sentences(text: str) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_PATTERN.split(text) if part.strip()]
    return parts or [text]
