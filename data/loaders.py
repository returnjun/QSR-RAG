from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import DatasetStats, Document, MultiHopExample


def load_multihop_dataset(
    path: str | Path,
    *,
    dataset: str = "auto",
    limit: int | None = None,
) -> list[MultiHopExample]:
    """Load a local multi-hop QA dataset from JSON, JSONL, or CSV.

    The loader normalizes common fields from HotpotQA, 2WikiMultiHopQA,
    MuSiQue-style data, and poisoned benchmark variants into one shape.
    """

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")

    raw_rows = _read_rows(data_path)
    examples: list[MultiHopExample] = []
    dataset_name = _infer_dataset_name(data_path, dataset)

    for row in raw_rows:
        examples.append(_normalize_example(row, dataset=dataset_name))
        if limit is not None and len(examples) >= limit:
            break

    return examples


def compute_stats(examples: Iterable[MultiHopExample]) -> DatasetStats:
    examples_list = list(examples)
    documents = [doc for example in examples_list for doc in example.documents]
    return DatasetStats(
        total_examples=len(examples_list),
        total_documents=len(documents),
        poisoned_documents=sum(1 for doc in documents if doc.is_poisoned),
        supporting_documents=sum(1 for doc in documents if doc.is_supporting is True),
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "examples", "items", "rows"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return [payload]
        raise ValueError(f"Unsupported JSON root type in {path}: {type(payload).__name__}")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Reading .parquet requires pandas and pyarrow. "
                "Install them with: python -m pip install pandas pyarrow"
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError(
        f"Unsupported dataset format: {suffix}. Use .json, .jsonl, .csv, or .parquet."
    )


def _normalize_example(row: dict[str, Any], *, dataset: str) -> MultiHopExample:
    example_id = str(_first_present(row, "_id", "id", "qid", "question_id", default=""))
    question = str(_first_present(row, "question", "query", "input", default="")).strip()
    answer = _normalize_answer(
        _first_present(
            row,
            "answer",
            "answers",
            "final_answer",
            "target",
            "gold_answer",
            default=None,
        )
    )

    if not example_id:
        example_id = _stable_id(question, answer)

    documents = _normalize_documents(row, example_id=example_id)
    known_fields = {
        "_id",
        "id",
        "qid",
        "question_id",
        "question",
        "query",
        "input",
        "answer",
        "answers",
        "final_answer",
        "target",
        "gold_answer",
        "context",
        "contexts",
        "paragraphs",
        "documents",
        "retrieved_docs",
        "supporting_facts",
        "evidences",
        "hops",
    }
    metadata = {key: value for key, value in row.items() if key not in known_fields}

    return MultiHopExample(
        example_id=example_id,
        question=question,
        answer=answer,
        documents=documents,
        supporting_facts=_maybe_json(
            _first_present(row, "supporting_facts", "evidences", default=None)
        ),
        hops=row.get("hops"),
        dataset=dataset,
        metadata=metadata,
    )


def _normalize_documents(row: dict[str, Any], *, example_id: str) -> list[Document]:
    raw_docs = _first_present(
        row,
        "documents",
        "retrieved_docs",
        "paragraphs",
        "context",
        "contexts",
        default=[],
    )
    raw_docs = _maybe_json(raw_docs)
    supporting_titles = _supporting_titles(row.get("supporting_facts"))

    if isinstance(raw_docs, str):
        return [
            Document(
                doc_id=f"{example_id}:doc0",
                title="context",
                text=raw_docs,
                source=_optional_str(_first_present(row, "source", "url", default=None)),
                timestamp=_optional_str(
                    _first_present(
                        row,
                        "evidence_timestamp",
                        "timestamp",
                        "date",
                        "time_scope",
                        default=None,
                    )
                ),
                is_supporting=None,
            )
        ]

    documents: list[Document] = []
    if not isinstance(raw_docs, list):
        return documents

    for index, raw_doc in enumerate(raw_docs):
        doc = _normalize_document(
            raw_doc,
            example_id=example_id,
            index=index,
            supporting_titles=supporting_titles,
        )
        if doc.text:
            documents.append(doc)

    return documents


def _normalize_document(
    raw_doc: Any,
    *,
    example_id: str,
    index: int,
    supporting_titles: set[str],
) -> Document:
    if isinstance(raw_doc, dict):
        title = str(_first_present(raw_doc, "title", "name", "doc_title", default=f"doc_{index}"))
        text_value = _first_present(raw_doc, "text", "contents", "paragraph_text", "context", default="")
        text = _text_from_value(text_value)
        sentence_list = _sentence_list_from_value(
            _first_present(raw_doc, "sentences", default=None) or text_value
        )
        doc_id = str(_first_present(raw_doc, "id", "doc_id", "document_id", default=f"{example_id}:doc{index}"))
        is_supporting = _first_present(raw_doc, "is_supporting", "supporting", "is_gold", default=None)
        if is_supporting is None and title in supporting_titles:
            is_supporting = True

        metadata = {
            key: value
            for key, value in raw_doc.items()
            if key
            not in {
                "id",
                "doc_id",
                "document_id",
                "title",
                "name",
                "doc_title",
                "text",
                "contents",
                "paragraph_text",
                "context",
                "sentences",
                "source",
                "url",
                "timestamp",
                "date",
                "time_scope",
                "is_supporting",
                "supporting",
                "is_gold",
                "is_poisoned",
                "poisoned",
                "poison_type",
                "attack_type",
            }
        }
        if sentence_list:
            metadata["sentences"] = sentence_list

        return Document(
            doc_id=doc_id,
            title=title,
            text=text,
            source=_optional_str(_first_present(raw_doc, "source", "url", default=None)),
            timestamp=_optional_str(_first_present(raw_doc, "timestamp", "date", "time_scope", default=None)),
            is_supporting=_optional_bool(is_supporting),
            is_poisoned=bool(_first_present(raw_doc, "is_poisoned", "poisoned", default=False)),
            poison_type=_optional_str(_first_present(raw_doc, "poison_type", "attack_type", default=None)),
            metadata=metadata,
        )

    if isinstance(raw_doc, list) and len(raw_doc) >= 2:
        title = str(raw_doc[0])
        text_value = raw_doc[1]
        text = _text_from_value(text_value)
        sentence_list = _sentence_list_from_value(text_value)
        return Document(
            doc_id=f"{example_id}:doc{index}",
            title=title,
            text=text,
            is_supporting=True if title in supporting_titles else None,
            metadata={"sentences": sentence_list} if sentence_list else {},
        )

    return Document(
        doc_id=f"{example_id}:doc{index}",
        title=f"doc_{index}",
        text=_text_from_value(raw_doc),
    )


def _supporting_titles(raw_supporting_facts: Any) -> set[str]:
    raw_supporting_facts = _maybe_json(raw_supporting_facts)
    titles: set[str] = set()
    if not isinstance(raw_supporting_facts, list):
        return titles

    for item in raw_supporting_facts:
        if isinstance(item, (list, tuple)) and item:
            titles.add(str(item[0]))
        elif isinstance(item, dict):
            title = _first_present(item, "title", "doc_title", "name", default=None)
            if title is not None:
                titles.add(str(title))
    return titles


def _infer_dataset_name(path: Path, dataset: str) -> str:
    if dataset != "auto":
        return dataset

    lower_name = path.name.lower()
    if "hotpot" in lower_name:
        return "hotpotqa"
    if "2wiki" in lower_name or "two_wiki" in lower_name:
        return "2wikimultihopqa"
    if "musique" in lower_name:
        return "musique"
    if "poison" in lower_name:
        return "poisoned_multihop"
    return "unknown"


def _first_present(mapping: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _text_from_value(value: Any) -> str:
    value = _maybe_json(value)
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip())
    if value is None:
        return ""
    return str(value).strip()


def _sentence_list_from_value(value: Any) -> list[str]:
    value = _maybe_json(value)
    if not isinstance(value, list):
        return []
    sentences = [str(part).strip() for part in value if str(part).strip()]
    return sentences


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _normalize_answer(value: Any) -> str | None:
    value = _maybe_json(value)
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("text", "answer", "answers", "value"):
            if key in value:
                return _normalize_answer(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            return _normalize_answer(value[0])
        return " | ".join(str(item) for item in value)
    text = str(value).strip()
    return text or None


def _stable_id(question: str, answer: Any) -> str:
    seed = f"{question}|{answer}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
