from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import faiss
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loaders import load_multihop_dataset
from retriever.encoders import BGEEncoder
from retriever.types import CorpusDocument


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the main experiment's dense retrieval index.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="models/bge-m3")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--preserve-example-documents",
        action="store_true",
        help="Keep duplicate passages separately for per-example scoped retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_multihop_dataset(args.data, dataset=args.dataset, limit=args.limit)
    documents = collect_documents(
        examples,
        preserve_example_documents=args.preserve_example_documents,
    )
    if not documents:
        raise SystemExit("No documents found in dataset.")

    texts = [document_text_for_index(doc) for doc in documents]
    encoder = BGEEncoder(args.embedding_model, device=args.embedding_device)
    dense_batches: list[np.ndarray] = []

    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        dense_batches.append(
            encoder.encode_dense(batch, batch_size=args.batch_size, max_length=args.max_length)
        )
        print(f"Encoded {min(start + len(batch), len(texts))}/{len(texts)} documents")

    dense = np.vstack(dense_batches).astype(np.float32)
    dense_index = faiss.IndexFlatIP(dense.shape[1])
    dense_index.add(dense)
    write_documents(output_dir / "documents.jsonl", documents)
    faiss.write_index(dense_index, str(output_dir / "dense.faiss"))
    manifest = {
        "format_version": 1,
        "document_count": len(documents),
        "dense_dimension": int(dense.shape[1]),
        "model_path": args.embedding_model,
        "max_length": args.max_length,
        "preserve_example_documents": args.preserve_example_documents,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Index written to {output_dir.resolve()}")


def collect_documents(
    examples,
    *,
    preserve_example_documents: bool = False,
) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    seen: set[str] = set()
    for example in examples:
        for doc in example.documents:
            identity = f"{doc.title}\n{doc.text}"
            if preserve_example_documents:
                identity = f"{example.example_id}\n{doc.doc_id}\n{identity}"
            key = hashlib.sha1(identity.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            documents.append(
                CorpusDocument(
                    doc_id=f"doc-{key[:20]}",
                    title=doc.title,
                    text=doc.text,
                    source=doc.source or example.dataset,
                    timestamp=doc.timestamp,
                    entities=list(doc.metadata.get("entities", [])) if isinstance(doc.metadata, dict) else [],
                    sentences=list(doc.metadata.get("sentences", [])) if isinstance(doc.metadata, dict) else [],
                    metadata={
                        "original_doc_id": doc.doc_id,
                        "example_id": example.example_id,
                        "is_supporting": doc.is_supporting,
                        "is_poisoned": doc.is_poisoned,
                        "poison_type": doc.poison_type,
                    },
                )
            )
    return documents


def document_text_for_index(document: CorpusDocument) -> str:
    return f"{document.title}\n{document.text}".strip()


def write_documents(path: Path, documents: list[CorpusDocument]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for doc in documents:
            file.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "text": doc.text,
                        "source": doc.source,
                        "timestamp": doc.timestamp,
                        "entities": doc.entities,
                        "sentences": doc.sentences,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
