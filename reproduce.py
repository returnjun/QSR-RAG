"""Run the six main QSR-RAG evaluations reported in the paper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.retrieval_helpers import DATASET_DEFAULTS


MODELS = ("gpt4o_mini", "qwen3_8b_non_thinking")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_DEFAULTS),
                        default=list(DATASET_DEFAULTS))
    parser.add_argument("--num-questions", type=int, default=500)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--embedding-model", default="models/bge-m3")
    parser.add_argument("--reranker-model", default="models/bge-reranker-v2-m3")
    args = parser.parse_args()
    if not 1 <= args.num_questions <= 500:
        parser.error("--num-questions must be between 1 and 500")
    if not Path(args.config).is_file():
        parser.error(f"missing model configuration: {args.config}")
    for dataset in args.datasets:
        paths = DATASET_DEFAULTS[dataset]
        for kind, path in (("dataset", paths["data"]), ("index", paths["index_dir"])):
            if not Path(path).exists():
                parser.error(f"missing {kind} for {dataset}: {path}")
        manifest = Path("manifests") / f"{dataset}_seed43_500.jsonl"
        if not manifest.is_file():
            parser.error(f"missing fixed question list: {manifest}")
    for model in args.models:
        for dataset in args.datasets:
            output_dir = Path("outputs/main") / model / dataset
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{dataset}_{model}_seed43"
            command = [
                sys.executable, "scripts/evaluate_iterative_question_resolution.py",
                "--dataset", dataset,
                "--selection", "random",
                "--seed", "43",
                "--num-questions", str(args.num_questions),
                "--question-ids-from", str(Path("manifests") / f"{dataset}_seed43_500.jsonl"),
                "--config", args.config,
                "--embedding-model", args.embedding_model,
                "--reranker-model", args.reranker_model,
                "--generator-model-config", model,
                "--answer-model-config", model,
                "--adapter-model-config", model,
                "--rewriter-model-config", model,
                "--candidate-top-k", "200",
                "--dense-only-top-k", "200",
                "--document-top-k", "60",
                "--sentence-top-k", "20",
                "--details-output", str(output_dir / f"{stem}_details.jsonl"),
                "--output", str(output_dir / f"{stem}_summary.json"),
            ]
            print(f"\n=== {model} / {dataset} / {args.num_questions} questions ===", flush=True)
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
