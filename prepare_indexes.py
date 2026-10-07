"""Build the three dense indexes from the complete development sets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.retrieval_helpers import DATASET_DEFAULTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_DEFAULTS),
                        default=list(DATASET_DEFAULTS))
    parser.add_argument("--embedding-model", default="models/bge-m3")
    args = parser.parse_args()
    for dataset in args.datasets:
        paths = DATASET_DEFAULTS[dataset]
        if not Path(paths["data"]).is_file():
            parser.error(f"missing development set for {dataset}: {paths['data']}")
        if (Path(paths["index_dir"]) / "dense.faiss").is_file():
            print(f"Existing index: {paths['index_dir']}", flush=True)
            continue
        command = [sys.executable, "scripts/build_index.py",
                   "--data", paths["data"], "--dataset", dataset,
                   "--output-dir", paths["index_dir"],
                   "--embedding-model", args.embedding_model]
        print(f"\n=== Building {dataset} index ===", flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
