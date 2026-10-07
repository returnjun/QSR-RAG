"""Convert the official 2WikiMultiHopQA dev.json to the expected Parquet path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/2wikimultihopqa/dev.json")
    parser.add_argument("--output", default="data/raw/2wikimultihopqa/dev.parquet")
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        parser.error("expected a JSON array of 2Wiki examples")
    columns = ["_id", "type", "question", "context", "supporting_facts", "evidences", "answer"]
    missing = [column for column in columns if any(column not in row for row in rows)]
    if missing:
        parser.error(f"missing required columns: {missing}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[columns].to_parquet(target, index=False)
    print(f"Wrote {len(rows)} examples to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
