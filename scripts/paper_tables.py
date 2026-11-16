"""The paper's live-LLM table: one column per named group of runs.

    poetry run python scripts/paper_tables.py \\
        --group "R1=claude-run2,claude-run3,claude-run4" --group "R2=claude-run5" \\
        --caption "..." --tex <paper>/tables/live.tex

Each group is evaluated as scripts/evaluate_run.py does (completed records
only) and becomes one column of the summary rows (evaluate_run.summary_rows).
The metrics of every group are also written next to the table (.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluate_run import collect, latex_table
from src.repositories.mongo_repository import MongoRepository


def parse_group(text: str) -> tuple[str, list[str]]:
    label, _, ids = text.partition("=")
    run_ids = [r.strip() for r in ids.split(",") if r.strip()]
    if not label.strip() or not run_ids:
        raise argparse.ArgumentTypeError(f"expected LABEL=run1,run2: {text!r}")
    return label.strip(), run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-LLM table of the paper.")
    parser.add_argument("--group", type=parse_group, action="append", required=True)
    parser.add_argument("--caption", default="Live LLM agents behind the gate")
    parser.add_argument("--label", default="tab:live")
    parser.add_argument("--tex", type=Path, required=True)
    args = parser.parse_args()

    repository = MongoRepository()
    columns = {label: collect(repository, run_ids)["metrics"] for label, run_ids in args.group}
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(latex_table(columns, args.caption, args.label), encoding="utf-8")
    args.tex.with_suffix(".json").write_text(json.dumps(columns, indent=2, default=float),
                                             encoding="utf-8")
    print(args.tex.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
