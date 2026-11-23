"""Compare models: the E4 metrics of each model's runs, side by side.

    poetry run python scripts/compare_models.py RUN_ID [RUN_ID ...] [--name NAME]

Runs are grouped by the model they were configured with (MODEL_NAME, stored
in runs.config.model). The gateway may have answered some calls with a
fallback model: `models_served` in comparison.json shows it, and a group
whose calls were not all answered by one model is flagged.

Writes results/compare-<name>/: comparison.json (every metric per model),
comparison.csv and comparison_table.tex (the paper rows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.evaluate_run import collect, latex_table, summary_rows
from src.repositories.mongo_repository import MongoRepository


def group_by_model(repository: MongoRepository, run_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for run in repository.runs.find({"_id": {"$in": run_ids}}):
        groups.setdefault(str((run.get("config") or {}).get("model")), []).append(run["_id"])
    missing = sorted(set(run_ids) - {r for ids in groups.values() for r in ids})
    if missing:
        raise ValueError(f"unknown run id(s): {', '.join(missing)}")
    return {model: sorted(ids) for model, ids in sorted(groups.items())}


def compare(repository: MongoRepository, run_ids: list[str], out: Path) -> dict[str, Any]:
    columns = {}
    for model, ids in group_by_model(repository, run_ids).items():
        metrics = collect(repository, ids)["metrics"]
        metrics["mixed_models_served"] = len(metrics["models_served"]) > 1
        columns[model] = metrics
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(columns, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame({model: dict(summary_rows(m)) for model, m in columns.items()}).to_csv(
        out / "comparison.csv"
    )
    (out / "comparison_table.tex").write_text(
        latex_table(columns, "Models behind the same gate", "tab:models"), encoding="utf-8"
    )
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models across stored runs.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--name", default="models", help="results/compare-<name>/")
    parser.add_argument(
        "--tex",
        type=Path,
        default=None,
        help="Also write the LaTeX table here (e.g. the paper's tables/).",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Column names, e.g. nvidia-network-slicing-model=Nemotron,"
        "gemini-network-slicing-model=Gemini",
    )
    args = parser.parse_args()
    out = Path("results") / f"compare-{args.name}"
    try:
        columns = compare(MongoRepository(), args.run_ids, out)
    except ValueError as exc:
        parser.error(str(exc))
    print(pd.DataFrame({model: dict(summary_rows(m)) for model, m in columns.items()}).to_string())
    for model, metrics in columns.items():
        if metrics["mixed_models_served"]:
            print(
                f"Note: {model} runs were partly answered by fallback models: "
                f"{metrics['models_served']}"
            )
    if args.tex:
        labels = dict(pair.split("=", 1) for pair in args.labels.split(",") if "=" in pair)
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(
            latex_table(
                {labels.get(m, m): v for m, v in columns.items()},
                "Models behind the same gate",
                "tab:models",
            ),
            encoding="utf-8",
        )
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
