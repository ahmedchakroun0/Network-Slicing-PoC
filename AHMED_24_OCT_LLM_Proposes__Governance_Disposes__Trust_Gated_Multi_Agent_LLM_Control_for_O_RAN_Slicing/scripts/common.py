"""Shared helpers for the reproducible experiments.

Every experiment imports the *unmodified* PoC code from the project
directory given with --project (default: ../network-slicing-PoC), so the
numbers reported in the paper are produced by the code under study, not by
a re-implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
FIG_DIR = HERE.parent / "figures"
KPIS = ("latency", "jitter", "packet_loss", "throughput")


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--project",
        type=Path,
        default=HERE.parent.parent / "network-slicing-PoC",
        help="Path to the network-slicing-PoC project root.",
    )
    p.add_argument("--seed", type=int, default=2026)
    return p


def use_project(project: Path) -> Path:
    """Put the PoC on sys.path and chdir into it (its prompts use relative
    paths such as prompts/monitoring.txt)."""
    project = project.resolve()
    if not (project / "agents" / "ran_trust_governance.py").is_file():
        raise SystemExit(f"Not a network-slicing-PoC checkout: {project}")
    sys.path.insert(0, str(project))
    os.chdir(project)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return project


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def dump_json(obj, name: str) -> Path:
    path = DATA_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def set_outdir(sub: str | None, module_globals: dict) -> None:
    """Write this experiment's outputs to data/<sub>/ (e.g. data/coloran/)."""
    global DATA_DIR
    if not sub:
        return
    DATA_DIR = HERE.parent / "data" / sub
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    module_globals["DATA_DIR"] = DATA_DIR
