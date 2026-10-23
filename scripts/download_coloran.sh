#!/usr/bin/env bash
# Download the ColO-RAN dataset (per-UE slice traces only).
#
#   M. Polese, L. Bonati, S. D'Oro, S. Basagni, T. Melodia, "ColO-RAN:
#   Developing Machine Learning-Based xApps for Open RAN Closed-Loop Control
#   on Programmable Experimental Platforms", IEEE Trans. Mobile Computing,
#   22(10):5787-5800, 2023.  https://github.com/wineslab/colosseum-oran-coloran-dataset
#   License: GPL-3.0 (see LICENSE in the downloaded folder).
#
# The repository holds ~40k files. Only the files the preprocessing needs are
# fetched (sparse checkout of slices_bs*/<imsi>_metrics.csv).  One experiment
# run (exp1) is ~750 MB; all five runs are ~3.7 GB.
#
# Usage:
#   bash scripts/download_coloran.sh                 # exp1 -> data/raw/coloran
#   bash scripts/download_coloran.sh DEST exp1 exp2  # custom folder / runs
set -euo pipefail

DEST=${1:-data/raw/coloran}
shift || true
RUNS=${*:-exp1}
REPO=https://github.com/wineslab/colosseum-oran-coloran-dataset

mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  git clone --quiet --depth 1 --filter=blob:none --no-checkout "$REPO" "$DEST"
fi

cd "$DEST"
git sparse-checkout init --no-cone
PATTERNS=("/README.md" "/LICENSE" "/CITATION.cff")
for run in $RUNS; do
  PATTERNS+=("/rome_static_medium/sched*/tr*/$run/bs*/slices_bs*/*_metrics.csv")
done
git sparse-checkout set --no-cone "${PATTERNS[@]}"
git checkout --quiet HEAD

N=$(find rome_static_medium -name '*_metrics.csv' | wc -l)
echo "ColO-RAN downloaded to $DEST: $N per-UE trace files ($(du -sh rome_static_medium | cut -f1)) for runs: $RUNS"
