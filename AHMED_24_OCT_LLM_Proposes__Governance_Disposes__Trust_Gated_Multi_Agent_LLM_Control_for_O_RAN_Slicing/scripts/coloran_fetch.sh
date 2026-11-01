#!/usr/bin/env bash
# Fetch the per-UE slice-metric traces of the ColO-RAN dataset
# (Polese et al., IEEE TMC 2022; GPL-3.0) for the selected experiment runs.
# Usage: bash coloran_fetch.sh <dest_dir> [exp ...]   (default: exp1 exp2)
set -euo pipefail
DEST=${1:?destination directory}
shift || true
EXPS=${*:-exp1 exp2}
mkdir -p "$DEST"
cd "$DEST"
if [ ! -d coloran/.git ]; then
  git clone -q --depth 1 --filter=blob:none --no-checkout \
    https://github.com/wineslab/colosseum-oran-coloran-dataset coloran
fi
cd coloran
git sparse-checkout init --no-cone
PATTERNS=()
for e in $EXPS; do PATTERNS+=("/rome_static_medium/sched*/tr*/$e/bs*/slices_bs*/*_metrics.csv"); done
git sparse-checkout set --no-cone "${PATTERNS[@]}" "/README.md" "/CITATION.cff" "/LICENSE"
git checkout -q HEAD
echo "files: $(find rome_static_medium -name '*_metrics.csv' | wc -l)"
du -sh rome_static_medium
