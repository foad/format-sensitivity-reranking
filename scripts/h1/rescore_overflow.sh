#!/usr/bin/env bash
# Re-score the H1 passages that exceeded a model's context, one job per model.
#
# Reads `{split}_overflow.json`, written by `scripts/h1/overflow.py`.
#
# Usage:
#   GPUS=0 bash scripts/h1/rescore_overflow.sh [SPLIT]
#
# Environment:
#   MODELS       comma-separated registry slugs (default the six base models)
#   GPUS         comma-separated GPUs to spread models across (default 0)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   FORCE        1 to re-score a model whose output is present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

SPLIT="${1:-test}"
case "$SPLIT" in
    test | nq_val) ;;
    *)
        echo "ERROR: SPLIT must be test or nq_val (got $SPLIT)" >&2
        exit 2
        ;;
esac

DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h1"
[ -f "$OUT_DIR/${SPLIT}_overflow.json" ] || {
    echo "ERROR: $OUT_DIR/${SPLIT}_overflow.json missing; run scripts/h1/overflow.py." >&2
    exit 2
}

base_roster() {
    PYTHONPATH="$REPO_ROOT/src" python3 -c "
from fsr.models.registry import BASE_MODELS
for model in BASE_MODELS:
    print(model.slug)
"
}

if [ -n "${MODELS:-}" ]; then
    IFS=', ' read -ra MODEL_LIST <<< "$MODELS"
else
    mapfile -t MODEL_LIST < <(base_roster)
fi

job_command() { JOB_CMD=(scripts/h1/rescore_overflow.py --data-root "$DATA_ROOT"
                         --split "$SPLIT" --model "$1"
                         --progress-file "$(job_progress "$1")"); }
job_output() { echo "$OUT_DIR/${SPLIT}_rescore_${1}.json"; }
job_log() { echo "$OUT_DIR/${SPLIT}_rescore_${1}.log"; }
job_progress() { echo "$OUT_DIR/${SPLIT}_rescore_${1}.progress"; }

echo "H1 overflow re-score"
echo "  models:  ${MODEL_LIST[*]}"
echo "  split:   $SPLIT"
echo "  results: $OUT_DIR/"

STARTED=$(date +%s)
fsr_dispatch "${MODEL_LIST[@]}"
echo
echo "re-score complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
