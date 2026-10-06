#!/usr/bin/env bash
# Cache the representations the head-capacity probe fits to.
#
# Usage:
#   MODELS=mxbai_v1,jina_v2 GPUS=0,1 bash scripts/h2/probe.sh
#
# Environment:
#   MODELS       comma-separated registry slugs (default mxbai_v1,jina_v2)
#   SPLITS       comma-separated splits (default train,dev)
#   GPUS         comma-separated GPUs to spread jobs across (default 0)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   NEGATIVES    negatives cached per record (default the script default)
#   EVAL_BATCH   pairs scored at once
#   LIMIT        cap on records, for a smoke pass
#   FORCE        1 to rebuild a cache that is present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

roster() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --list; }

DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h2/probe"
IFS=',' read -ra MODEL_LIST <<< "${MODELS:-mxbai_v1,jina_v2}"
IFS=',' read -ra SPLIT_LIST <<< "${SPLITS:-train,dev}"
mkdir -p "$OUT_DIR"

for NAME in "${MODEL_LIST[@]}"; do
    if ! roster | grep -qx "$NAME"; then
        echo "ERROR: unknown model '$NAME'. Pick from:" >&2
        roster >&2
        exit 2
    fi
done

ARGS=(--data-root "$DATA_ROOT")
[ -n "${EVAL_BATCH:-}" ] && ARGS+=(--batch-size "$EVAL_BATCH")
[ -n "${NEGATIVES:-}" ] && ARGS+=(--negatives "$NEGATIVES")
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)

tag() { set -- $1; echo "capture_${1}_${2}"; }

job_name() { tag "$1"; }
job_log() { echo "$OUT_DIR/$(tag "$1").log"; }
job_progress() { echo "$OUT_DIR/$(tag "$1").progress"; }
job_output() {
    set -- $1
    echo "$DATA_ROOT/h2/probe/${2}_features_${1}.npy"
}
job_command() {
    set -- $1
    JOB_CMD=(scripts/h2/capture_features.py --model "$1" --split "$2"
             "${ARGS[@]}" --progress-file "$(job_progress "$1 $2")")
}

KEYS=()
for SPLIT in "${SPLIT_LIST[@]}"; do
    for NAME in "${MODEL_LIST[@]}"; do
        KEYS+=("$NAME $SPLIT")
    done
done

echo "Head-probe feature capture"
echo "  models:  ${MODEL_LIST[*]}"
echo "  splits:  ${SPLIT_LIST[*]}"
echo "  results: $OUT_DIR/"

STARTED=$(date +%s)
fsr_dispatch "${KEYS[@]}"
echo
echo "feature capture complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
echo "  results: $OUT_DIR/"
