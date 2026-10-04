#!/usr/bin/env bash
# The score axis: cross-query format sensitivity for one split.
#
# Usage:
#   bash scripts/h1/cross_query.sh [SPLIT]
#
# Environment:
#   GPUS         comma-separated GPUs to spread models across (default 0)
#   MODELS       comma-separated registry slugs (default the whole roster)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   BATCH_SIZE   scoring batch size (default 16)
#   LIMIT        cap on records, for a smoke run
#   FORCE        1 to re-score every model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

SPLIT="${1:-test}"
case "$SPLIT" in
    train | dev | test | nq_val | all) ;;
    *)
        echo "ERROR: SPLIT must be train, dev, test, nq_val or all (got $SPLIT)" >&2
        exit 2
        ;;
esac

MEASURE=scripts/h1/cross_query.py
DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h1"
mkdir -p "$OUT_DIR"

COMMON=(--in-dir "$DATA_ROOT" --out-dir "$OUT_DIR" --split "$SPLIT"
        --batch-size "${BATCH_SIZE:-16}")
SUFFIX=""
if [ -n "${LIMIT:-}" ]; then
    COMMON+=(--limit "$LIMIT")
    SUFFIX="_limit${LIMIT}"
fi

job_command() { JOB_CMD=("$MEASURE" "${COMMON[@]}" --models "$1"
                         --out-tag "${1}${SUFFIX}"); }
# with_body is written after metadata_only, so it means the model finished.
job_output() { echo "$OUT_DIR/${SPLIT}_cross_with_body_${1}${SUFFIX}.json"; }
job_log() { echo "$OUT_DIR/${SPLIT}_cross_${1}${SUFFIX}.log"; }

if [ -n "${MODELS:-}" ]; then
    IFS=', ' read -ra MODEL_LIST <<< "$MODELS"
else
    ROSTER_LOG="$OUT_DIR/cross_roster.log"
    mapfile -t MODEL_LIST < <(fsr_roster "$MEASURE" "$ROSTER_LOG")
    rm -f "$ROSTER_LOG" "${ROSTER_LOG}.hare.log"
fi
[ "${#MODEL_LIST[@]}" -gt 0 ] || { echo "ERROR: no models to score" >&2; exit 1; }

fsr_dispatch "${MODEL_LIST[@]}"
echo "done: $OUT_DIR/${SPLIT}_cross_{with_body,metadata_only}_*${SUFFIX}.json"
