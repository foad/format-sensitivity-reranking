#!/usr/bin/env bash
# The answer axis: within-query format sensitivity for one split.
#
# Usage:
#   bash scripts/h1/within_query.sh [SPLIT]
#
# Environment:
#   GPUS         comma-separated GPUs to spread models across (default 0)
#   MODELS       comma-separated registry slugs (default the whole roster)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   BATCH_SIZE   scoring batch size (default 32)
#   LIMIT        cap on queries, for a smoke run
#   FORCE        1 to rebuild the cache and re-score every model

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

MEASURE=scripts/h1/within_query.py
DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h1"
mkdir -p "$OUT_DIR"

COMMON=(--data-root "$DATA_ROOT" --split "$SPLIT"
        --mode "${MODE:-both}" --batch-size "${BATCH_SIZE:-32}")
[ -n "${LIMIT:-}" ] && COMMON+=(--limit "$LIMIT")
SUFFIX=""
[ -n "${LIMIT:-}" ] && SUFFIX="_limit${LIMIT}"

job_command() { JOB_CMD=("$MEASURE" "${COMMON[@]}" --models "$1"
                         --out-tag "${1}${SUFFIX}"
                         --progress-file "$(job_progress "$1")"); }
job_output() { echo "$OUT_DIR/${SPLIT}_within_with_body_${1}${SUFFIX}.json"; }
job_log() { echo "$OUT_DIR/${SPLIT}_within_${1}${SUFFIX}.log"; }
# The watcher reads this; it is machine state, so it stays out of the log.
job_progress() { echo "$OUT_DIR/${SPLIT}_within_${1}${SUFFIX}.progress"; }

if [ -n "${MODELS:-}" ]; then
    IFS=', ' read -ra MODEL_LIST <<< "$MODELS"
else
    ROSTER_LOG="$OUT_DIR/within_roster.log"
    mapfile -t MODEL_LIST < <(fsr_roster "$MEASURE" "$ROSTER_LOG")
    rm -f "$ROSTER_LOG" "${ROSTER_LOG}.hare.log"
fi
[ "${#MODEL_LIST[@]}" -gt 0 ] || { echo "ERROR: no models to score" >&2; exit 1; }

# Shared cache for candidate lists.
CACHE="$OUT_DIR/${SPLIT}_candidates${SUFFIX}.json"
if [ ! -f "$CACHE" ] || [ "${FORCE:-0}" = "1" ]; then
    echo "building candidate lists for $SPLIT"
    PREPARE=("${COMMON[@]}" --prepare-only)
    [ "${FORCE:-0}" = "1" ] && PREPARE+=(--force)
    GPU="$(fsr_gpus | head -1)" fsr_run \
        "$OUT_DIR/${SPLIT}_candidates${SUFFIX}.log" \
        "$MEASURE" "${PREPARE[@]}"
else
    echo "reusing candidate lists at $CACHE"
fi

fsr_dispatch "${MODEL_LIST[@]}"
echo "done: $OUT_DIR/${SPLIT}_within_{with_body,metadata_only}_*${SUFFIX}.json"
