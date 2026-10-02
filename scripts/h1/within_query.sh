#!/usr/bin/env bash
# Within-query measurement for one split, across every model in the roster.
#
# Usage:
#   bash scripts/h1/within_query.sh [SPLIT]
#
# Environment:
#   GPUS         comma-separated GPUs to spread models across (default 0)
#   GPU          a single GPU, an alias for GPUS
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

SPLIT="${1:-test}"
case "$SPLIT" in
    train | dev | test | nq_val) ;;
    *)
        echo "ERROR: SPLIT must be train, dev, test or nq_val (got $SPLIT)" >&2
        exit 2
        ;;
esac

DATA_ROOT="${DATA_ROOT:-data/nq}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_DIR="$DATA_ROOT/h1_within_query"
mkdir -p "$OUT_DIR"

IFS=', ' read -ra GPU_LIST <<< "${GPUS:-${GPU:-0}}"
if [ -n "${MODELS:-}" ]; then
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
else
    # Ask through the runner: the roster lives in the registry, and on a
    # cluster the only Python that can import it is inside the container.
    ROSTER="$OUT_DIR/roster.log"
    GPU="${GPU_LIST[0]}" fsr_run "$ROSTER" \
        scripts/h1/within_query.py --list-models > /dev/null
    mapfile -t MODEL_LIST < <(grep -E '^[a-z0-9_]+$' "$ROSTER")
fi
if [ "${#MODEL_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no models to score" >&2
    exit 1
fi

COMMON=(--data-root "$DATA_ROOT" --split "$SPLIT" --batch-size "$BATCH_SIZE")
[ -n "${LIMIT:-}" ] && COMMON+=(--limit "$LIMIT")
[ "${FORCE:-0}" = "1" ] && COMMON+=(--force)
SUFFIX=""
[ -n "${LIMIT:-}" ] && SUFFIX="_limit${LIMIT}"

# Shared cache.
CACHE="$OUT_DIR/${SPLIT}_candidates${SUFFIX}.json"
if [ ! -f "$CACHE" ] || [ "${FORCE:-0}" = "1" ]; then
    echo "building candidate lists for $SPLIT"
    GPU="${GPU_LIST[0]}" \
        fsr_run "$OUT_DIR/${SPLIT}_candidates${SUFFIX}.log" \
        scripts/h1/within_query.py "${COMMON[@]}" --prepare-only
else
    echo "reusing candidate lists at $CACHE"
fi

run_model() {
    local SLUG="$1" DEVICE="$2"
    local TAG="wq_${SLUG}${SUFFIX}"
    local OUT_JSON="$OUT_DIR/${SPLIT}_${TAG}.json"
    if [ -f "$OUT_JSON" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "  [skip] $SLUG: $OUT_JSON is already present"
        return 0
    fi
    echo "  [gpu $DEVICE] $SLUG"
    GPU="$DEVICE" fsr_run "$OUT_DIR/${SPLIT}_${TAG}.log" \
        scripts/h1/within_query.py "${COMMON[@]}" \
        --models "$SLUG" --out-tag "$TAG"
}

echo "scoring ${#MODEL_LIST[@]} models across ${#GPU_LIST[@]} GPUs"
FAILED=()
i=0
while [ "$i" -lt "${#MODEL_LIST[@]}" ]; do
    PIDS=()
    SLUGS=()
    for DEVICE in "${GPU_LIST[@]}"; do
        [ "$i" -lt "${#MODEL_LIST[@]}" ] || break
        SLUG="${MODEL_LIST[$i]}"
        run_model "$SLUG" "$DEVICE" &
        PIDS+=($!)
        SLUGS+=("$SLUG")
        i=$((i + 1))
    done
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || FAILED+=("${SLUGS[$j]}")
    done
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}" >&2
    exit 1
fi
echo "done: $OUT_DIR/${SPLIT}_wq_*${SUFFIX}.json"
