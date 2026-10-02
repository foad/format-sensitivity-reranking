#!/usr/bin/env bash
# Within-query measurement for one split, across every model in the roster.
#
# Usage:
#   bash scripts/h1/within_query.sh [SPLIT]
#
# Environment:
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   BATCH_SIZE   scoring batch size (default 32)
#   OUT_TAG      output file tag (default within_query)
#   FORCE        1 to re-run when the output is already present

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
OUT_TAG="${OUT_TAG:-within_query}"
OUT_DIR="$DATA_ROOT/h1_within_query"
OUT_JSON="$OUT_DIR/${SPLIT}_${OUT_TAG}.json"
OUT_LOG="$OUT_DIR/${SPLIT}_${OUT_TAG}.log"
mkdir -p "$OUT_DIR"

if [ -f "$OUT_JSON" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Skipping $SPLIT: $OUT_JSON is already present. Set FORCE=1 to re-run."
    exit 0
fi

echo "within-query measurement: split=$SPLIT"
fsr_run "$OUT_LOG" scripts/h1/within_query.py \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --out-tag "$OUT_TAG"

if [ ! -f "$OUT_JSON" ]; then
    echo "FAILED: $OUT_JSON was not written. See $OUT_LOG" >&2
    exit 1
fi
echo "done"
echo "  json: $OUT_JSON"
echo "  log:  $OUT_LOG"
