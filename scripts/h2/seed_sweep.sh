#!/usr/bin/env bash
# Run Phase 2 of H2 under further training seeds, one model on one GPU.
#
# Usage:
#   MODEL=jina_v2 GPU=2 bash scripts/h2/seed_sweep.sh
#
# Environment:
#   MODEL       registry slug, required
#   GPU         the one GPU the sweep runs on (default 0)
#   SEEDS       space-separated training seeds (default 1 2 3 4); seed 0 only
#               for a variant that reuses another model's weight
#   STAGES      run.sh stages per seed (default 4,5,6,7,8)
#   HOLD_GB     memory to hold on Hex, in GiB, or 0 for no hold (default 3)
#   HOLD_HOURS  release the hold after this long whatever happens (default 168)
#   FSR_RUNNER  runner to source (default scripts/runners/local.sh)
#   DATA_ROOT   corpus directory (default data/nq)
# Every other run.sh setting passes through unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"

if [ -z "${MODEL:-}" ]; then
    echo "ERROR: MODEL is required." >&2
    exit 2
fi
GPU="${GPU:-0}"
WEIGHT_FROM="$(PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry \
    --config "$MODEL" | sed -n "s/^WEIGHT_FROM=//p" | tr -d "'")"
read -ra SEED_LIST <<< "${SEEDS:-1 2 3 4}"
for SEED in "${SEED_LIST[@]}"; do
    if [ "$SEED" = "0" ] && [ -z "$WEIGHT_FROM" ]; then
        echo "ERROR: seed 0 is the published run; name further seeds only." >&2
        exit 2
    fi
done
STAGES="${STAGES:-4,5,6,7,8}"
OUT_DIR="${DATA_ROOT:-data/nq}/h2"
mkdir -p "$OUT_DIR"

fsr_hold "$GPU" "hold_${MODEL}_gpu${GPU}" "$OUT_DIR"

echo "H2 seed sweep for $MODEL on GPU $GPU"
echo "  seeds:  ${SEED_LIST[*]}"
echo "  stages: $STAGES"

STARTED=$(date +%s)
FAILED=()
if [ -n "$WEIGHT_FROM" ]; then
    echo
    echo "################ baseline ################"
    SEED=0 STAGES=0 GPUS="$GPU" MODEL="$MODEL" bash scripts/h2/run.sh
fi
for SEED in "${SEED_LIST[@]}"; do
    echo
    echo "################ seed $SEED ################"
    SEED="$SEED" STAGES="$STAGES" GPUS="$GPU" MODEL="$MODEL" \
        bash scripts/h2/run.sh || FAILED+=("$SEED")
done

echo
echo "seed sweep for $MODEL complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED seeds: ${FAILED[*]}" >&2
    exit 1
fi
