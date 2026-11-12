#!/usr/bin/env bash
# Run Phase 2 of H2 under further training seeds, one model on one GPU.
#
# Each seed retrains the five held-out folds at weight 0 and at the weight
# Phase 1 chose, then scores and compares them (stages 4 to 8 of run.sh).
# Seeds run one after another. A block of GPU memory is held for the whole
# sweep, so the GPU stays visibly claimed between jobs and during light ones;
# it is released when the sweep ends, however it ends.
#
# Usage:
#   MODEL=jina_v2 GPU=2 bash scripts/h2/seed_sweep.sh
#
# Environment:
#   MODEL       registry slug, required
#   GPU         the one GPU the sweep runs on (default 0)
#   SEEDS       space-separated training seeds (default 1 2 3 4)
#   STAGES      run.sh stages per seed (default 4,5,6,7,8)
#   HOLD_GB     memory to hold, in GiB, or 0 for no hold (default 3)
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
read -ra SEED_LIST <<< "${SEEDS:-1 2 3 4}"
for SEED in "${SEED_LIST[@]}"; do
    if [ "$SEED" = "0" ]; then
        echo "ERROR: seed 0 is the published run; name further seeds only." >&2
        exit 2
    fi
done
STAGES="${STAGES:-4,5,6,7,8}"
HOLD_GB="${HOLD_GB:-3}"
OUT_DIR="${DATA_ROOT:-data/nq}/h2"
mkdir -p "$OUT_DIR"

HOLD_NAME="hold_${MODEL}_gpu${GPU}"
HOLD_FLAG="$OUT_DIR/${HOLD_NAME}.flag"
HOLD_PID=""

release() {
    rm -f "$HOLD_FLAG"
    if [ -n "$HOLD_PID" ]; then
        wait "$HOLD_PID" 2>/dev/null || true
        HOLD_PID=""
        echo "released the hold on GPU $GPU"
    fi
}
trap release EXIT
trap 'exit 130' INT TERM

if [ "$HOLD_GB" != "0" ]; then
    touch "$HOLD_FLAG"
    GPU="$GPU" fsr_run "$OUT_DIR/${HOLD_NAME}.log" scripts/hold_gpu.py \
        --gb "$HOLD_GB" --minutes "$(( ${HOLD_HOURS:-168} * 60 ))" \
        --until-gone "$HOLD_FLAG" > /dev/null &
    HOLD_PID=$!
    echo "holding ${HOLD_GB} GiB on GPU $GPU until the sweep ends"
fi

echo "H2 seed sweep for $MODEL on GPU $GPU"
echo "  seeds:  ${SEED_LIST[*]}"
echo "  stages: $STAGES"

STARTED=$(date +%s)
FAILED=()
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
