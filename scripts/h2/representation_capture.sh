#!/usr/bin/env bash
# Capture the final-layer representation.
#
# Usage:
#   MODELS=minilm_l6,minilm_l12 SPLITS=dev GPUS=0 bash scripts/h2/representation_capture.sh
#   MODELS=minilm_l6 SPLITS=dev WEIGHTS="0 winner" SEEDS="0 1 2 3 4" GPUS=0 \
#       bash scripts/h2/representation_capture.sh
#
# Environment:
#   MODELS       comma-separated registry slugs (default every base model)
#   SPLITS       comma-separated splits (default dev)
#   GPUS         comma-separated GPUs to spread jobs across (default 0)
#   HOLD_GB      memory to hold on Hex on a single GPU until the captures end, in GiB,
#                or 0 for no hold (default 3); no hold across several GPUs
#   HOLD_HOURS   release the hold after this long whatever happens (default 168)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   NEGATIVES    negatives cached per record (default the script default)
#   EVAL_BATCH   pairs scored at once by the capture
#   LIMIT        cap on records, for a smoke pass
#   FORCE        1 to rebuild an artefact that is present
#   WEIGHTS      space-separated invariance weights
#   SEEDS        space-separated training seeds of those arms (default 0)
#   FEATURE_DIR  where trained-arm captures go (default $FSR_SCRATCH/head_probe)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

roster() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --list; }
base_models() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --base | paste -sd,; }

DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h2/head_probe"
IFS=',' read -ra MODEL_LIST <<< "${MODELS:-$(base_models)}"
IFS=',' read -ra SPLIT_LIST <<< "${SPLITS:-dev}"
read -ra WEIGHT_LIST <<< "${WEIGHTS:-}"
read -ra SEED_LIST <<< "${SEEDS:-0}"
if [ "${#WEIGHT_LIST[@]}" -gt 0 ]; then
    FEATURE_DIR="${FEATURE_DIR:-${FSR_SCRATCH:+$FSR_SCRATCH/head_probe}}"
    if [ -z "$FEATURE_DIR" ]; then
        echo "ERROR: trained captures need FEATURE_DIR or FSR_SCRATCH." >&2
        exit 2
    fi
    mkdir -p "$FEATURE_DIR"
fi
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

tag() { set -- $1; echo "capture_${1}_${2}${3:+_$3}"; }

trained_arms() {
    PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.representation.arms \
        --data-root "$DATA_ROOT" --model "$1" \
        --weights "${WEIGHT_LIST[@]}" --seeds "${SEED_LIST[@]}"
}

capture_features() {
    job_name() { tag "$1"; }
    job_log() { echo "$OUT_DIR/$(tag "$1").log"; }
    job_progress() { echo "$OUT_DIR/$(tag "$1").progress"; }
    job_output() {
        set -- $1
        if [ -n "${3:-}" ]; then
            echo "$FEATURE_DIR/${2}_features_${1}_${3}.json"
        else
            echo "$OUT_DIR/${2}_features_${1}.json"
        fi
    }
    job_command() {
        set -- $1
        JOB_CMD=(scripts/h2/capture_features.py --model "$1" --split "$2"
                 "${ARGS[@]}" --progress-file "$(job_progress "$1 $2 ${3:-}")")
        [ -n "${3:-}" ] && JOB_CMD+=(--arm "$3" --feature-dir "$FEATURE_DIR")
        return 0
    }
    # Split-major, so a wave holds jobs of similar length. fsr_dispatch waits
    # for every job in a wave, and a train job runs far longer than a dev one.
    local KEYS=() NAME SPLIT ARM ARMS
    for SPLIT in "${SPLIT_LIST[@]}"; do
        for NAME in "${MODEL_LIST[@]}"; do
            if [ "${#WEIGHT_LIST[@]}" -eq 0 ]; then
                KEYS+=("$NAME $SPLIT")
                continue
            fi
            ARMS="$(trained_arms "$NAME")"
            for ARM in $ARMS; do
                KEYS+=("$NAME $SPLIT $ARM")
            done
        done
    done
    fsr_dispatch "${KEYS[@]}"
}

mapfile -t GPU_LIST < <(fsr_gpus)
if [ "${#GPU_LIST[@]}" -eq 1 ]; then
    fsr_hold "${GPU_LIST[0]}" "hold_representation_gpu${GPU_LIST[0]}" "$OUT_DIR"
fi

echo "Representation capture"
echo "  models:  ${MODEL_LIST[*]}"
echo "  splits:  ${SPLIT_LIST[*]}"
[ "${#WEIGHT_LIST[@]}" -gt 0 ] && echo "  weights: ${WEIGHT_LIST[*]}, seeds ${SEED_LIST[*]} -> $FEATURE_DIR"
echo "  results: $OUT_DIR/"

STARTED=$(date +%s)
capture_features
echo
echo "representation capture complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
echo "  results: $OUT_DIR/"
