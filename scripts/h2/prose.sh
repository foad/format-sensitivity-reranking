#!/usr/bin/env bash
# Rank ordinary prose with the Phase 2 arms and compare each to the baseline.
#
# Usage:
#   SEEDS="1 2" GPUS=0 bash scripts/h2/prose.sh
#   MODELS=minilm_l6 SEEDS="3 4" GPUS=1 bash scripts/h2/prose.sh
#
# Environment:
#   MODELS       comma-separated registry slugs (default every base model)
#   SEEDS        space-separated training seeds (default 0)
#   GPUS         comma-separated GPUs to spread jobs across (default 0)
#   HOLD_GB      memory to hold on Hex on a single GPU until the run ends, in GiB,
#                or 0 for no hold (default 3); no hold across several GPUs
#   HOLD_HOURS   release the hold after this long whatever happens (default 168)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   EVAL_BATCH   pairs scored at once
#   LIMIT        cap on records, for a smoke pass
#   FORCE        1 to redo a job whose output is present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

roster() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --list; }
base_models() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --base | paste -sd,; }

DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h2"
IFS=',' read -ra MODEL_LIST <<< "${MODELS:-$(base_models)}"
read -ra SEED_LIST <<< "${SEEDS:-0}"
if [ "${#MODEL_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no models to rank." >&2
    exit 2
fi
mkdir -p "$OUT_DIR/comparison"

for NAME in "${MODEL_LIST[@]}"; do
    if ! roster | grep -qx "$NAME"; then
        echo "ERROR: unknown model '$NAME'. Pick from:" >&2
        roster >&2
        exit 2
    fi
    if [ ! -f "$OUT_DIR/prose_mrr_${NAME}_base.json" ]; then
        echo "ERROR: no prose baseline for $NAME; run h2 stage 9 at seed 0." >&2
        exit 2
    fi
done

ARGS=(--data-root "$DATA_ROOT")
EVAL_ARGS=()
[ -n "${EVAL_BATCH:-}" ] && EVAL_ARGS+=(--batch-size "$EVAL_BATCH")
[ -n "${LIMIT:-}" ] && EVAL_ARGS+=(--limit "$LIMIT")
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)

trained_arms() {
    PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.representation.arms \
        --data-root "$DATA_ROOT" --model "$1" --weights 0 winner --seeds "$2"
}

rank_prose() {
    job_log() { set -- $1; echo "$OUT_DIR/prose_mrr_${1}_${2}.log"; }
    job_progress() { set -- $1; echo "$OUT_DIR/prose_mrr_${1}_${2}.progress"; }
    job_output() { set -- $1; echo "$OUT_DIR/prose_mrr_${1}_${2}.json"; }
    job_command() {
        set -- $1
        JOB_CMD=(scripts/h2/eval_prose.py --model "$1" --arm "$2"
                 "${ARGS[@]}" "${EVAL_ARGS[@]}"
                 --progress-file "$(job_progress "$1 $2")")
    }
    fsr_dispatch "$@"
}

compare_prose() {
    job_log() { set -- $1; echo "$OUT_DIR/compare_prose_${1}_${2}.log"; }
    job_output() { set -- $1; echo "$OUT_DIR/comparison/${1}_prose_${2}.json"; }
    job_command() {
        set -- $1
        JOB_CMD=(scripts/h2/compare_prose.py --model "$1" --arm "$2" "${ARGS[@]}")
    }
    fsr_dispatch "$@"
}

mapfile -t GPU_LIST < <(fsr_gpus)
if [ "${#GPU_LIST[@]}" -eq 1 ]; then
    fsr_hold "${GPU_LIST[0]}" \
        "hold_prose_s$(IFS=-; echo "${SEED_LIST[*]}")_gpu${GPU_LIST[0]}" "$OUT_DIR"
fi

echo "Prose ranking"
echo "  models: ${MODEL_LIST[*]}"
echo "  seeds:  ${SEED_LIST[*]}"
echo "  results: $OUT_DIR/"

STARTED=$(date +%s)
FAILED=()
for SEED in "${SEED_LIST[@]}"; do
    for NAME in "${MODEL_LIST[@]}"; do
        echo
        echo "== $NAME, seed $SEED"
        KEYS=()
        for ARM in $(trained_arms "$NAME" "$SEED"); do
            KEYS+=("$NAME $ARM")
        done
        if rank_prose "${KEYS[@]}"; then
            compare_prose "${KEYS[@]}" || FAILED+=("$NAME s$SEED compare")
        else
            FAILED+=("$NAME s$SEED rank")
        fi
    done
done

echo
echo "prose ranking complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}" >&2
    exit 1
fi
