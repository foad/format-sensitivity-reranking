#!/usr/bin/env bash
# Cache the representations the head-capacity probe fits to.
#
# Usage:
#   MODELS=mxbai_v1,jina_v2 GPUS=0,1 bash scripts/h2/head_probe.sh
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
OUT_DIR="$DATA_ROOT/h2/head_probe"
IFS=',' read -ra MODEL_LIST <<< "${MODELS:-mxbai_v1,jina_v2}"
IFS=',' read -ra SPLIT_LIST <<< "${SPLITS:-train,dev}"
STAGES="${STAGES:-0,1}"
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

FIT_ARGS=()
[ -n "${HEADS:-}" ] && FIT_ARGS+=(--heads ${HEADS})
[ -n "${LAMBDAS:-}" ] && FIT_ARGS+=(--lambdas ${LAMBDAS})
[ -n "${SEEDS:-}" ] && FIT_ARGS+=(--seeds ${SEEDS})
[ -n "${MAX_STEPS:-}" ] && FIT_ARGS+=(--max-steps "$MAX_STEPS")
[ "${FORCE:-0}" = "1" ] && FIT_ARGS+=(--force)

wanted() { case ",${STAGES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

stage() {
    local NUMBER="$1" TITLE="$2"
    shift 2
    wanted "$NUMBER" || return 0
    echo
    echo "============================================================"
    echo "=== stage $NUMBER: $TITLE"
    echo "============================================================"
    "$@"
}

tag() { set -- $1; echo "capture_${1}_${2}"; }

job_name() { tag "$1"; }
job_log() { echo "$OUT_DIR/$(tag "$1").log"; }
job_progress() { echo "$OUT_DIR/$(tag "$1").progress"; }
job_output() {
    set -- $1
    echo "$DATA_ROOT/h2/head_probe/${2}_features_${1}.npy"
}
job_command() {
    set -- $1
    JOB_CMD=(scripts/h2/capture_features.py --model "$1" --split "$2"
             "${ARGS[@]}" --progress-file "$(job_progress "$1 $2")")
}

capture_features() {
    job_name() { tag "$1"; }
    job_log() { echo "$OUT_DIR/$(tag "$1").log"; }
    job_progress() { echo "$OUT_DIR/$(tag "$1").progress"; }
    job_output() {
        set -- $1
        echo "$OUT_DIR/${2}_features_${1}.npy"
    }
    job_command() {
        set -- $1
        JOB_CMD=(scripts/h2/capture_features.py --model "$1" --split "$2"
                 "${ARGS[@]}" --progress-file "$(job_progress "$1 $2")")
    }
    # Split-major, so a wave holds jobs of similar length. fsr_dispatch waits
    # for every job in a wave, and a train job runs far longer than a dev one.
    local KEYS=() NAME SPLIT
    for SPLIT in "${SPLIT_LIST[@]}"; do
        for NAME in "${MODEL_LIST[@]}"; do
            KEYS+=("$NAME $SPLIT")
        done
    done
    fsr_dispatch "${KEYS[@]}"
}

fit_heads() {
    job_name() { echo "fit_heads_$1"; }
    job_log() { echo "$OUT_DIR/$(job_name "$1").log"; }
    job_progress() { echo "$OUT_DIR/$(job_name "$1").progress"; }
    job_output() { echo "$OUT_DIR/frontier_${1}.json"; }
    job_command() {
        JOB_CMD=(scripts/h2/fit_heads.py --model "$1" --data-root "$DATA_ROOT"
                 "${FIT_ARGS[@]}" --progress-file "$(job_progress "$1")")
    }
    fsr_dispatch "${MODEL_LIST[@]}"
}

echo "Head probe"
echo "  models:  ${MODEL_LIST[*]}"
echo "  splits:  ${SPLIT_LIST[*]}"
echo "  stages:  $STAGES"
echo "  results: $OUT_DIR/"

STARTED=$(date +%s)
stage 0 "cache the representations" capture_features
stage 1 "fit every head of each model" fit_heads
echo
echo "head probe complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
echo "  results: $OUT_DIR/"
