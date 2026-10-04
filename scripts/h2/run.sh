#!/usr/bin/env bash
# The full H2 pass for one model.
#
# Phase 1 sweeps over all lambda values and picks a winner.
# Phase 2 trains five folds at lambda=0 and the winner from Phase 1.
#
# Usage:
#   MODEL=bge_base GPUS=0,1,2,3,4 bash scripts/h2/run.sh
#
# Environment:
#   MODEL        registry slug, required
#   GPUS         comma-separated GPUs to spread jobs across (default 0)
#   STAGES       comma-separated stage numbers (default 0,1,2,3,4,5,6,7,8)
#   LAMBDAS      space-separated invariance weights (default the published set)
#   FOLDS        space-separated held-out formats (default all five)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   MAX_STEPS    optimiser steps per arm (default the published budget)
#   LIMIT        cap on records, for a smoke pass
#   FORCE        1 to redo every stage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

roster() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --list; }

if [ -z "${MODEL:-}" ]; then
    echo "ERROR: MODEL is required. Pick one of:" >&2
    roster >&2
    exit 2
fi
if ! roster | grep -qx "$MODEL"; then
    echo "ERROR: unknown MODEL '$MODEL'. Pick one of:" >&2
    roster >&2
    exit 2
fi

DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h2"
TRAIN_DIR="$OUT_DIR/train"
STAGES="${STAGES:-0,1,2,3,4,5,6,7,8}"
read -ra LAMBDA_LIST <<< "${LAMBDAS:-0 0.01 0.1 1 10}"
read -ra FOLD_LIST <<< "${FOLDS:-yaml json toml inline_kv markdown}"
mkdir -p "$OUT_DIR"

COMMON=(--data-root "$DATA_ROOT" --model "$MODEL")
TRAIN_ARGS=()
[ -n "${MAX_STEPS:-}" ] && TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
LIMIT_ARGS=()
TRAIN_LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARGS+=(--limit "$LIMIT")
    TRAIN_LIMIT_ARGS+=(--limit-train "$LIMIT" --limit-dev "$LIMIT")
fi
[ "${FORCE:-0}" = "1" ] && COMMON+=(--force)

STARTED=$(date +%s)

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

log_for() { echo "$OUT_DIR/$1.log"; }
progress_for() { echo "$OUT_DIR/$1.progress"; }
job_log() { log_for "$(job_name "$1")"; }
job_progress() { progress_for "$(job_name "$1")"; }

arm_of() {
    local FOLD="$1" WEIGHT="$2"
    [ "$FOLD" = "none" ] && FOLD="5fmt"
    echo "${FOLD}_lam${WEIGHT}"
}

baseline_eval() {
    job_name() { echo "${1}_cross_${MODEL}_base"; }
    job_command() {
        JOB_CMD=(scripts/h2/eval.py "${COMMON[@]}" --baseline --split "$1"
                 "${LIMIT_ARGS[@]}" --progress-file "$(job_progress "$1")")
    }
    job_output() { echo "$OUT_DIR/${1}_cross_${MODEL}_base.json"; }
    fsr_dispatch dev test
}

train_phase1() {
    job_name() { echo "train_${MODEL}_$(arm_of none "$1")"; }
    job_command() {
        JOB_CMD=(scripts/h2/train.py "${COMMON[@]}" --lambda-inv "$1"
                 "${TRAIN_ARGS[@]}" "${TRAIN_LIMIT_ARGS[@]}"
                 --progress-file "$(job_progress "$1")")
    }
    job_output() { echo "$TRAIN_DIR/${MODEL}_$(arm_of none "$1")/adapter/adapter_config.json"; }
    fsr_dispatch "${LAMBDA_LIST[@]}"
}

eval_phase1() {
    job_name() { echo "dev_cross_${MODEL}_$(arm_of none "$1")"; }
    job_command() {
        JOB_CMD=(scripts/h2/eval.py "${COMMON[@]}" --split dev --lambda-inv "$1"
                 "${LIMIT_ARGS[@]}" --progress-file "$(job_progress "$1")")
    }
    job_output() { echo "$OUT_DIR/dev_cross_${MODEL}_$(arm_of none "$1").json"; }
    fsr_dispatch "${LAMBDA_LIST[@]}"
}

select_weight() {
    local OUT="$OUT_DIR/selection/${MODEL}_lambda.json"
    if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "  [skip] selection: $OUT is already present"
        return 0
    fi
    GPU="$(fsr_gpus | head -1)" fsr_run "$(log_for "select_${MODEL}_lambda")" \
        scripts/h2/select.py "${COMMON[@]}" --sweep lambda
}

winner_weight() {
    local OUT="$OUT_DIR/selection/${MODEL}_lambda.json"
    [ -f "$OUT" ] || { echo "ERROR: $OUT missing; run stages 0 to 3." >&2; exit 2; }
    python3 -c "
import json, sys
chosen = json.load(open(sys.argv[1]))
if chosen.get('status') != 'selected':
    sys.exit('no weight was chosen: ' + chosen.get('reason', 'unknown'))
print(chosen['winner_lambda'])
" "$OUT"
}

train_folds() {
    local WEIGHT="$1"
    job_name() { echo "train_${MODEL}_$(arm_of "$1" "$WEIGHT")"; }
    job_command() {
        JOB_CMD=(scripts/h2/train.py "${COMMON[@]}" --held-out-format "$1"
                 --lambda-inv "$WEIGHT" "${TRAIN_ARGS[@]}" "${TRAIN_LIMIT_ARGS[@]}"
                 --progress-file "$(job_progress "$1")")
    }
    job_output() {
        echo "$TRAIN_DIR/${MODEL}_$(arm_of "$1" "$WEIGHT")/adapter/adapter_config.json"
    }
    fsr_dispatch "${FOLD_LIST[@]}"
}

fold_arms() {
    local WEIGHT="$1" FOLD
    for FOLD in "${FOLD_LIST[@]}"; do
        echo "$FOLD 0"
        echo "$FOLD $WEIGHT"
    done
}

eval_folds() {
    job_name() { set -- $1; echo "test_cross_${MODEL}_$(arm_of "$1" "$2")"; }
    job_command() {
        set -- $1
        JOB_CMD=(scripts/h2/eval.py "${COMMON[@]}" --split test
                 --held-out-format "$1" --lambda-inv "$2" "${LIMIT_ARGS[@]}"
                 --progress-file "$(progress_for "test_cross_${MODEL}_$(arm_of "$1" "$2")")")
    }
    job_output() { set -- $1; echo "$OUT_DIR/test_cross_${MODEL}_$(arm_of "$1" "$2").json"; }
    mapfile -t ARMS < <(fold_arms "$1")
    fsr_dispatch "${ARMS[@]}"
}

within_folds() {
    job_name() { set -- $1; echo "test_within_${MODEL}_$(arm_of "$1" "$2")"; }
    job_command() {
        set -- $1
        local ARM OUT
        ARM="$(arm_of "$1" "$2")"
        OUT="$OUT_DIR/test_within_${MODEL}_${ARM}.json"
        JOB_CMD=(scripts/within_query.py --data-root "$DATA_ROOT" --split test
                 --mode with_body --models "$MODEL"
                 --lora-adapter "$TRAIN_DIR/${MODEL}_${ARM}/adapter"
                 --out-path "$OUT" "${LIMIT_ARGS[@]}"
                 --progress-file "$(progress_for "test_within_${MODEL}_${ARM}")")
    }
    job_output() { set -- $1; echo "$OUT_DIR/test_within_${MODEL}_$(arm_of "$1" "$2").json"; }
    mapfile -t ARMS < <(fold_arms "$1")
    fsr_dispatch "${ARMS[@]}"
}

compare_folds() {
    local WEIGHT="$1" FOLD W OUT
    for FOLD in "${FOLD_LIST[@]}"; do
        for W in 0 "$WEIGHT"; do
            OUT="$OUT_DIR/comparison/${MODEL}_$(arm_of "$FOLD" "$W").json"
            if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
                echo "  [skip] compare $(arm_of "$FOLD" "$W"): already present"
                continue
            fi
            GPU="$(fsr_gpus | head -1)" fsr_run \
                "$(log_for "compare_${MODEL}_$(arm_of "$FOLD" "$W")")" \
                scripts/h2/compare.py "${COMMON[@]}" --split test \
                --held-out-format "$FOLD" --lambda-inv "$W"
        done
    done
}

prose_rank() {
    job_name() { echo "prose_mrr_${MODEL}_$1"; }
    job_command() {
        local SELECT=(--arm "$1")
        [ "$1" = "base" ] && SELECT=(--baseline)
        JOB_CMD=(scripts/h2/eval_prose.py "${COMMON[@]}" "${SELECT[@]}"
                 "${LIMIT_ARGS[@]}" --progress-file "$(job_progress "$1")")
    }
    job_output() { echo "$OUT_DIR/prose_mrr_${MODEL}_${1}.json"; }
    local WEIGHT="$1" ARMS=(base)
    local FOLD W
    for FOLD in "${FOLD_LIST[@]}"; do
        for W in 0 "$WEIGHT"; do
            ARMS+=("$(arm_of "$FOLD" "$W")")
        done
    done
    fsr_dispatch "${ARMS[@]}"
}

compare_prose() {
    local WEIGHT="$1" FOLD W ARM OUT
    for FOLD in "${FOLD_LIST[@]}"; do
        for W in 0 "$WEIGHT"; do
            ARM="$(arm_of "$FOLD" "$W")"
            OUT="$OUT_DIR/comparison/${MODEL}_prose_${ARM}.json"
            if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
                echo "  [skip] prose compare ${ARM}: already present"
                continue
            fi
            GPU="$(fsr_gpus | head -1)" fsr_run \
                "$(log_for "compare_prose_${MODEL}_${ARM}")" \
                scripts/h2/compare_prose.py "${COMMON[@]}" --arm "$ARM"
        done
    done
}

echo "H2 pass for $MODEL"
echo "  weights: ${LAMBDA_LIST[*]}"
echo "  folds:   ${FOLD_LIST[*]}"
echo "  stages:  $STAGES"
echo "  results: $OUT_DIR/"

stage 0 "baseline eval, dev and test" baseline_eval
stage 1 "Phase 1: train one arm per weight" train_phase1
stage 2 "Phase 1: dev eval" eval_phase1
stage 3 "Phase 1: choose the weight" select_weight

WINNER=""
if wanted 4 || wanted 5 || wanted 6 || wanted 7 || wanted 8 \
    || wanted 9 || wanted 10; then
    WINNER="$(winner_weight)"
    echo
    echo "chosen weight: $WINNER"
fi

stage 4 "Phase 2: folds at weight 0" train_folds 0
stage 5 "Phase 2: folds at weight $WINNER" train_folds "$WINNER"
stage 6 "Phase 2: test eval, score axis" eval_folds "$WINNER"
stage 7 "Phase 2: test eval, answer axis" within_folds "$WINNER"
stage 8 "Phase 2: compare each arm against the baseline" compare_folds "$WINNER"
stage 9 "prose ranking" prose_rank "$WINNER"
stage 10 "prose comparison" compare_prose "$WINNER"

echo
echo "H2 pass for $MODEL complete in $(( ($(date +%s) - STARTED) / 60 )) min"
echo "  results: $OUT_DIR/"
