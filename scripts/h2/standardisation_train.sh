#!/usr/bin/env bash
# Train the arms of the standardisation study, once per seed.
#
# Each single-format arm trains on one format alone at lambda=0, the
# invariance term being empty with a single format. The five-format arms at
# lambda=0 and at the chosen weight retrain the Phase 1 arms under the same
# seed, for the seed sweep. The step budget and batch geometry match the
# published arms, so format coverage is the only difference. Score the arms
# afterwards with scripts/h2/standardisation.sh.
#
# The single-format arms run first, so a fault shows within minutes rather
# than after a five-format run. An arm whose adapter is present is skipped,
# so the published seed-0 arms are never retrained.
#
# Usage:
#   MODELS=bge_base GPUS=0,1 bash scripts/h2/standardisation_train.sh
#   MODELS=bge_base SEEDS=2 GPUS=3 bash scripts/h2/standardisation_train.sh
#
# Environment:
#   MODELS       comma-separated registry slugs, required
#   SEEDS        space-separated training seeds (default 0)
#   FORMATS      space-separated single formats (default all five)
#   FIVE_FORMAT  1 to add the five-format arms (default 1)
#   GPUS         comma-separated GPUs to spread jobs across (default 0)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   MAX_STEPS    optimiser steps per arm (default the published budget)
#   CHECKPOINT   steps between resume checkpoints (default 100)
#   LIMIT        cap on records, for a smoke pass
#   FORCE        1 to retrain an arm that is present
#   LIST         1 to print the arm names, comma-separated, and stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

roster() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --list; }

if [ -z "${MODELS:-}" ]; then
    echo "ERROR: MODELS is required. Pick from:" >&2
    roster >&2
    exit 2
fi
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for NAME in "${MODEL_LIST[@]}"; do
    if ! roster | grep -qx "$NAME"; then
        echo "ERROR: unknown model '$NAME'. Pick from:" >&2
        roster >&2
        exit 2
    fi
done

DATA_ROOT="${DATA_ROOT:-data/nq}"
TRAIN_DIR="$DATA_ROOT/h2/train"
OUT_DIR="$DATA_ROOT/h2/standardisation"
read -ra SEED_LIST <<< "${SEEDS:-0}"
read -ra FORMAT_LIST <<< "${FORMATS:-yaml json toml inline_kv markdown}"

TRAIN_ARGS=(--data-root "$DATA_ROOT"
            --checkpoint-every "${CHECKPOINT:-100}" --resume)
[ -n "${MAX_STEPS:-}" ] && TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
[ -n "${LIMIT:-}" ] && TRAIN_ARGS+=(--limit-train "$LIMIT" --limit-dev "$LIMIT")
[ "${FORCE:-0}" = "1" ] && TRAIN_ARGS+=(--force)

# The weight Phase 1 chose for one model, read from its selection file.
winner_weight() {
    local OUT="$DATA_ROOT/h2/selection/${1}_lambda.json"
    [ -f "$OUT" ] || { echo "ERROR: $OUT missing; run h2 stages 0 to 3." >&2; exit 2; }
    python3 -c "
import json, sys
chosen = json.load(open(sys.argv[1]))
if chosen.get('status') != 'selected':
    sys.exit('no weight was chosen: ' + chosen.get('reason', 'unknown'))
print(chosen['winner_lambda'])
" "$OUT"
}

# The arm name, as fsr.h2_layout.arm writes it. FOLD is a format for a
# single-format arm, or 5fmt.
arm_of() {
    local FOLD="$1" WEIGHT="$2" SEED="$3" NAME
    if [ "$FOLD" = "5fmt" ]; then
        NAME="5fmt_lam${WEIGHT}"
    else
        NAME="only_${FOLD}_lam${WEIGHT}"
    fi
    [ "$SEED" != "0" ] && NAME="${NAME}_s${SEED}"
    echo "$NAME"
}

# One key per arm, as "model seed fold weight", single formats first.
KEYS=()
for NAME in "${MODEL_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
        for FORMAT in "${FORMAT_LIST[@]}"; do
            KEYS+=("$NAME $SEED $FORMAT 0")
        done
        if [ "${FIVE_FORMAT:-1}" = "1" ]; then
            KEYS+=("$NAME $SEED 5fmt 0")
            WINNER="$(winner_weight "$NAME")"
            [ "$WINNER" != "0" ] && KEYS+=("$NAME $SEED 5fmt $WINNER")
        fi
    done
done

if [ "${LIST:-0}" = "1" ]; then
    NAMES=()
    for KEY in "${KEYS[@]}"; do
        set -- $KEY
        NAMES+=("$(arm_of "$3" "$4" "$2")")
    done
    (IFS=','; echo "${NAMES[*]}")
    exit 0
fi

mkdir -p "$OUT_DIR"

train() {
    job_name() { set -- $1; echo "train_${1}_$(arm_of "$3" "$4" "$2")"; }
    job_log() { echo "$OUT_DIR/$(job_name "$1").log"; }
    job_progress() { echo "$OUT_DIR/$(job_name "$1").progress"; }
    job_output() {
        set -- $1
        echo "$TRAIN_DIR/${1}_$(arm_of "$3" "$4" "$2")/adapter/adapter_config.json"
    }
    job_command() {
        local KEY="$1"
        set -- $KEY
        JOB_CMD=(scripts/h2/train.py --model "$1" --seed "$2" --lambda-inv "$4")
        if [ "$3" != "5fmt" ]; then
            JOB_CMD+=(--only-format "$3")
        fi
        JOB_CMD+=("${TRAIN_ARGS[@]}" --progress-file "$(job_progress "$KEY")")
    }
    fsr_dispatch "${KEYS[@]}"
}

echo "Standardisation study training"
echo "  models:  ${MODEL_LIST[*]}"
echo "  seeds:   ${SEED_LIST[*]}"
echo "  formats: ${FORMAT_LIST[*]}"
echo "  5fmt:    $([ "${FIVE_FORMAT:-1}" = "1" ] && echo "lambda=0 and the chosen weight" || echo "off")"
echo "  arms:    ${#KEYS[@]}"
echo "  results: $TRAIN_DIR/"

STARTED=$(date +%s)
train
echo
echo "standardisation training complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
echo "  arms:    $TRAIN_DIR/"
