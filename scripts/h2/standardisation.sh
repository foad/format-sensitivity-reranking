#!/usr/bin/env bash
# Score every candidate under every format, for the standardisation study.
#
# The study compares normalising a mixed-format corpus to one serialisation
# against training the model to tolerate the mix. Each capture writes one
# score tensor; every measurement after that is offline.
#
# Usage:
#   GPUS=0,1 bash scripts/h2/standardisation.sh
#
# Environment:
#   MODELS       comma-separated registry slugs (default the six base models)
#   ARMS         comma-separated arms (default base,5fmt_lam0,5fmt_lam<winner>)
#   SPLIT        split to score (default test)
#   GPUS         comma-separated GPUs to spread jobs across (default 0)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   EVAL_BATCH   pairs scored at once
#   LIMIT        cap on records, for a smoke pass
#   FORCE        1 to rebuild a capture that is present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

roster() { PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --list; }

# The roster without the tanh variant, which H2-D53 ruled out as an
# intervention study. Name it in MODELS to score it anyway.
base_roster() {
    PYTHONPATH="$REPO_ROOT/src" python3 -c "
from fsr.models.registry import BASE_MODELS
for model in BASE_MODELS:
    print(model.slug)
"
}

DATA_ROOT="${DATA_ROOT:-data/nq}"
OUT_DIR="$DATA_ROOT/h2/standardisation"
SPLIT="${SPLIT:-test}"
mkdir -p "$OUT_DIR"

if [ -n "${MODELS:-}" ]; then
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
else
    mapfile -t MODEL_LIST < <(base_roster)
fi

for NAME in "${MODEL_LIST[@]}"; do
    if ! roster | grep -qx "$NAME"; then
        echo "ERROR: unknown model '$NAME'. Pick from:" >&2
        roster >&2
        exit 2
    fi
done

ARGS=(--data-root "$DATA_ROOT" --split "$SPLIT")
[ -n "${EVAL_BATCH:-}" ] && ARGS+=(--batch-size "$EVAL_BATCH")
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)

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

# The untrained model, the rank-only control, and the chosen weight. The
# control and the chosen weight coincide when Phase 1 chose zero.
arms_for() {
    local NAME="$1" WINNER
    if [ -n "${ARMS:-}" ]; then
        printf '%s\n' "${ARMS//,/ }"
        return
    fi
    WINNER="$(winner_weight "$NAME")"
    local LIST=(base 5fmt_lam0)
    [ "$WINNER" != "0" ] && LIST+=("5fmt_lam${WINNER}")
    printf '%s\n' "${LIST[@]}"
}

tag() { set -- $1; echo "capture_${1}_${2}"; }

capture() {
    job_name() { tag "$1"; }
    job_log() { echo "$OUT_DIR/$(tag "$1").log"; }
    job_progress() { echo "$OUT_DIR/$(tag "$1").progress"; }
    job_output() {
        set -- $1
        echo "$OUT_DIR/${SPLIT}_scores_${1}_${2}.npy"
    }
    job_command() {
        set -- $1
        JOB_CMD=(scripts/h2/capture_standardisation.py --model "$1" --arm "$2"
                 "${ARGS[@]}" --progress-file "$(job_progress "$1 $2")")
    }
    # Model-major, so a wave holds the arms of one model. Every job scores the
    # same pair count, so its cost follows the model and not the arm.
    local KEYS=() NAME ARM
    for NAME in "${MODEL_LIST[@]}"; do
        while read -r ARM; do
            KEYS+=("$NAME $ARM")
        done < <(arms_for "$NAME")
    done
    fsr_dispatch "${KEYS[@]}"
}

echo "Standardisation study"
echo "  models:  ${MODEL_LIST[*]}"
echo "  split:   $SPLIT"
echo "  results: $OUT_DIR/"

STARTED=$(date +%s)
capture
echo
echo "standardisation capture complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
echo "  results: $OUT_DIR/"
