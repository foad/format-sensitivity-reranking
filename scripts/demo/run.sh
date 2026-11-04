#!/usr/bin/env bash
# The pipeline demonstration: draw the pools, score them, pick the case.
#
# The pools are synthetic. Each arm scores every pool under every format, and
# the selection reads the tensors offline. The JSON hold-out arms are scored
# beside the deployed ones, as a check that the case does not rest on the
# invariance arm having seen the format in training.
#
# Usage:
#   GPUS=0,1 bash scripts/demo/run.sh
#
# Environment:
#   MODEL        registry slug (default bge_base)
#   ARMS         comma-separated arms (default base,5fmt_lam0,5fmt_lam<winner>,
#                json_lam0,json_lam<winner>)
#   N_SCENARIOS  pools to draw (default 2000)
#   SEED         seed of the draw (default 0)
#   K            candidates the reranker passes on (default 3)
#   GPUS         comma-separated GPUs to spread jobs across (default 0)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory, where the adapters live (default data/nq)
#   DEMO_ROOT    demonstration directory (default data/demo)
#   EVAL_BATCH   pairs scored at once
#   FORCE        1 to rebuild outputs that are present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

MODEL="${MODEL:-bge_base}"
DATA_ROOT="${DATA_ROOT:-data/nq}"
DEMO_ROOT="${DEMO_ROOT:-data/demo}"
N_SCENARIOS="${N_SCENARIOS:-2000}"
SEED="${SEED:-0}"
K="${K:-3}"
mkdir -p "$DEMO_ROOT"

FORCE_ARG=()
[ "${FORCE:-0}" = "1" ] && FORCE_ARG=(--force)

# The weight Phase 1 chose for the model, read from its selection file.
winner_weight() {
    local OUT="$DATA_ROOT/h2/selection/${MODEL}_lambda.json"
    [ -f "$OUT" ] || { echo "ERROR: $OUT missing; run h2 stages 0 to 3." >&2; exit 2; }
    python3 -c "
import json, sys
chosen = json.load(open(sys.argv[1]))
if chosen.get('status') != 'selected':
    sys.exit('no weight was chosen: ' + chosen.get('reason', 'unknown'))
print(chosen['winner_lambda'])
" "$OUT"
}

if [ -n "${ARMS:-}" ]; then
    IFS=',' read -ra ARM_LIST <<< "$ARMS"
else
    WINNER="$(winner_weight)"
    if [ "$WINNER" = "0" ]; then
        echo "ERROR: the sweep chose a weight of zero for $MODEL; no invariance arm." >&2
        exit 2
    fi
    ARM_LIST=(base 5fmt_lam0 "5fmt_lam${WINNER}" json_lam0 "json_lam${WINNER}")
fi

echo "Pipeline demonstration"
echo "  model:   $MODEL"
echo "  arms:    ${ARM_LIST[*]}"
echo "  pools:   $N_SCENARIOS (seed $SEED), top $K kept"
echo "  results: $DEMO_ROOT/"

STARTED=$(date +%s)

# The drawing and the selection need no GPU, but go through the runner like
# every other stage, so they run where the scoring does.
FIRST_GPU="$(fsr_gpus | head -1)"

echo
echo "[1/3] drawing the pools"
GPU="$FIRST_GPU" fsr_run "$DEMO_ROOT/build_scenarios.log" \
    scripts/demo/build_scenarios.py --demo-root "$DEMO_ROOT" \
    --n "$N_SCENARIOS" --seed "$SEED" "${FORCE_ARG[@]}"

echo
echo "[2/3] scoring each arm"
SCORE_ARGS=(--model "$MODEL" --demo-root "$DEMO_ROOT" --data-root "$DATA_ROOT")
[ -n "${EVAL_BATCH:-}" ] && SCORE_ARGS+=(--batch-size "$EVAL_BATCH")
[ "${FORCE:-0}" = "1" ] && SCORE_ARGS+=(--force)

job_log() { echo "$DEMO_ROOT/score_${MODEL}_${1}.log"; }
job_progress() { echo "$DEMO_ROOT/score_${MODEL}_${1}.progress"; }
job_output() { echo "$DEMO_ROOT/scores_${MODEL}_${1}.npy"; }
job_command() {
    JOB_CMD=(scripts/demo/score.py --arm "$1" "${SCORE_ARGS[@]}"
             --progress-file "$(job_progress "$1")")
}
fsr_dispatch "${ARM_LIST[@]}"

echo
echo "[3/3] selecting the case"
GPU="$FIRST_GPU" fsr_run "$DEMO_ROOT/select_${MODEL}.log" \
    scripts/demo/select.py --model "$MODEL" --demo-root "$DEMO_ROOT" \
    --data-root "$DATA_ROOT" --k "$K"

echo
echo "pipeline demonstration complete in $(( ( $(date +%s) - STARTED ) / 60 )) min"
echo "  results: $DEMO_ROOT/"
