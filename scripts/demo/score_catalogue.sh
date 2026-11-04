#!/usr/bin/env bash
# Score the grid's catalogue: every wording against every passage version, one
# arm per job, spread across GPUs.
#
# Usage:
#   GPUS=0,1,2 bash scripts/demo/score_catalogue.sh
#
# Environment:
#   MODEL       registry slug (default bge_base)
#   ARMS        comma-separated arms (default base,5fmt_lam0,5fmt_lam1,json_lam0,json_lam1)
#   GPUS        comma-separated GPUs to spread the arms across (default 0)
#   FSR_RUNNER  runner to source (default scripts/runners/local.sh)
#   DATA_ROOT   corpus directory, where the adapters live (default data/nq)
#   DEMO_ROOT   demonstration directory (default data/demo)
#   EVAL_BATCH  pairs scored at once
#   FORCE       1 to rescore arms that are present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

MODEL="${MODEL:-bge_base}"
DATA_ROOT="${DATA_ROOT:-data/nq}"
DEMO_ROOT="${DEMO_ROOT:-data/demo}"
GRID_DIR="$DEMO_ROOT/grid"
IFS=',' read -ra ARM_LIST <<< "${ARMS:-base,5fmt_lam0,5fmt_lam1,json_lam0,json_lam1}"

if [ ! -f "$GRID_DIR/catalogue.json" ]; then
    echo "ERROR: $GRID_DIR/catalogue.json is missing; push it first." >&2
    exit 2
fi

SCORE_ARGS=(--model "$MODEL" --demo-root "$DEMO_ROOT" --data-root "$DATA_ROOT")
[ -n "${EVAL_BATCH:-}" ] && SCORE_ARGS+=(--batch-size "$EVAL_BATCH")
[ "${FORCE:-0}" = "1" ] && SCORE_ARGS+=(--force)

echo "Grid catalogue scoring"
echo "  model: $MODEL"
echo "  arms:  ${ARM_LIST[*]}"
echo "  out:   $GRID_DIR/"

job_log() { echo "$GRID_DIR/score_${1}.log"; }
job_progress() { echo "$GRID_DIR/score_${1}.progress"; }
job_output() { echo "$GRID_DIR/scores_${1}.npy"; }
job_command() {
    JOB_CMD=(scripts/demo/score_catalogue.py --arms "$1" "${SCORE_ARGS[@]}"
             --progress-file "$(job_progress "$1")")
}
fsr_dispatch "${ARM_LIST[@]}"
echo
echo "grid catalogue scored"
