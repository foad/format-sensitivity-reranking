#!/usr/bin/env bash
# The iteration loop of the pipeline demonstration: one scoring worker per GPU.
#
# Each worker keeps every arm loaded, scores the job files pushed to the
# inbox, and stops after a spell with nothing to do, so no GPU is held idle.
# Launch it again to resume; the score cache in the outbox carries over.
#
# Push the inbox before the first launch. The container writes as root, so a
# directory the worker created could not be pushed to afterwards.
#
# Usage:
#   GPUS=0 bash scripts/demo/loop.sh
#
# Environment:
#   MODEL         registry slug (default bge_base)
#   ARMS          comma-separated arms (default: the study's three and the
#                 JSON hold-out pair)
#   GPUS          comma-separated GPUs, one worker each (default 0)
#   IDLE_MINUTES  minutes without a job before a worker stops (default 10)
#   WALL_HOURS    hours after which a worker stops between jobs (default 6)
#   FSR_RUNNER    runner to source (default scripts/runners/local.sh)
#   DATA_ROOT     corpus directory, where the adapters live (default data/nq)
#   DEMO_ROOT     demonstration directory (default data/demo)
#   EVAL_BATCH    pairs scored at once

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"
source "$REPO_ROOT/scripts/runners/dispatch.sh"

MODEL="${MODEL:-bge_base}"
DATA_ROOT="${DATA_ROOT:-data/nq}"
DEMO_ROOT="${DEMO_ROOT:-data/demo}"
LOOP_DIR="$DEMO_ROOT/loop"

if [ ! -d "$LOOP_DIR/inbox" ]; then
    echo "ERROR: $LOOP_DIR/inbox is missing; push it before launching." >&2
    exit 2
fi

WORKER_ARGS=(--model "$MODEL" --demo-root "$DEMO_ROOT" --data-root "$DATA_ROOT"
             --idle-minutes "${IDLE_MINUTES:-10}" --wall-hours "${WALL_HOURS:-6}")
[ -n "${ARMS:-}" ] && WORKER_ARGS+=(--arms "$ARMS")
[ -n "${EVAL_BATCH:-}" ] && WORKER_ARGS+=(--batch-size "$EVAL_BATCH")

mapfile -t WORKERS < <(fsr_gpus | sed 's/^/gpu/')

echo "Demonstration loop"
echo "  model:   $MODEL"
echo "  workers: ${WORKERS[*]}"
echo "  inbox:   $LOOP_DIR/inbox/"
echo "  outbox:  $LOOP_DIR/outbox/"

job_log() { echo "$LOOP_DIR/worker_${1}.log"; }
job_progress() { echo "$LOOP_DIR/worker_${1}.progress"; }
# A worker leaves no file that means it is finished: it serves until idle, and
# a later launch must start it again. So the dispatcher never skips one.
job_output() { echo "$LOOP_DIR/.never"; }
job_command() {
    JOB_CMD=(scripts/demo/loop_worker.py --worker "$1" "${WORKER_ARGS[@]}"
             --progress-file "$(job_progress "$1")")
}

# fsr_dispatch pairs the jobs with the GPUs in order, so worker gpuN lands on
# GPU N.
fsr_dispatch "${WORKERS[@]}"
echo
echo "every worker has stopped"
