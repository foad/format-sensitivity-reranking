#!/usr/bin/env bash
# Hold a block of GPU memory for a fixed time, to test how Hex reports a GPU
# that is claimed but idle.
#
# Usage:
#   GPU=3 HOLD_GB=3 HOLD_MINUTES=12 bash scripts/hold_gpu.sh
#
# Environment:
#   GPU           the device to hold (default 0)
#   HOLD_GB       memory to hold, in GiB (default 3)
#   HOLD_MINUTES  how long to hold it (default 12)
#   FSR_RUNNER    runner to source (default scripts/runners/local.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "${FSR_RUNNER:-$REPO_ROOT/scripts/runners/local.sh}"

GPU="${GPU:-0}"
LOG_DIR="data/hold_gpu"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/hold_gpu_${GPU}.log"

echo "holding ${HOLD_GB:-3} GiB on GPU $GPU for ${HOLD_MINUTES:-12} min"
echo "  log: $LOG"
GPU="$GPU" fsr_run "$LOG" scripts/hold_gpu.py \
    --gb "${HOLD_GB:-3}" --minutes "${HOLD_MINUTES:-12}"
