#!/usr/bin/env bash
# Launch a measurement locally.
#
# Usage:
#   fsr_run <log> <script> [args...]

fsr_run() {
    local LOG="$1"
    shift
    local DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
    echo "[runner] local, CUDA device ${DEVICE}, log ${LOG}"
    CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 \
        uv run python "$@" >"$LOG" 2>&1
}
