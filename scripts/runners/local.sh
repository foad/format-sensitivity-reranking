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
    if [ -t 1 ]; then
        CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 \
            uv run python "$@" 2>&1 | tee "$LOG"
        return "${PIPESTATUS[0]}"
    fi
    CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 \
        uv run python "$@" >"$LOG" 2>&1
}
