#!/usr/bin/env bash
# Spread jobs across GPUs in waves.
#
#   job_command <key>   sets JOB_CMD to the script and arguments for one job
#   job_output  <key>   echoes the file that means the job is done
#   job_log     <key>   echoes the file its detailed output goes to
#
# Environment:
#   GPUS   comma or space separated GPUs (default 0)
#   GPU    a single GPU, an alias for GPUS
#   FORCE  1 to re-run a job whose output is present

fsr_gpus() {
    local LIST
    IFS=', ' read -ra LIST <<< "${GPUS:-${GPU:-0}}"
    printf '%s\n' "${LIST[@]}"
}

# The model slugs a measurement script reports, for the passes keyed by model.
fsr_roster() {
    local SCRIPT="$1" LOG="$2"
    GPU="$(fsr_gpus | head -1)" fsr_run "$LOG" "$SCRIPT" --list-models > /dev/null
    grep -E '^[a-z0-9_]+$' "$LOG"
}

_fsr_run_one() {
    local KEY="$1" DEVICE="$2" PARALLEL="$3" STARTED ELAPSED
    local rc=0
    STARTED=$(date +%s)
    JOB_CMD=()
    job_command "$KEY"
    if [ "$PARALLEL" = 1 ]; then
        GPU="$DEVICE" fsr_run "$(job_log "$KEY")" "${JOB_CMD[@]}" \
            > /dev/null || rc=$?
    else
        GPU="$DEVICE" fsr_run "$(job_log "$KEY")" "${JOB_CMD[@]}" || rc=$?
    fi
    ELAPSED=$(( $(date +%s) - STARTED ))
    if [ "$rc" -eq 0 ]; then
        echo "  [gpu $DEVICE] $KEY done in ${ELAPSED}s"
    else
        echo "  [gpu $DEVICE] $KEY FAILED rc=$rc after ${ELAPSED}s"
    fi
    return "$rc"
}

# fsr_dispatch <key>...
# Returns non-zero when any job failed, after running the rest.
fsr_dispatch() {
    local KEYS=("$@")
    local GPU_LIST=()
    mapfile -t GPU_LIST < <(fsr_gpus)

    local PENDING=() KEY
    for KEY in "${KEYS[@]}"; do
        if [ -f "$(job_output "$KEY")" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "  [skip] $KEY: $(job_output "$KEY") is already present"
        else
            PENDING+=("$KEY")
        fi
    done
    [ "${#PENDING[@]}" -eq 0 ] && return 0

    local PARALLEL=0
    [ "${#GPU_LIST[@]}" -gt 1 ] && [ "${#PENDING[@]}" -gt 1 ] && PARALLEL=1
    echo "running ${#PENDING[@]} jobs across ${#GPU_LIST[@]} GPUs"
    if [ "$PARALLEL" = 1 ]; then
        echo "follow a job with:"
        for KEY in "${PENDING[@]}"; do
            echo "  tail -F $(job_log "$KEY")"
        done
    fi

    local FAILED=() PIDS=() NAMES=() DEVICE i j
    i=0
    while [ "$i" -lt "${#PENDING[@]}" ]; do
        PIDS=()
        NAMES=()
        for DEVICE in "${GPU_LIST[@]}"; do
            [ "$i" -lt "${#PENDING[@]}" ] || break
            KEY="${PENDING[$i]}"
            echo "  [gpu $DEVICE] $KEY started"
            _fsr_run_one "$KEY" "$DEVICE" "$PARALLEL" &
            PIDS+=($!)
            NAMES+=("$KEY")
            i=$((i + 1))
        done
        for j in "${!PIDS[@]}"; do
            wait "${PIDS[$j]}" || FAILED+=("${NAMES[$j]}")
        done
    done

    if [ "${#FAILED[@]}" -gt 0 ]; then
        echo "FAILED: ${FAILED[*]}" >&2
        return 1
    fi
    return 0
}
