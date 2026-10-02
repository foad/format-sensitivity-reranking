#!/usr/bin/env bash
# Spread per-model jobs across GPUs in waves.
#
#   job_command <slug>   sets JOB_CMD to the script and arguments for one model
#   job_output  <slug>   echoes the file that means the model is done
#   job_log     <slug>   echoes the file its detailed output goes to
#
# Environment:
#   GPUS   comma or space separated GPUs (default 0)
#   GPU    a single GPU, an alias for GPUS
#   FORCE  1 to re-run a model whose output is present

fsr_gpus() {
    local LIST
    IFS=', ' read -ra LIST <<< "${GPUS:-${GPU:-0}}"
    printf '%s\n' "${LIST[@]}"
}

fsr_roster() {
    local SCRIPT="$1" LOG="$2"
    GPU="$(fsr_gpus | head -1)" fsr_run "$LOG" "$SCRIPT" --list-models > /dev/null
    grep -E '^[a-z0-9_]+$' "$LOG"
}

_fsr_run_one() {
    local SLUG="$1" DEVICE="$2" PARALLEL="$3" STARTED ELAPSED
    local rc=0
    STARTED=$(date +%s)
    JOB_CMD=()
    job_command "$SLUG"
    if [ "$PARALLEL" = 1 ]; then
        GPU="$DEVICE" fsr_run "$(job_log "$SLUG")" "${JOB_CMD[@]}" \
            > /dev/null || rc=$?
    else
        GPU="$DEVICE" fsr_run "$(job_log "$SLUG")" "${JOB_CMD[@]}" || rc=$?
    fi
    ELAPSED=$(( $(date +%s) - STARTED ))
    if [ "$rc" -eq 0 ]; then
        echo "  [gpu $DEVICE] $SLUG done in ${ELAPSED}s"
    else
        echo "  [gpu $DEVICE] $SLUG FAILED rc=$rc after ${ELAPSED}s"
    fi
    return "$rc"
}

# fsr_dispatch <slug>...
# Returns non-zero when any model failed, after running the rest.
fsr_dispatch() {
    local SLUGS=("$@")
    local GPU_LIST=()
    mapfile -t GPU_LIST < <(fsr_gpus)

    local PENDING=() SLUG
    for SLUG in "${SLUGS[@]}"; do
        if [ -f "$(job_output "$SLUG")" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "  [skip] $SLUG: $(job_output "$SLUG") is already present"
        else
            PENDING+=("$SLUG")
        fi
    done
    [ "${#PENDING[@]}" -eq 0 ] && return 0

    local PARALLEL=0
    [ "${#GPU_LIST[@]}" -gt 1 ] && [ "${#PENDING[@]}" -gt 1 ] && PARALLEL=1
    echo "running ${#PENDING[@]} models across ${#GPU_LIST[@]} GPUs"
    if [ "$PARALLEL" = 1 ]; then
        echo "follow a model with:"
        for SLUG in "${PENDING[@]}"; do
            echo "  tail -F $(job_log "$SLUG")"
        done
    fi

    local FAILED=() PIDS=() NAMES=() DEVICE i j
    i=0
    while [ "$i" -lt "${#PENDING[@]}" ]; do
        PIDS=()
        NAMES=()
        for DEVICE in "${GPU_LIST[@]}"; do
            [ "$i" -lt "${#PENDING[@]}" ] || break
            SLUG="${PENDING[$i]}"
            echo "  [gpu $DEVICE] $SLUG started"
            _fsr_run_one "$SLUG" "$DEVICE" "$PARALLEL" &
            PIDS+=($!)
            NAMES+=("$SLUG")
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
