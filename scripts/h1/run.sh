#!/usr/bin/env bash
# The full H1 pass.
##
# Usage:
#   GPUS=0,1,2 bash scripts/h1/run.sh
#
# Environment:
#   GPUS         comma-separated GPUs to spread models across (default 0)
#   STAGES       comma-separated stage numbers to run (default all)
#   MODELS       comma-separated registry slugs (default the whole roster)
#   FSR_RUNNER   runner to source (default scripts/runners/local.sh)
#   DATA_ROOT    corpus directory (default data/nq)
#   LIMIT        cap on records and queries, for a smoke pass
#   FORCE        1 to re-run every stage from scratch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAGES="${STAGES:-1,2,3,4}"
DATA_ROOT="${DATA_ROOT:-data/nq}"
export DATA_ROOT
STARTED=$(date +%s)

wanted() {
    case ",${STAGES}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

stage() {
    local NUMBER="$1" TITLE="$2"
    shift 2
    wanted "$NUMBER" || return 0
    echo
    echo "============================================================"
    echo "=== stage $NUMBER: $TITLE"
    echo "============================================================"
    "$@"
}

stage 1 "cross-query, test" bash scripts/h1/measurement.sh test
stage 2 "within-query, test" bash scripts/h1/within_query.sh test
stage 3 "cross-query, nq_val" bash scripts/h1/measurement.sh nq_val
stage 4 "within-query, nq_val" bash scripts/h1/within_query.sh nq_val
stage 5 "cross-query, all" bash scripts/h1/measurement.sh all
stage 6 "within-query, all" bash scripts/h1/within_query.sh all

echo
echo "H1 pass complete in $(( ($(date +%s) - STARTED) / 60 )) min"
echo "  cross-query:  $DATA_ROOT/h1_measurement/"
echo "  within-query: $DATA_ROOT/h1_within_query/"
