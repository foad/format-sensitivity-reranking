#!/usr/bin/env bash
# Package the release assets: the corpus, the split redraws, and the adapters.
#
# Usage:
#   bash scripts/release/package.sh
#
# Environment:
#   OUT          directory the archives are written to (default dist/release)
#   DATA_ROOT    corpus directory (default data/nq)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUT="${OUT:-dist/release}"
DATA_ROOT="${DATA_ROOT:-data/nq}"
TRAIN_DIR="$DATA_ROOT/h2/train"
mkdir -p "$OUT"

CORPUS=(
    "$DATA_ROOT/parsed_train.json"
    "$DATA_ROOT/parsed_validation.json"
    "$DATA_ROOT/splits"
    "$DATA_ROOT/prose.json"
    "$DATA_ROOT/prose_eval.json"
)
for path in "${CORPUS[@]}"; do
    [ -e "$path" ] || { echo "ERROR: $path missing" >&2; exit 2; }
done
echo "corpus"
tar -czf "$OUT/corpus.tar.gz" data/LICENSE "${CORPUS[@]}"

mapfile -t REDRAWS < <(find data -maxdepth 2 -path "data/nq_s*/splits" -type d | sort)
if [ "${#REDRAWS[@]}" -gt 0 ]; then
    echo "split redraws: ${#REDRAWS[@]}"
    PARTS=()
    for dir in "${REDRAWS[@]}"; do
        root="$(dirname "$dir")"
        PARTS+=("$dir" "$root/parsed_train.json" "$root/parsed_validation.json")
    done
    tar -czf "$OUT/corpus-redraws.tar.gz" data/LICENSE "${PARTS[@]}"
fi

# Each adapter archive carries its base model's licence beside the adapters.
LICENCES="$SCRIPT_DIR/licences"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$TRAIN_DIR"

licence_notice() {
    local model="$1" model_id="$2" pooler=""
    [[ "$model" == *_pooler ]] &&
        pooler=" The pooler layer of the base model is also trained, and saved with each adapter."
    local intro="The adapters under ${TRAIN_DIR}/${model}_*/adapter/ are LoRA adapters trained on
top of the frozen base model ${model_id}
(https://huggingface.co/${model_id}).${pooler} They modify the base model and
are distributed under its licence."
    case "$model_id" in
        jinaai/*)
            printf '%s\n\nLicence: Creative Commons Attribution-NonCommercial 4.0 International\n(CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/\n\nThe adapters may not be used for commercial purposes. The base model is\npublished by Jina AI under CC BY-NC 4.0.\n' "$intro"
            ;;
        BAAI/bge-reranker-base)
            printf '%s\n\nLicence: MIT, the licence of FlagEmbedding, under which the base model is\nreleased. Its text follows.\n\n' "$intro"
            cat "$LICENCES/mit-flagembedding.txt"
            ;;
        *)
            printf '%s\n\nLicence: Apache License 2.0. Its text follows.\n\n' "$intro"
            cat "$LICENCES/apache-2.0.txt"
            ;;
    esac
}

ARM='(5fmt|only_[a-z_]+|yaml|json|toml|inline_kv|markdown)_lam[0-9.]+(_s[0-9]+)?'
TOTAL=0
while read -r MODEL; do
    mapfile -t ADAPTERS < <(
        find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
            grep -E "^${MODEL}_${ARM}$" | sort |
            while read -r name; do
                [ -f "$TRAIN_DIR/$name/adapter/adapter_model.safetensors" ] &&
                    echo "$TRAIN_DIR/$name/adapter"
            done
    )
    if [ "${#ADAPTERS[@]}" -eq 0 ]; then
        echo "WARNING: no adapters for $MODEL" >&2
        continue
    fi
    echo "adapters, $MODEL: ${#ADAPTERS[@]}"
    MODEL_ID="$(PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --config "$MODEL" |
        sed -n "s/^MODEL_ID=//p" | tr -d "'")"
    NOTICE="$TRAIN_DIR/LICENSE-adapters-$MODEL.txt"
    licence_notice "$MODEL" "$MODEL_ID" > "$STAGE/$NOTICE"
    tar -czf "$OUT/adapters-$MODEL.tar.gz" -C "$STAGE" "$NOTICE" -C "$REPO_ROOT" "${ADAPTERS[@]}"
    TOTAL=$((TOTAL + ${#ADAPTERS[@]}))
done < <(PYTHONPATH="$REPO_ROOT/src" python3 -m fsr.models.registry --list)
echo "adapters in all: $TOTAL"

(cd "$OUT" && sha256sum ./*.tar.gz > SHA256SUMS)
echo "written to $OUT"
ls -lh "$OUT"
