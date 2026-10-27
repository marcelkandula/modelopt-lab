#!/usr/bin/env bash
# Nsight Systems profile of TensorRT-LLM batch-1 decode, summarized into results/profile_qwen3_8b.jsonl.
#   bash scripts/profile_decode.sh <name> <model id or $LAB_DATA/exports/...> [tokens] [profile_decode.py options]
set -euo pipefail
: "${LAB_DATA:?run 'source env.sh' first}"

NAME="$1"; MODEL="$2"; TOKENS="${3:-64}"; shift $(( $# < 3 ? $# : 3 ))
OUT="$LAB_DATA/profiles/$NAME"
mkdir -p "$LAB_DATA/profiles"
rm -f "$OUT.json"

# TRT-LLM venv on PATH, main venv out of the way (same as bench_deploy.py)
env -u VIRTUAL_ENV -u PYTHONPATH PATH="$LAB_DATA/venvs/trtllm/bin:$PATH" \
  "$CUDA_HOME/bin/nsys" profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
  --trace=cuda,nvtx --sample=none --force-overwrite=true -o "$OUT" \
  "$LAB_DATA/venvs/trtllm/bin/python" scripts/profile_decode.py "$MODEL" --tokens "$TOKENS" --stats-out "$OUT.json" "$@" \
  || { [ -s "$OUT.json" ] && echo "crashed at teardown after the capture (TensorRT-LLM 1.0 with EAGLE-3), keeping it"; }

python scripts/summarize_profile.py "$OUT.nsys-rep" --name "$NAME" --stats "$OUT.json"
