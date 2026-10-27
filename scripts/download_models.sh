#!/usr/bin/env bash
set -euo pipefail
: "${HF_HOME:?run 'source env.sh' first}"

case "${1:-llm}" in
  llm) models=(Qwen/Qwen3-8B) ;;
  *) echo "usage: $0 [llm]"; exit 1 ;;
esac

for m in "${models[@]}"; do
  echo ">>> $m"
  # One pattern per --include: space-separated patterns are silently misparsed
  hf download "$m" --include "*.json" --include "*.safetensors" --include "*.txt" --include "*.model" --include "*.jinja"
done
du -sh "$HF_HOME/hub"
