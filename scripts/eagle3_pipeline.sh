#!/usr/bin/env bash
# EAGLE-3 head for a Qwen3 target with ModelOpt's examples/speculative_decoding, one stage at a time:
#   bash scripts/eagle3_pipeline.sh <generate|vocab|train|validate|export|all>
# MODEL=Qwen/Qwen3-8B NAME=qwen3-8b N_CONV=15000 MAX_TOKENS=1024 SEQ_LEN=2048 EPOCHS=1 DRAFT_VOCAB=32000 (env overrides)
set -euo pipefail
: "${LAB_DATA:?run 'source env.sh' first}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
NAME="${NAME:-qwen3-8b}"
N_CONV="${N_CONV:-15000}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
SEQ_LEN="${SEQ_LEN:-2048}"
EPOCHS="${EPOCHS:-1}"
# Needs at least this many distinct tokens in the generated data
DRAFT_VOCAB="${DRAFT_VOCAB:-32000}"
PORT="${PORT:-8101}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPEC="$ROOT/third_party/Model-Optimizer/examples/speculative_decoding"
DATA="$LAB_DATA/datasets/eagle3"
PROMPTS="$DATA/prompts/daring_anteater.jsonl"
SUBSET="$DATA/prompts/subset_${N_CONV}.jsonl"
GEN="$DATA/generated/${NAME}_${N_CONV}.jsonl"
VOCAB="$DATA/draft_vocab/${NAME}"
CKPT="$LAB_DATA/eagle3_ckpts/${NAME}_${N_CONV}"
HEAD="$LAB_DATA/heads/${NAME}_eagle3_own_${N_CONV}"
mkdir -p "$DATA/generated" "$LAB_DATA/logs"

generate() {
  # Fixed subset (first N): server_generate.py resumes by position; it reads "conversations", make_dataset writes "messages"
  [ -f "$SUBSET" ] || python - "$PROMPTS" "$SUBSET" "$N_CONV" <<'EOF'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f, open(dst, "w") as g:
    for i, line in zip(range(n), f):
        g.write(json.dumps({"conversations": json.loads(line)["messages"]}) + "\n")
EOF
  local log="$LAB_DATA/logs/eagle3_generate_${NAME}.log"
  # Thinking off, as in the MT-Bench benchmark, so the head learns the distribution it will serve
  vllm serve "$MODEL" --port "$PORT" --served-model-name model --max-model-len 4096 --gpu-memory-utilization 0.85 \
    --default-chat-template-kwargs '{"enable_thinking": false}' > "$log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT
  until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $pid 2>/dev/null || { echo "vLLM exited, see $log"; return 1; }
    sleep 5
  done
  python "$SPEC/scripts/server_generate.py" --data_path "$SUBSET" --output_path "$GEN" --url "http://localhost:$PORT/v1" \
    --model model --num_threads 128 --max_tokens "$MAX_TOKENS" --temperature 0
  kill $pid; wait $pid 2>/dev/null || true
  echo "generated $(wc -l < "$GEN") conversations -> \$LAB_DATA/datasets/eagle3/generated/$(basename "$GEN")"
}

vocab() {
  python "$SPEC/scripts/calibrate_draft_vocab.py" --model "$MODEL" --data "$GEN" --draft_vocab_size "$DRAFT_VOCAB" --save_dir "$VOCAB"
}

train() {
  # launch_train.sh resolves main.py and accelerate relative to the example directory
  (cd "$SPEC" && ./launch_train.sh --config "$ROOT/configs/eagle3_qwen3.yaml" \
    model.model_name_or_path="$MODEL" data.data_path="$GEN" data.draft_vocab_cache="$(ls "$VOCAB"/*/d2t.pt "$VOCAB"/d2t.pt 2>/dev/null | head -1)" \
    training.output_dir="$CKPT" training.training_seq_len="$SEQ_LEN" training.num_train_epochs="$EPOCHS" \
    eagle.eagle_architecture_config.draft_vocab_size="$DRAFT_VOCAB" "$@")
}

validate() {
  python "$SPEC/scripts/ar_validate.py" --model_path "$CKPT" --steps 3 --osl 128 --per_category
}

export_head() {
  python "$SPEC/scripts/export_hf_checkpoint.py" --model_path "$CKPT" --export_path "$HEAD"
  # TensorRT-LLM 1.0 loads a 1-layer EAGLE-3 head as `midlayer.*` (original EAGLE naming); ModelOpt exports `layers.0.*`
  python - "$HEAD" <<'EOF'
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file
for shard in Path(sys.argv[1]).glob("*.safetensors"):
    sd = load_file(shard)
    save_file({k.replace("layers.0.", "midlayer.", 1): v for k, v in sd.items()}, shard, metadata={"format": "pt"})
EOF
  echo "head -> \$LAB_DATA/heads/$(basename "$HEAD")"
}

case "${1:-}" in
  generate) generate ;;
  vocab) vocab ;;
  train) shift; train "$@" ;;
  validate) validate ;;
  export) export_head ;;
  all) generate && vocab && train && validate && export_head ;;
  *) echo "usage: $0 <generate|vocab|train|validate|export|all>"; exit 1 ;;
esac
