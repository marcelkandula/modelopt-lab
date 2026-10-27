#!/usr/bin/env bash
# TensorRT-LLM in its own venv: it pins torch 2.7 / transformers 4.53 / numpy<2, and releases >= 1.1 need CUDA 13 (driver >= 580)
set -euo pipefail
: "${LAB_DATA:?run 'source env.sh' first}"

VENV="$LAB_DATA/venvs/trtllm"
uv venv --python 3.12 --allow-existing "$VENV"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.nvidia.com \
  "torch==2.7.1" "torchvision==0.22.1" "tensorrt_llm==1.0.0" "pyarrow<20"
"$VENV/bin/python" -c "import tensorrt_llm, torch; print('tensorrt_llm', tensorrt_llm.__version__, 'torch', torch.__version__)"
