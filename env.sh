# Usage: source env.sh

export LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env without overriding variables already exported in the shell
if [ -f "$LAB_ROOT/.env" ]; then
    while IFS='=' read -r _key _val || [ -n "$_key" ]; do
        _key="${_key//[[:space:]]/}"
        [[ -z "$_key" || "$_key" == \#* ]] && continue
        [ -z "${!_key:-}" ] && export "$_key=${_val%$'\r'}"
    done < "$LAB_ROOT/.env"
    unset _key _val
fi
export LAB_DATA="${LAB_DATA:-$(dirname "$LAB_ROOT")/llmopt-data}"

export HF_HOME="$LAB_DATA/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR="$LAB_DATA/uv_cache"
export VLLM_CACHE_ROOT="$LAB_DATA/vllm_cache"
export TRITON_CACHE_DIR="$LAB_DATA/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DATA/inductor_cache"

# System nvcc (11.5) can't build ModelOpt's C++20 extensions; use the local 12.9 toolkit from `make cuda`
if [ -x "$LAB_DATA/cuda-12.9/bin/nvcc" ]; then
    export CUDA_HOME="$LAB_DATA/cuda-12.9"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
export TORCH_CUDA_ARCH_LIST="8.6"
export TORCH_EXTENSIONS_DIR="$LAB_DATA/torch_extensions"

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TORCH_EXTENSIONS_DIR" "$LAB_DATA/exports" "$LAB_DATA/datasets"

if [ -f "$LAB_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$LAB_ROOT/.venv/bin/activate"
fi
