"""Import `lab` before transformers/huggingface_hub: it sets HF_HOME etc., which they read at import time."""

import os as _os
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]

if (_ROOT / ".env").exists():
    for _line in (_ROOT / ".env").read_text().splitlines():
        _key, _sep, _val = _line.partition("=")
        _key = _key.strip()
        if _sep and _key and not _key.startswith("#") and _key not in _os.environ:
            _os.environ[_key] = _val.strip()
_DATA = _os.environ.setdefault("LAB_DATA", str(_ROOT.parent / "llmopt-data"))
for _k, _v in {
    "HF_HOME": f"{_DATA}/hf_cache",
    "VLLM_CACHE_ROOT": f"{_DATA}/vllm_cache",
    "TRITON_CACHE_DIR": f"{_DATA}/triton_cache",
    "TORCHINDUCTOR_CACHE_DIR": f"{_DATA}/inductor_cache",
    "TORCH_EXTENSIONS_DIR": f"{_DATA}/torch_extensions",
    "TORCH_CUDA_ARCH_LIST": "8.6",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    _os.environ.setdefault(_k, _v)
if "CUDA_HOME" not in _os.environ and _os.path.exists(f"{_DATA}/cuda-12.9/bin/nvcc"):
    _os.environ["CUDA_HOME"] = f"{_DATA}/cuda-12.9"
    _os.environ["PATH"] = f"{_DATA}/cuda-12.9/bin:" + _os.environ.get("PATH", "")
