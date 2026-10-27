"""ModelOpt's specdec_bench run.py with our TensorRT-LLM 1.0 workarounds. Run with the TensorRT-LLM venv's python:

    $LAB_DATA/venvs/trtllm/bin/python scripts/specdec_run.py [--patch-lm-head] -- <specdec_bench run.py args>
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "third_party/Model-Optimizer/examples/specdec_bench"

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")


def tolerate_newer_kwargs():
    import tensorrt_llm.llmapi as llmapi

    def wrap(cls):
        class Compat(cls):
            def __init__(self, **kwargs):
                dropped = {k for k, v in kwargs.items() if k not in cls.model_fields and v is None}
                if dropped:
                    print(f"[specdec_run] {cls.__name__}: dropping unset {sorted(dropped)} (not in TensorRT-LLM 1.0)")
                super().__init__(**{k: v for k, v in kwargs.items() if k not in dropped})

        Compat.__name__ = cls.__name__
        return Compat

    for name in ("EagleDecodingConfig", "CudaGraphConfig", "KvCacheConfig"):
        setattr(llmapi, name, wrap(getattr(llmapi, name)))

    import tensorrt_llm

    base_llm = tensorrt_llm.LLM

    class LLM(base_llm):
        def __init__(self, *args, **kwargs):
            if kwargs.get("sampler_type") == "TorchSampler":
                del kwargs["sampler_type"]
                print("[specdec_run] LLM: dropping sampler_type=TorchSampler (TensorRT-LLM 1.0 default)")
            super().__init__(*args, **kwargs)

    tensorrt_llm.LLM = LLM


def main():
    tolerate_newer_kwargs()
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ours, theirs = argv[:split], argv[split + 1:]
    if "--patch-lm-head" in ours:
        sys.path.insert(0, str(ROOT / "scripts"))
        from trtllm_lm_head_patch import apply

        apply()
    sys.path.insert(0, str(BENCH))
    sys.argv = [str(BENCH / "run.py"), *theirs]
    runpy.run_path(str(BENCH / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
