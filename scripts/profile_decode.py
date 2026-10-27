"""Batch-1 decode under Nsight Systems, with TensorRT-LLM (PyTorch backend). Run with the TensorRT-LLM venv's python:

    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -o <out> \\
        $LAB_DATA/venvs/trtllm/bin/python scripts/profile_decode.py <model or checkpoint dir> [--tokens 64]

Only the measured generate call is captured (warm-up excluded). Summarize with scripts/summarize_profile.py.
"""

import argparse
import json
import os
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")  # in-process worker, so nsys sees its kernels

import torch  # noqa: E402
from tensorrt_llm import LLM, SamplingParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="exports carry a transformers-5 tokenizer TRT-LLM can't read")
    ap.add_argument("--patch-lm-head", action="store_true", help="serve a quantized lm_head (trtllm_lm_head_patch.py)")
    ap.add_argument("--draft", help="EAGLE-3 head directory (speculative decoding)")
    ap.add_argument("--draft-length", type=int, default=4)
    ap.add_argument("--stats-out", help="write engine steps and wall time of the measured call here (JSON)")
    args = ap.parse_args()
    if args.patch_lm_head:
        from trtllm_lm_head_patch import apply

        apply()

    spec = {}
    if args.draft:
        from tensorrt_llm.llmapi import EagleDecodingConfig

        # Overlap scheduler off: EAGLE-3 crashes with it on TensorRT-LLM 1.0 (as in configs/specdec_qwen3_8b.yaml)
        eagle = EagleDecodingConfig(max_draft_len=args.draft_length, speculative_model_dir=args.draft, eagle3_one_model=True)
        spec = {"speculative_config": eagle, "disable_overlap_scheduler": True}
    llm = LLM(model=args.model, tokenizer=args.tokenizer, max_seq_len=2048, max_batch_size=1, enable_iter_perf_stats=True, **spec)
    params = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True)
    prompt = "Explain how a GPU executes a matrix multiplication."
    for _ in range(3):
        llm.generate(prompt, params)
    llm.get_stats(timeout=2)

    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    out = llm.generate(prompt, params)
    wall_ms = 1e3 * (time.perf_counter() - t0)
    torch.cuda.profiler.stop()
    # One stats entry per engine iteration (prefill + target forward passes)
    steps = len(llm.get_stats(timeout=2))
    tokens = len(out.outputs[0].token_ids)
    print(f"generated {tokens} tokens in {steps} steps, {wall_ms:.0f} ms")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump({"tokens": tokens, "steps": steps, "wall_ms": round(wall_ms, 1)}, f)
    llm.shutdown()


if __name__ == "__main__":
    main()
