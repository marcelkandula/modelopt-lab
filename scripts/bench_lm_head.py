"""Time the lm_head GEMM in BF16 vs TensorRT-LLM's INT4 (W4A16, group 128) kernel. Run with the TensorRT-LLM venv:

    $LAB_DATA/venvs/trtllm/bin/python scripts/bench_lm_head.py

Random weights: only the kernel time matters. Logs to results/lm_head_kernel.jsonl.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import tensorrt_llm  # noqa: E402, F401  registers torch.ops.trtllm
from tensorrt_llm.quantization.functional import preprocess_weights_for_mixed_gemm  # noqa: E402

from lab.results import log_result  # noqa: E402

HIDDEN, VOCAB, GROUP = 4096, 151936, 128


def time_ms(fn, iters: int = 200) -> float:
    for _ in range(20):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    w_bf16 = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
    packed = torch.randint(-128, 128, (HIDDEN, VOCAB // 2), dtype=torch.int8)  # [in, out/2], two INT4 per byte
    w_int4 = preprocess_weights_for_mixed_gemm(packed, torch.quint4x2, torch.float16).cuda()
    scales = torch.rand(HIDDEN // GROUP, VOCAB, dtype=torch.bfloat16, device="cuda") * 0.01

    for tokens in (1, 8, 32):
        x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        bf16 = time_ms(lambda: torch.nn.functional.linear(x, w_bf16))
        int4 = time_ms(lambda: torch.ops.trtllm.finegrained_mixed_dtype_gemm(
            input=x, weight=w_int4, scales=scales, group_size=GROUP, has_zero_point=False,
            output_dtype=torch.bfloat16, bias=None, zeros=None))
        log_result("lm_head_kernel", tokens=tokens, shape=[HIDDEN, VOCAB], bf16_ms=round(bf16, 3), int4_ms=round(int4, 3),
                   bf16_gbs=round(w_bf16.numel() * 2 / bf16 / 1e6), int4_gbs=round(w_int4.numel() / int4 / 1e6))
        print(f"tokens {tokens:3d}: BF16 {bf16:.3f} ms ({w_bf16.numel() * 2 / bf16 / 1e6:.0f} GB/s)   "
              f"INT4 {int4:.3f} ms ({w_int4.numel() / int4 / 1e6:.0f} GB/s)   speedup {bf16 / int4:.2f}x")


if __name__ == "__main__":
    main()
