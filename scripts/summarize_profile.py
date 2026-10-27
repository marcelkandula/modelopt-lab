"""GPU time per decode step by kernel category, from a profile_decode.py capture.

    python scripts/summarize_profile.py <report.nsys-rep> --name trtllm_int4_awq --stats <report>.json

"""

import argparse
import csv
import io
import json
import os
import re
import subprocess

import lab  # noqa: F401

from lab.results import log_result

# First match wins, names are lower-cased kernel names
CATEGORIES = [
    ("gemm_int4", r"fpaintb|w4a16|weight_?only|finegrained|mixed_?dtype|int4"),
    ("norm", r"norm"),
    ("attention", r"mha|xqa|attention|flash_?attn|paged|kv_?cache"),
    ("gemm_bf16", r"gemm|gemv|cutlass|xmma|cublas|matmul|s16816"),
    ("activation", r"silu|swiglu|gelu|act_and_mul|activation"),
    ("elementwise", r"elementwise"),  # with AWQ: x * pre_quant_scale before o_proj / down_proj
    ("sampling", r"topk|top_k|sampl|argmax|softmax|logits"),
]


def category(name: str) -> str:
    n = name.lower()
    return next((c for c, pat in CATEGORIES if re.search(pat, n)), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--name", required=True)
    ap.add_argument("--stats", required=True, help="JSON written by profile_decode.py --stats-out")
    args = ap.parse_args()
    with open(args.stats) as f:
        stats = json.load(f)
    steps = stats["steps"]

    nsys = os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nsys")
    out = subprocess.run([nsys if os.path.exists(nsys) else "nsys", "stats", "--report", "cuda_gpu_kern_sum",
                          "--format", "csv", "--force-export=true", args.report],
                         check=True, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(out[out.index("Time (%)"):])))
    per_cat, top = {}, []
    for r in rows:
        ns = float(r["Total Time (ns)"])
        cat = category(r["Name"])
        per_cat[cat] = per_cat.get(cat, 0.0) + ns
        top.append((ns, cat, r["Name"][:90], int(r["Instances"])))
    total = sum(per_cat.values())
    step_ms = {c: round(v / steps / 1e6, 3) for c, v in sorted(per_cat.items(), key=lambda x: -x[1])}
    gpu_ms, wall_ms = total / steps / 1e6, stats["wall_ms"] / steps
    log_result("profile_qwen3_8b", server=args.name, tokens=stats["tokens"], steps=steps,
               tokens_per_step=round(stats["tokens"] / steps, 3), gpu_ms_per_step=round(gpu_ms, 3),
               wall_ms_per_step=round(wall_ms, 3), ms_per_step_by_category=step_ms,
               top_kernels=[t[1:] for t in sorted(top, reverse=True)[:10]])
    print(f"{args.name}: {stats['tokens']} tokens in {steps} steps ({stats['tokens'] / steps:.2f} per step), "
          f"{gpu_ms:.2f} ms GPU time / {wall_ms:.2f} ms wall per step")
    for c, ms in step_ms.items():
        print(f"  {c:12s} {ms:7.3f} ms  {100 * per_cat[c] / total:5.1f}%")
    for ns, cat, name, n in sorted(top, reverse=True)[:8]:
        print(f"  {ns / total:6.1%}  [{cat}] {name}  x{n}")


if __name__ == "__main__":
    main()
