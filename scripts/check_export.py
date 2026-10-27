"""Unpack an exported INT4-AWQ / INT8 weight-only checkpoint back into BF16 and evaluate it in HF.

    python scripts/check_export.py qwen3-8b_int4_awq [--tasks wikitext,gsm8k] [--limit 250]

"""

import argparse
import json

import lab  # noqa: F401
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.evaluate import lm_eval_task, wikitext_ppl
from lab.paths import EXPORTS_DIR
from lab.results import log_result


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    # [out/2, in] uint8 -> [out, in] int8; low nibble = even output row, high nibble = odd row (ModelOpt pack_int4_in_uint8)
    p = packed.to(torch.int16)
    lo, hi = p & 0x0F, (p >> 4) & 0x0F
    lo, hi = torch.where(lo > 7, lo - 16, lo), torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], dim=1).reshape(-1, packed.shape[1]).to(torch.int8)


def dequantize(f, name: str) -> torch.Tensor:
    w, scale = f.get_tensor(f"{name}.weight"), f.get_tensor(f"{name}.weight_scale").float()
    if w.dtype == torch.uint8:
        w = unpack_int4(w).float()
        w = w * scale.repeat_interleave(w.shape[1] // scale.shape[1], dim=1)
    else:
        w = w.float() * scale.reshape(-1, 1)
    if f"{name}.pre_quant_scale" in f.keys():
        # y = W (x * pqs) = (W * pqs) x
        w = w * f.get_tensor(f"{name}.pre_quant_scale").float()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--tasks", default="wikitext,gsm8k")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    d = EXPORTS_DIR / args.export
    info = json.loads((d / "export_info.json").read_text())
    model_id = info["model"]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    params = dict(model.named_parameters())
    replaced = 0
    for shard in sorted(d.glob("*.safetensors")):
        with safe_open(shard, "pt", device="cuda") as f:
            for key in f.keys():
                if key.endswith(".weight_scale"):
                    name = key.removesuffix(".weight_scale")
                    params[f"{name}.weight"].data.copy_(dequantize(f, name).to(torch.bfloat16))
                    replaced += 1
                elif key.endswith("norm.weight") and key in params:
                    # Norm weights carry the folded q/k/v and gate/up pre_quant_scale
                    params[key].data.copy_(f.get_tensor(key))
    print(f"{args.export}: replaced {replaced} quantized weights")

    for task in args.tasks.split(","):
        if task == "wikitext":
            metric, value = "word_ppl", wikitext_ppl(model, model_id)["word_ppl"]
        else:
            m = lm_eval_task(model, tok, task, 5, args.limit, batch_size=32)
            metric = "exact_match,flexible-extract"
            value = m[metric]
        log_result("export_check", export=args.export, fmt=info["fmt"], model=model_id, task=task, limit=args.limit,
                   metric=metric, value=value)
        print(f"  {task:10s} {metric:30s} {value:.4f}")


if __name__ == "__main__":
    main()
