"""Quantize with a study config's format + default calibration and export a deployable checkpoint to $LAB_DATA/exports.

    python scripts/export_checkpoint.py configs/ptq_qwen3_8b.yaml int4_awq [int8_sq ...]

"""

import argparse
import gc
import json

import lab  # noqa: F401
import torch
import yaml
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.paths import EXPORTS_DIR
from lab.quant import calib_id, calib_loader, forward_loop_from, make_config, precision_summary, quantize


def export_dir(model_id: str, fmt: str):
    return EXPORTS_DIR / f"{model_id.split('/')[-1].lower()}_{fmt}"


def add_legacy_config_keys(out):
    # transformers 5 writes "dtype" and nests rope_theta in "rope_parameters"; engines on transformers 4.x
    # (TensorRT-LLM 1.0) read "torch_dtype" / "rope_theta" and silently fall back to defaults (RoPE base 10k instead of 1M)
    path = out / "config.json"
    cfg = json.loads(path.read_text())
    cfg.setdefault("torch_dtype", cfg.get("dtype", "bfloat16"))
    rope = cfg.get("rope_parameters") or {}
    if "rope_theta" in rope:
        cfg.setdefault("rope_theta", rope["rope_theta"])
    if rope.get("rope_type", "default") != "default":
        cfg.setdefault("rope_scaling", rope)
    path.write_text(json.dumps(cfg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("formats", nargs="+")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_id, calib = cfg["model"], cfg["calib"]
    tok = AutoTokenizer.from_pretrained(model_id)
    for fmt in args.formats:
        out = export_dir(model_id, fmt)
        if (out / "export_info.json").exists():
            print(f"{fmt}: already exported")
            continue
        spec = cfg["formats"][fmt]
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        qcfg = make_config(spec["base"], spec.get("exclude_layers", ()), model.config.num_hidden_layers, spec.get("kv_cache"),
                           spec.get("quantize_lm_head", False))
        model, secs = quantize(model, qcfg, forward_loop_from(calib_loader(tok, **calib)))
        info = {"model": model_id, "fmt": fmt, "spec": spec, "calib": calib_id(calib), "quant_seconds": round(secs, 1),
                **precision_summary(model)}
        export_hf_checkpoint(model, export_dir=out)
        tok.save_pretrained(out)
        add_legacy_config_keys(out)
        info["disk_gb"] = round(sum(p.stat().st_size for p in out.glob("*.safetensors")) / 1e9, 2)
        (out / "export_info.json").write_text(json.dumps(info, indent=2))
        print(f"{fmt}: {info['disk_gb']} GB -> $LAB_DATA/exports/{out.name}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
