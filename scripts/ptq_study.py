"""Quantize one model per (format, calibration) with ModelOpt, evaluate a task list, log one JSONL row per task.

    python scripts/ptq_study.py configs/ptq_qwen3_8b.yaml [--formats int4_awq,int8_sq] [--tasks wikitext,gsm8k]
                                [--limit 20] [--dry-run]

Resumable: runs whose (model, format spec, calibration, task) are already in the results file are skipped.
--limit N caps samples per lm-eval task and writes to <results>_limit<N>.jsonl instead.
"""

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import lab  # noqa: F401
import modelopt.torch.quantization as mtq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.evaluate import lm_eval_task, wikitext_ppl
from lab.quant import AMPERE_DEPLOYABLE, calib_id, calib_loader, forward_loop_from, make_config, precision_summary, quantize
from lab.results import load_results, log_result


def run_key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def task_key(t: dict) -> dict:
    # batch_size and metric don't change the measurement
    return {k: v for k, v in t.items() if k not in ("batch_size", "metric")}


def run_task(model, tok, model_id: str, t: dict, eval_cfg: dict, limit: int | None) -> dict:
    if t["task"] == "wikitext":
        return wikitext_ppl(model, model_id, t.get("max_len", 2048), t.get("stride", 1024))
    return lm_eval_task(model, tok, t["task"], t.get("num_fewshot"), limit,
                        batch_size=t.get("batch_size", eval_cfg["batch_size"]), max_length=eval_cfg["max_length"])


def run(model_id, tok, fmt, spec, calib, todo, keys, cfg, results, args):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    secs = 0.0
    if spec:
        qcfg = make_config(spec["base"], spec.get("exclude_layers", ()), model.config.num_hidden_layers,
                           spec.get("kv_cache"), spec.get("quantize_lm_head", False))
        model, secs = quantize(model, qcfg, forward_loop_from(calib_loader(tok, **calib)))
    precision = precision_summary(model)
    quant_peak = torch.cuda.max_memory_allocated() / 1e9
    if spec:
        mtq.fold_weight(model)

    base = spec.get("base", "bf16")
    for t in todo:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            metrics = run_task(model, tok, model_id, t, cfg["eval"], args.limit)
        except Exception as e:
            hint = " (lower its batch_size)" if isinstance(e, torch.OutOfMemoryError) else ""
            print(f"  {t['task']}: {type(e).__name__}{hint}, skipped, reruns next time:\n{traceback.format_exc()}",
                  flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        metric = t["metric"] if "," in t["metric"] or t["task"] == "wikitext" else f"{t['metric']},none"
        stderr = metrics.get(metric.replace(",", "_stderr,", 1)) if "," in metric else None
        row = log_result(
            results, run_key=keys[t["task"]], config=Path(args.config).name, model=model_id, fmt=fmt, base=base,
            kv_cache=spec.get("kv_cache"), calib=calib_id(calib) if calib else None, calib_cfg=calib,
            task=t["task"], num_fewshot=t.get("num_fewshot"), limit=args.limit, metric=metric,
            value=metrics[metric], stderr=stderr, metrics=metrics,
            deployable_on_ampere=base == "bf16" or base in AMPERE_DEPLOYABLE, quant_seconds=round(secs, 1),
            eval_seconds=round(time.perf_counter() - t0, 1), quant_peak_gb=round(quant_peak, 1),
            eval_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), **precision,
        )
        print(f"  {t['task']:14s} {metric:30s} {row['value']:8.4f}  ({row['eval_seconds']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--formats", help="comma-separated subset")
    ap.add_argument("--tasks", help="comma-separated subset")
    ap.add_argument("--limit", type=int, help="samples per lm-eval task (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="list pending runs and exit")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_id = cfg["model"]
    results = cfg["results"] + (f"_limit{args.limit}" if args.limit else "")
    formats = {k: v for k, v in cfg["formats"].items() if not args.formats or k in args.formats.split(",")}
    tasks = [t for t in cfg["tasks"] if not args.tasks or t["task"] in args.tasks.split(",")]
    calibs = [{**cfg["calib"], **c} for c in cfg.get("calib_grid", [{}])]
    df = load_results(results)
    done = set(df.run_key) if "run_key" in df else set()

    tok = AutoTokenizer.from_pretrained(model_id)
    for fmt, spec in formats.items():
        for calib in calibs if spec else [None]:
            calib_key = {k: v for k, v in calib.items() if k != "batch_size"} if calib else None
            keys = {t["task"]: run_key(model_id, spec, calib_key, task_key(t), args.limit) for t in tasks}
            todo = [t for t in tasks if keys[t["task"]] not in done]
            label = f"{fmt} [{calib_id(calib) if calib else 'no calib'}]"
            if not todo or args.dry_run:
                print(f"{label:50s} {'done' if not todo else 'pending: ' + ', '.join(t['task'] for t in todo)}")
                continue

            print(f"=== {label}: {', '.join(t['task'] for t in todo)}", flush=True)
            try:
                run(model_id, tok, fmt, spec, calib, todo, keys, cfg, results, args)
            except Exception:
                print(f"=== {label} FAILED, skipped:\n{traceback.format_exc()}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
