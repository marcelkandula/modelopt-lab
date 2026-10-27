"""Speculative decoding benchmark: every (target, draft) pair from the config through ModelOpt's specdec_bench on TensorRT-LLM.

    python scripts/bench_specdec.py configs/specdec_qwen3_8b.yaml [--targets bf16] [--drafts none] [--num-requests 4]

Acceptance length (tokens per target step) and timing per pair, one row each in results/<results>.jsonl.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import lab  # noqa: F401
import yaml

from lab.paths import DATASETS_DIR, EXPORTS_DIR, LAB_DATA, LAB_ROOT
from lab.results import load_results, log_result

TRTLLM_PY = LAB_DATA / "venvs" / "trtllm" / "bin" / "python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--targets", help="comma-separated subset")
    ap.add_argument("--drafts", help="comma-separated subset")
    ap.add_argument("--num-requests", type=int, help="fewer MT-Bench questions (smoke test, not logged)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    b = cfg["bench"]
    done = set(load_results(cfg["results"]).get("run", []))
    params = Path(tempfile.mkstemp(suffix=".yaml")[1])
    params.write_text(yaml.safe_dump({"chat_template_args": cfg["chat_template_args"], "engine_args": cfg.get("engine_args", {})}))

    jobs = [(tname, target, dname, draft, k)
            for tname, target in cfg["targets"].items() if not args.targets or tname in args.targets.split(",")
            for dname, draft in cfg["drafts"].items() if not args.drafts or dname in args.drafts.split(",")
            for k in (draft.get("draft_lengths", b["draft_lengths"]) if draft else [0])]

    for tname, target, dname, draft, k in jobs:
        run = f"{tname}+{dname}" + (f"+k{k}" if draft else "")
        if run in done and not args.num_requests:
            print(f"{run:40s} done")
            continue
        out = LAB_DATA / "specdec" / run
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.json"):
            f.unlink()  # specdec_bench appends to existing metric files
        model = target.get("model") or str(EXPORTS_DIR / target["export"])
        cmd = [str(TRTLLM_PY), str(LAB_ROOT / "scripts/specdec_run.py")]
        cmd += ["--patch-lm-head"] if target.get("patch_lm_head") or draft.get("patch_lm_head") else []
        cmd += ["--", "--model_dir", model, "--tokenizer", cfg["tokenizer"], "--mtbench", str(DATASETS_DIR / cfg["mtbench"]),
                "--engine", "TRTLLM", "--tp_size", "1", "--ep_size", "1", "--concurrency", str(b["concurrency"]),
                "--output_length", str(b["output_length"]), "--max_seq_len", str(b["max_seq_len"]),
                "--runtime_params", str(params), "--postprocess", "base", "--save_dir", str(out)]
        if draft:
            cmd += ["--speculative_algorithm", "EAGLE3", "--draft_model_dir", str(LAB_DATA / "heads" / draft["head"]),
                    "--draft_length", str(k)]
        else:
            cmd += ["--speculative_algorithm", "NONE"]
        if args.num_requests:
            cmd += ["--num_requests", str(args.num_requests)]
        print(f"=== {run}", flush=True)
        log = LAB_DATA / "logs" / f"specdec_{run}.log"
        with log.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  env={key: val for key, val in os.environ.items() if key not in ("VIRTUAL_ENV", "PYTHONPATH")})
        # TensorRT-LLM 1.0 sometimes segfaults in its cuBLAS destructor at process exit, after the metrics are written
        if not (out / "acceptance_rate.json").exists():
            print(f"=== {run} FAILED ({proc.returncode}), see logs/{log.name}", flush=True)
            continue
        if proc.returncode != 0:
            print(f"  exited with {proc.returncode} after writing its metrics (teardown crash), keeping them", flush=True)
        mt = json.loads((out / "acceptance_rate.json").read_text())[-1]
        timing = json.loads((out / "timing.json").read_text())[-1]
        row = dict(run=run, target=tname, draft=dname, draft_length=k,
                   acceptance_length=mt["Average_AL"], category_al=mt["Category_AL"],
                   joint_acceptance=mt.get("Joint_Acceptance_Rate"), al_histogram=mt.get("Acceptance_Length_Histogram"),
                   output_tok_s=timing["Output TPS"],
                   step_ms=1000 * float(timing["Request Generation Step Time"]["mean"]), timing=timing)
        print(f"  AL {row['acceptance_length']:.2f}   step {row['step_ms']:.1f} ms   output {row['output_tok_s']:.1f} tok/s",
              flush=True)
        if not args.num_requests:
            log_result(cfg["results"], config=Path(args.config).name, **row)

if __name__ == "__main__":
    main()
