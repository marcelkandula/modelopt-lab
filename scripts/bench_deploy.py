"""Serve each (engine, format) on real kernels, then measure startup, memory, KV capacity, accuracy and serving speed.

    python scripts/bench_deploy.py configs/deploy_qwen3_8b.yaml [--servers trtllm_int4_awq] [--skip-accuracy] [--dry-run]

Every server is driven by the same client (`vllm bench serve` against /v1/completions) and the same lm-eval task.
Resumable: servers with all their rows already in the results file are skipped.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import lab  # noqa: F401
import yaml
from transformers import AutoConfig

from lab.evaluate import api_eval_task
from lab.hardware import A6000_MEM_BW_GBS, decode_ceiling_tok_s, kv_cache_bytes_per_token
from lab.paths import LAB_DATA, LAB_ROOT
from lab.results import load_results, log_result

sys.path.insert(0, str(LAB_ROOT / "scripts"))
from export_checkpoint import add_legacy_config_keys, export_dir  # noqa: E402

TRTLLM_BIN = LAB_DATA / "venvs" / "trtllm" / "bin"
KV_TOKENS = {  # server-log patterns for the KV-cache capacity in tokens
    "vllm": r"GPU KV cache size: ([\d,]+) tokens",
    "trtllm": r"for max tokens in paged KV cache \(([\d,]+)\)",
}


def hf_disk_gb(repo_id: str) -> float:
    from huggingface_hub import try_to_load_from_cache

    index = json.loads(Path(try_to_load_from_cache(repo_id, "model.safetensors.index.json")).read_text())
    return round(index["metadata"]["total_size"] / 1e9, 2)


def gpu_used_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0]) / 1024


def server_cmd(engine: str, path: str, name: str, s: dict, tokenizer: str, patch_lm_head: bool = False) -> tuple[list[str], dict]:
    env = dict(os.environ)
    if engine == "vllm":
        return ["vllm", "serve", path, "--port", str(s["port"]), "--served-model-name", name,
                "--max-model-len", str(s["max_seq_len"]), "--max-num-seqs", str(s["max_batch_size"]),
                "--gpu-memory-utilization", str(s["gpu_mem_fraction"])], env
    # Own venv first on PATH; the main venv's packages must not leak in
    env["PATH"] = f"{TRTLLM_BIN}:{env['PATH']}"
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    # Single GPU: run the worker in-process; spawning it through the system OpenMPI hangs
    env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    opts = Path(tempfile.mkstemp(suffix=".yaml")[1])
    opts.write_text(yaml.safe_dump({"kv_cache_config": {"free_gpu_memory_fraction": s["gpu_mem_fraction"]}}))
    # The export's tokenizer files are written by transformers 5, which TRT-LLM's transformers 4.53 can't parse
    launcher = ([str(TRTLLM_BIN / "python"), str(LAB_ROOT / "scripts/trtllm_lm_head_patch.py")] if patch_lm_head
                else [str(TRTLLM_BIN / "trtllm-serve")])
    return [*launcher, path, "--backend", "pytorch", "--port", str(s["port"]),
            "--tokenizer", tokenizer,
            "--max_seq_len", str(s["max_seq_len"]), "--max_batch_size", str(s["max_batch_size"]),
            "--extra_llm_api_options", str(opts)], env


def wait_ready(proc, url: str, log: Path, timeout_s: int = 1800) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited ({proc.returncode}):\n" + log.read_text()[-3000:])
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=5)
            return time.perf_counter() - t0
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"server not ready after {timeout_s}s")


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def bench(url: str, name: str, tokenizer: str, b: dict, concurrency: int, out_dir: Path) -> dict:
    n = max(b["min_prompts"], b["prompts_per_concurrency"] * concurrency)
    fname = f"{name}_c{concurrency}.json"
    subprocess.run(["vllm", "bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
                    "--base-url", url, "--model", tokenizer, "--served-model-name", name, "--tokenizer", tokenizer,
                    "--dataset-name", "random", "--random-input-len", str(b["input_len"]),
                    "--random-output-len", str(b["output_len"]), "--num-prompts", str(n),
                    "--max-concurrency", str(concurrency), "--ignore-eos", "--temperature", "0", "--seed", "0",
                    "--percentile-metrics", "ttft,tpot,itl", "--metric-percentiles", "50,99",
                    "--save-result", "--result-dir", str(out_dir), "--result-filename", fname],
                   check=True, capture_output=True, text=True)
    r = json.loads((out_dir / fname).read_text())
    if r.get("failed") or not r.get("completed"):
        raise RuntimeError(f"{r.get('failed')} of {n} requests failed, see the server log")
    keep = ["completed", "failed", "duration", "request_throughput", "output_throughput", "total_token_throughput",
            "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p50_tpot_ms", "p99_tpot_ms",
            "mean_itl_ms", "p99_itl_ms"]
    return {k: r[k] for k in keep if k in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--servers", help="comma-separated subset")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s, b, acc = cfg["serve"], cfg["bench"], cfg["accuracy"]
    model_id, url = cfg["model"], f"http://localhost:{cfg['serve']['port']}"
    servers = {k: v for k, v in cfg["servers"].items() if not args.servers or k in args.servers.split(",")}
    df = load_results(cfg["results"])
    done = set(zip(df.server, df.kind, df.concurrency.fillna(0))) if not df.empty else set()
    wanted = [("serve", 0)] + [("bench", c) for c in b["concurrency"]] + ([] if args.skip_accuracy else [("accuracy", 0)])

    hf = AutoConfig.from_pretrained(model_id)
    embed_bytes = hf.vocab_size * hf.hidden_size * 2  # decode reads one embedding row, not the table
    kv_mid = kv_cache_bytes_per_token(hf) * (b["input_len"] + b["output_len"] // 2)
    bench_dir = Path(tempfile.mkdtemp(prefix="bench_"))

    for name, spec in servers.items():
        todo = [w for w in wanted if (name, *w) not in done]
        if not todo or args.dry_run:
            print(f"{name:20s} {'done' if not todo else 'pending: ' + ', '.join(f'{k}{c or ""}' for k, c in todo)}")
            continue
        engine, fmt = spec["engine"], spec["fmt"]
        if fmt == "bf16":
            path, disk_gb = model_id, hf_disk_gb(model_id)
        else:
            out = export_dir(model_id, fmt)
            if not (out / "export_info.json").exists():
                subprocess.run([sys.executable, str(LAB_ROOT / "scripts/export_checkpoint.py"), cfg["ptq_config"], fmt],
                               check=True)
            add_legacy_config_keys(out)
            path, disk_gb = str(out), json.loads((out / "export_info.json").read_text())["disk_gb"]
        common = dict(server=name, engine=engine, fmt=fmt, model=model_id, config=Path(args.config).name)

        log = LAB_DATA / "logs" / f"serve_{name}.log"
        cmd, env = server_cmd(engine, path, name, s, model_id, spec.get("patch_lm_head", False))
        cmd += spec.get("args", [])
        base_mem = gpu_used_gb()
        print(f"=== {name}: starting ({log.name})", flush=True)
        with log.open("w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            startup_s = wait_ready(proc, url, log)
            kv = re.findall(KV_TOKENS[engine], log.read_text())
            kv_tokens = int(kv[-1].replace(",", "")) if kv else None
            ceiling = decode_ceiling_tok_s(disk_gb * 1e9 - embed_bytes + kv_mid)
            if ("serve", 0) in todo:
                row = log_result(cfg["results"], **common, kind="serve", concurrency=None, startup_s=round(startup_s, 1),
                                 server_gpu_gb=round(gpu_used_gb() - base_mem, 1), disk_gb=disk_gb, kv_tokens=kv_tokens,
                                 max_seqs_at_max_len=kv_tokens / s["max_seq_len"] if kv_tokens else None,
                                 decode_ceiling_tok_s=round(ceiling, 1), mem_bw_gbs=A6000_MEM_BW_GBS)
                print(f"  ready in {row['startup_s']}s, {row['server_gpu_gb']} GB on GPU, KV tokens {kv_tokens}", flush=True)

            bench(url, name, path, {**b, "min_prompts": 4, "prompts_per_concurrency": 1}, 1, bench_dir)  # warm-up
            for c in b["concurrency"]:
                if ("bench", c) not in todo:
                    continue
                r = bench(url, name, path, b, c, bench_dir)
                tok_s_1 = 1000 / r["mean_tpot_ms"]
                log_result(cfg["results"], **common, kind="bench", concurrency=c, input_len=b["input_len"],
                           output_len=b["output_len"], decode_tok_s_per_user=round(tok_s_1, 1),
                           pct_of_ceiling=round(100 * tok_s_1 / ceiling, 1) if c == 1 else None, **r)
                print(f"  c={c:<3d} out {r['output_throughput']:7.1f} tok/s  TTFT {r['mean_ttft_ms']:7.1f} ms  "
                      f"TPOT {r['mean_tpot_ms']:6.2f} ms", flush=True)

            if ("accuracy", 0) in todo:
                t0 = time.perf_counter()
                metrics = api_eval_task(url, name, path, acc["task"], acc.get("num_fewshot"), acc["num_concurrent"])
                metric = "exact_match,flexible-extract"
                log_result(cfg["results"], **common, kind="accuracy", concurrency=None, task=acc["task"],
                           num_fewshot=acc.get("num_fewshot"), metric=metric, value=metrics[metric],
                           stderr=metrics.get(metric.replace(",", "_stderr,", 1)), metrics=metrics,
                           eval_seconds=round(time.perf_counter() - t0, 1))
                print(f"  {acc['task']} {metric} {metrics[metric]:.4f}", flush=True)
        except Exception as e:
            print(f"=== {name} FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            stop(proc)
            time.sleep(10)


if __name__ == "__main__":
    main()
