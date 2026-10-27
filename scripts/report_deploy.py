"""Speed / memory / real-kernel accuracy tables and plots from the deployment benchmarks.

    python scripts/report_deploy.py configs/deploy_qwen3_8b.yaml

Writes report/<results>.md and report/figures/<results>_{throughput,batch1}.png.
"""

import argparse

import lab  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from lab.paths import FIGURES_DIR, LAB_ROOT
from lab.quant import calib_id
from lab.results import load_results

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED, INK = "#8a8984", "#52514e"


def style(ax, title):
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    df = load_results(cfg["results"])
    if df.empty:
        raise SystemExit(f"results/{cfg['results']}.jsonl is empty: run scripts/bench_deploy.py first")
    df = df.drop_duplicates(["server", "kind", "concurrency"], keep="last")
    order = [s for s in cfg["servers"] if s in set(df.server)]
    color = {s: COLORS[i % len(COLORS)] for i, s in enumerate(cfg["servers"])}
    conc = cfg["bench"]["concurrency"]

    serve = df[df.kind == "serve"].set_index("server")
    bench = df[df.kind == "bench"]
    table = serve[["engine", "fmt", "disk_gb", "server_gpu_gb", "kv_tokens", "startup_s", "decode_ceiling_tok_s"]].copy()
    b1 = bench[bench.concurrency == 1].set_index("server")
    table["c1_tok_s"] = b1.decode_tok_s_per_user
    table["c1_%ceiling"] = b1.pct_of_ceiling
    table["c1_ttft_ms"] = b1.mean_ttft_ms
    for c in conc[1:]:
        bc = bench[bench.concurrency == c].set_index("server")
        table[f"c{c}_out_tok_s"] = bc.output_throughput
        table[f"c{c}_tpot_ms"] = bc.mean_tpot_ms
        table[f"c{c}_ttft_ms"] = bc.mean_ttft_ms
    table = table.reindex([s for s in order if s in table.index])

    # Real-kernel accuracy vs the fake-quant number for the same format (default calibration)
    acc = df[df.kind == "accuracy"].set_index("server")[["fmt", "value", "stderr"]].rename(columns={"value": "server"})
    ptq = yaml.safe_load(open(LAB_ROOT / cfg["ptq_config"]))
    study = load_results(ptq["results"])
    if not study.empty and not acc.empty:
        sim = study[(study.task == cfg["accuracy"]["task"]) & (study.calib.isna() | (study.calib == calib_id(ptq["calib"])))]
        sim = sim.drop_duplicates("fmt", keep="last").set_index("fmt").value
        acc["fake_quant"] = acc.fmt.map(sim)
        acc[["server", "fake_quant", "stderr"]] *= 100
        acc["diff"] = acc.server - acc.fake_quant
    acc = acc.reindex([s for s in order if s in acc.index])

    lines = [f"# Deployment benchmarks: {cfg['model']}", "",
             f"Workload: {cfg['bench']['input_len']} input / {cfg['bench']['output_len']} output tokens (random, "
             "`--ignore-eos`), `vllm bench serve` against `/v1/completions` for every engine. `c1_tok_s` = 1000 / TPOT at "
             "concurrency 1; ceiling = 768 GB/s / (weights − embedding table + KV cache at mid-sequence).", "",
             table.round(1).to_markdown(), "",
             f"## Real-kernel accuracy ({cfg['accuracy']['task']}, through the server)", "",
             "`fake_quant` = the same format in the PTQ study (HF, simulated). Differences within ~2 stderr are noise.", "",
             acc.round(2).to_markdown(), ""]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in order:
        d = bench[bench.server == s].sort_values("concurrency")
        ax.plot(d.concurrency, d.output_throughput, marker="o", ms=8, lw=2, color=color[s], label=s)
    ax.set_xscale("log", base=2)
    ax.set_xticks(conc, [str(c) for c in conc])
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("output tokens/s (all users)")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Serving throughput")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{cfg['results']}_throughput.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = table.dropna(subset=["c1_tok_s"])
    x = range(len(t))
    ax.bar(x, t.c1_tok_s, color=[color[s] for s in t.index], width=0.6)
    ax.scatter(x, t.decode_ceiling_tok_s, marker="_", s=900, color=INK, linewidth=2, zorder=3, label="bandwidth ceiling")
    for i, (s, r) in enumerate(t.iterrows()):
        ax.annotate(f"{r.c1_tok_s:.0f} ({r['c1_%ceiling']:.0f}%)", (i, r.c1_tok_s), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x), t.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("decode tokens/s, batch 1")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax, "Batch-1 decode vs memory-bandwidth ceiling")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{cfg['results']}_batch1.png", dpi=150)

    prof = load_results("profile_qwen3_8b")
    if not prof.empty:
        # The deployed servers' plain decode profiles (make profile); speculative decoding ones go to report_specdec.py
        prof = prof[prof.server.str.startswith("trtllm_")].drop_duplicates("server", keep="last").set_index("server")
        cats = pd.DataFrame(prof.ms_per_step_by_category.tolist(), index=prof.index).fillna(0)
        cats.insert(0, "gpu_ms_per_step", prof.gpu_ms_per_step)
        lines += ["## Where a batch-1 decode step goes (Nsight Systems, ms per token)", "",
                  "TensorRT-LLM, 64 tokens after warm-up. In the INT4 run, `gemm_bf16` is the BF16 `lm_head` (one GEMV per "
                  "step); `elementwise` is AWQ's `x * pre_quant_scale` before `o_proj` / `down_proj`.", "",
                  cats.round(2).to_markdown(), ""]

    lines += [f"![throughput](figures/{cfg['results']}_throughput.png)", "",
              f"![batch-1](figures/{cfg['results']}_batch1.png)", ""]
    out = LAB_ROOT / "report" / f"{cfg['results']}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote report/{out.name}")


if __name__ == "__main__":
    main()
