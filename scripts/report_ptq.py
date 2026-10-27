"""Tables and the accuracy-vs-size plot from a PTQ study's results file.

    python scripts/report_ptq.py configs/ptq_qwen3_8b.yaml [--limit 20]

Writes report/<results>.md and report/figures/<results>_accuracy_size.png. Numbers come only from results/*.jsonl.
"""

import argparse

import lab  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from lab.paths import FIGURES_DIR, LAB_ROOT
from lab.quant import calib_id
from lab.results import load_results

REAL, SIMULATED, MUTED, INK = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
MARKERS = {16: "o", 8: "s", 4: "^"}


def fmt_table(df: pd.DataFrame, index, tasks: list[str]) -> pd.DataFrame:
    t = df.pivot_table(index=index, columns="task", values="value", aggfunc="last")
    t = t[[c for c in tasks if c in t]]
    acc = [c for c in t if c != "wikitext"]
    t[acc] = t[acc] * 100
    t["avg"] = t[acc].mean(axis=1).where(t[acc].notna().all(axis=1))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, help="report a --limit smoke run")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    name = cfg["results"] + (f"_limit{args.limit}" if args.limit else "")
    df = load_results(name)
    if df.empty:
        raise SystemExit(f"results/{name}.jsonl is empty: run scripts/ptq_study.py first")
    df = df.drop_duplicates("run_key", keep="last")
    tasks = [t["task"] for t in cfg["tasks"]]
    order = {f: i for i, f in enumerate(cfg["formats"])}
    meta_cols = ["weight_gb", "bits_per_param", "act_bits", "kv_bits", "deployable_on_ampere"]

    default = calib_id(cfg["calib"])
    main_df = df[df.calib.isna() | (df.calib == default)]
    table = fmt_table(main_df, "fmt", tasks)
    meta = main_df.groupby("fmt")[meta_cols].last()
    table = table.join(meta).sort_index(key=lambda s: s.map(lambda f: order.get(f, len(order))))
    if "bf16" in table.index:
        table.insert(table.columns.get_loc("avg") + 1, "avg_vs_bf16", table.avg - table.loc["bf16", "avg"])
    # The study flags formats by their weights only; FP8 / NVFP4 KV caches need sm_89+ too
    table["deployable_on_ampere"] = (table.deployable_on_ampere & (table.kv_bits >= 16)).map({True: "yes", False: "simulated"})

    lines = [f"# PTQ study: {cfg['model']}", "",
             f"Calibration: `{default}`. Accuracy in %, WikiText as word perplexity (lower is better). "
             "`avg` = mean accuracy over the lm-eval tasks, `avg_vs_bf16` = difference from BF16 in points.", "",
             table.round(2).to_markdown(), ""]

    ablation = df[df.calib.notna() & df.fmt.isin(df[df.calib.notna() & (df.calib != default)].fmt.unique())]
    if not ablation.empty:
        abl = fmt_table(ablation, ["fmt", "calib"], tasks)
        lines += ["## Calibration ablation", "", abl.round(2).to_markdown(), ""]

    pts = table.dropna(subset=["avg"])
    if len(pts) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for fmt, r in pts.iterrows():
            ax.scatter(r.weight_gb, r.avg, s=70, zorder=2, marker=MARKERS.get(int(r.act_bits), "D"),
                       color=REAL if r.deployable_on_ampere == "yes" else SIMULATED, edgecolor="white", linewidth=1.5)
        fig.canvas.draw()
        placed = []  # label boxes in pixels; push a label down until it clears the ones already placed
        for fmt, r in pts.sort_values("avg", ascending=False).iterrows():
            for dy in range(4, -200, -9):
                moved = dy != 4
                xy = dict(xytext=(28 if moved else 6, dy), textcoords="offset points", fontsize=8, color=INK,
                          va="center" if moved else "baseline")
                label = ax.annotate(fmt, (r.weight_gb, r.avg), **xy)
                box = label.get_window_extent(fig.canvas.get_renderer()).expanded(1.0, 1.15)
                label.remove()
                if not any(box.overlaps(b) for b in placed):
                    break
            placed.append(box)
            # A displaced label gets a leader line back to its own point
            ax.annotate(fmt, (r.weight_gb, r.avg), **xy,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6) if moved else None)
        handles = [plt.Line2D([], [], ls="", marker="o", color=REAL, label="format runs on Ampere"),
                   plt.Line2D([], [], ls="", marker="o", color=SIMULATED, label="simulated (needs sm_89+)")]
        handles += [plt.Line2D([], [], ls="", marker=m, color=INK, label=f"A{b}") for b, m in MARKERS.items()
                    if b in set(pts.act_bits.astype(int))]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
        ax.set_xlabel("model size (GB, quantized weights + scales)")
        ax.set_ylabel("mean accuracy (%)")
        ax.set_title(f"{cfg['model']}: accuracy vs size", loc="left")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        png = FIGURES_DIR / f"{name}_accuracy_size.png"
        fig.savefig(png, dpi=150)
        lines += [f"![accuracy vs size]({png.relative_to(LAB_ROOT / 'report')})", ""]

    out = LAB_ROOT / "report" / f"{name}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote report/{out.name}")


if __name__ == "__main__":
    main()
