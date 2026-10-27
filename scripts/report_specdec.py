"""Speculative decoding table and speed-vs-draft-length plot from results/<results>.jsonl.

    python scripts/report_specdec.py configs/specdec_qwen3_8b.yaml
"""

import argparse

import lab  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from lab.paths import FIGURES_DIR, LAB_ROOT
from lab.results import load_results

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SHORT = {"bf16": "BF16", "int4_awq_lm_head": "INT4"}
STYLES = ["-", "--"]
MUTED, INK = "#8a8984", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    df = load_results(cfg["results"]).drop_duplicates("run", keep="last")
    base = df[df.draft == "none"].set_index("target").output_tok_s
    ref = base.get("bf16")
    df["speedup_vs_bf16"] = df.output_tok_s / ref
    df["speedup_vs_target"] = df.output_tok_s / df.target.map(base)
    table = df.sort_values(["target", "draft", "draft_length"])[
        ["target", "draft", "draft_length", "acceptance_length", "step_ms", "output_tok_s", "speedup_vs_target",
         "speedup_vs_bf16"]]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    drafts = [d for d in cfg["drafts"] if d != "none"]
    targets = sorted(df.target.unique(), key=lambda t: -base.get(t, 0))
    for (target, draft), d in df[df.draft != "none"].groupby(["target", "draft"]):
        d = d.sort_values("draft_length")
        label = f"{SHORT.get(target, target)} target + {cfg['drafts'][draft].get('label', draft)}"
        ax.plot(d.draft_length, d.output_tok_s, marker="o", ms=7, lw=2, color=COLORS[drafts.index(draft) % len(COLORS)],
                ls=STYLES[targets.index(target) % len(STYLES)], label=label)
    for target, tok_s in base.items():
        ax.axhline(tok_s, color=MUTED, lw=1, ls="--")
        ax.annotate(f"{SHORT.get(target, target)} target, no draft", (df.draft_length.max(), tok_s), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8, color=INK)
    ax.set_xticks(sorted(df[df.draft != "none"].draft_length.unique()))
    ax.set_xlabel("draft length (tokens proposed per step)")
    ax.set_ylabel("output tokens/s, batch 1")
    ax.set_title(f"EAGLE-3 on {cfg['tokenizer']}: MT-Bench, TensorRT-LLM, RTX A6000", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = FIGURES_DIR / f"{cfg['results']}.png"
    fig.savefig(png, dpi=150)

    lines = [f"# Speculative decoding: {cfg['tokenizer']}", "",
             "MT-Bench (80 questions * 2 turns), batch 1, greedy, thinking off, TensorRT-LLM 1.0 via ModelOpt's "
             "specdec_bench. `acceptance_length` = tokens produced per target forward pass; `step_ms` = time per step "
             "(draft + verify).", "", table.round(2).to_markdown(index=False), "",
             f"![speed vs draft length](figures/{png.name})", ""]
    prof = load_results("profile_qwen3_8b")
    if not prof.empty and "kind" in prof:
        split = prof[prof.kind == "specdec_split"].drop_duplicates("server", keep="last")
        rows = [{"profile": r.server, **r.ms_per_step} for r in split.itertuples()]
        if rows:
            lines += ["## Where a speculative decoding step goes (Nsight Systems, ms per step)", "",
                      ("`scripts/split_specdec_profile.py`: medians over one 256-token generation; `target` = verifying "
                       "the draft, `draft` = all draft passes (incl. its BF16 `fc`), `idle` = GPU waiting on the host."), "",
                      pd.DataFrame(rows).round(2).to_markdown(index=False), ""]
    out = LAB_ROOT / "report" / f"{cfg['results']}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines[:-2]))
    print(f"wrote report/{out.name}")


if __name__ == "__main__":
    main()
