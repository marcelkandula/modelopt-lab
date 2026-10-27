"""MT-Bench questions, EAGLE-3 training prompts and the public Qwen3-8B EAGLE-3 head, under $LAB_DATA.

    python scripts/prepare_data.py
"""

import json
import shutil
import subprocess
import sys

import lab  # noqa: F401
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from lab.paths import DATASETS_DIR, LAB_DATA, LAB_ROOT


def mt_bench():
    out = DATASETS_DIR / "mt_bench" / "question.jsonl"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download("HuggingFaceH4/mt_bench_prompts", "raw/question.jsonl", repo_type="dataset")
    with open(src) as f, open(out, "w") as g:
        for line in f:
            r = json.loads(line)
            g.write(json.dumps({"category": r["category"], "turns": r["prompt"]}) + "\n")


def eagle3_prompts():
    work = DATASETS_DIR / "eagle3"
    if (work / "prompts" / "daring_anteater.jsonl").exists():
        return
    work.mkdir(parents=True, exist_ok=True)
    script = LAB_ROOT / "third_party/Model-Optimizer/examples/dataset/make_dataset.py"
    seeded = ("import os, random, runpy, sys; random.seed(0); sys.argv = sys.argv[1:]; "
              "sys.path.insert(0, os.path.dirname(sys.argv[0])); runpy.run_path(sys.argv[0], run_name='__main__')")
    subprocess.run([sys.executable, "-c", seeded, str(script), "-c", str(LAB_ROOT / "configs/eagle3_data.yaml")],
                   cwd=work, check=True)


def public_head():
    out = LAB_DATA / "heads" / "tengyunw_qwen3_8b_eagle3"
    if (out / "model.safetensors").exists():
        return
    for name in ("config.json", "pytorch_model.bin"):
        hf_hub_download("Tengyunw/qwen3_8b_eagle3", name, local_dir=out)
    sd = torch.load(out / "pytorch_model.bin", map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, out / "model.safetensors", metadata={"format": "pt"})
    (out / "pytorch_model.bin").unlink()
    shutil.rmtree(out / ".cache", ignore_errors=True)


def main():
    mt_bench()
    eagle3_prompts()
    public_head()
    print(f"data ready under {LAB_DATA.name}/")


if __name__ == "__main__":
    main()
