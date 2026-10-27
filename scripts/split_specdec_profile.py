"""Where one EAGLE-3 step goes: draft head, target verification, GPU idle. From a profile_decode.py --draft capture.

    python scripts/split_specdec_profile.py $LAB_DATA/profiles/int4_lm_head_eagle3_int4_k4.nsys-rep --name int4_lm_head_eagle3_int4_k4

"""

import argparse
import itertools
import os
import sqlite3
import statistics
import subprocess

import lab  # noqa: F401
from lab.results import log_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    db_path = args.report.removesuffix(".nsys-rep") + ".sqlite"
    if not os.path.exists(db_path):
        nsys = os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nsys")
        subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db_path, args.report],
                       check=True, capture_output=True)
    rows = sqlite3.connect(db_path).execute(
        """select k.start, k."end", s.value from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on s.id = coalesce(k.shortName, k.demangledName) order by k.start""").fetchall()
    fc = [i for i, r in enumerate(rows) if "bf16_s16816gemm" in r[2]]

    busy = lambda rs: sum(e - s for s, e, _ in rs)
    parts = {"step": [], "draft": [], "draft_fc": [], "target": [], "idle": []}
    for a, b in itertools.pairwise(fc[5:-1]):
        _, t0 = max((rows[i][0] - rows[i - 1][1], i) for i in range(a + 1, b))
        step = rows[b][0] - rows[a][0]
        draft, target = busy(rows[a:t0]), busy(rows[t0:b])
        for k, v in zip(parts, (step, draft, rows[a][1] - rows[a][0], target, step - draft - target)):
            parts[k].append(v)
    ms = {k: round(statistics.median(v) / 1e6, 3) for k, v in parts.items()}
    log_result("profile_qwen3_8b", server=args.name, kind="specdec_split", steps=len(parts["step"]), ms_per_step=ms)
    print(f"{args.name}: {ms['step']:.2f} ms per step over {len(parts['step'])} steps")
    for k in ("target", "draft", "idle"):
        print(f"  {k:7s} {ms[k]:6.2f} ms  {100 * ms[k] / ms['step']:5.1f}%")
    print(f"  (draft includes the BF16 fc: {ms['draft_fc']:.2f} ms)")


if __name__ == "__main__":
    main()
