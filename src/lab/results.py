import datetime as _dt
import functools
import json
import subprocess

from .paths import LAB_ROOT, RESULTS_DIR


@functools.cache
def _provenance():
    info = {}
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=LAB_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=LAB_ROOT, capture_output=True, text=True)
        info["git_dirty"] = bool(dirty.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = None
    import torch
    import transformers

    info["torch"] = torch.__version__
    info["transformers"] = transformers.__version__
    try:
        import modelopt

        info["modelopt"] = modelopt.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def log_result(file: str, **record) -> dict:
    row = {"time": _dt.datetime.now().isoformat(timespec="seconds"), **record, **_provenance()}
    with (RESULTS_DIR / f"{file}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_results(file: str):
    import pandas as pd

    path = RESULTS_DIR / f"{file}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)
