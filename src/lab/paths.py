import os
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
LAB_DATA = Path(os.environ["LAB_DATA"])

RESULTS_DIR = LAB_ROOT / "results"
FIGURES_DIR = LAB_ROOT / "report" / "figures"
EXPORTS_DIR = LAB_DATA / "exports"
DATASETS_DIR = LAB_DATA / "datasets"

for _d in (RESULTS_DIR, FIGURES_DIR, EXPORTS_DIR, DATASETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

