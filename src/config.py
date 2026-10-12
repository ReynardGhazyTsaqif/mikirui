"""Konfigurasi global proyek. Semua anggota memakai nilai yang sama."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DSC_DATA_DIR", ROOT / "Dataset"))

SEED = 42
N_FOLDS = 5
N_THREADS = 4

FEATURES_DIR = ROOT / "features"
FOLDS_PATH = ROOT / "folds" / "folds.csv"
OOF_DIR = ROOT / "oof"
PREDS_DIR = ROOT / "preds"
LOG_PATH = ROOT / "experiments" / "log.csv"
SUBMISSIONS_DIR = ROOT / "submissions"

CAT_COLS = ["loan_purpose", "region", "employment_type", "housing_status"]

LGB_DEV = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_child_samples=100,
    feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, max_bin=255,
    seed=SEED, deterministic=True, force_row_wise=True,
    num_threads=N_THREADS, verbose=-1,
)
