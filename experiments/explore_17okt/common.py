"""Utilitas bersama script eksplorasi 17 Okt: baca data panitia dan OOF dari checkout utama (hanya baca).

Semua script di folder ini menulis keluaran ke folder ini sendiri (worktree), tidak ke checkout utama.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(os.environ.get("DSC_MAIN", r"D:/Unand/Sems 7/Lomba/DSC/dsc-logika"))
HERE = Path(__file__).resolve().parent
DATA = MAIN / "Dataset"
sys.path.insert(0, str(MAIN))  # src.* dari checkout utama (src/features tidak ada di git)

DATES = {
    "train_applications": ["application_datetime", "decision_datetime"],
    "test_applications": ["application_datetime", "decision_datetime"],
    "previous_applications": ["application_datetime", "decision_datetime"],
    "repayment_history": ["due_date", "last_payment_datetime"],
    "bureau_observations": ["report_datetime", "available_datetime"],
    "account_behavior": ["month"],
    "ecommerce_transactions": ["transaction_datetime"],
}


def load(name, usecols=None):
    dates = [c for c in DATES.get(name, []) if usecols is None or c in usecols]
    return pd.read_csv(DATA / f"{name}.csv", usecols=usecols, parse_dates=dates)


def apps():
    """Train + test (urutan load_apps: train dulu lalu test) + kolom fold (-1 = test)."""
    tr = load("train_applications")
    te = load("test_applications")
    tr["is_test"], te["is_test"] = 0, 1
    a = pd.concat([tr, te], ignore_index=True)
    f = pd.read_csv(MAIN / "folds" / "folds.csv")
    a = a.merge(f, on="application_id", how="left")
    a["fold"] = a["fold"].fillna(-1).astype(int)
    return a


def bag_oof(name, kind="oof"):
    fs = sorted(p for p in glob.glob(str(MAIN / kind / f"{name}_s*.npy"))
                if Path(p).stem[len(name) + 2:].isdigit())
    return np.mean([np.load(p) for p in fs], axis=0), len(fs)


def out(name):
    return HERE / name
