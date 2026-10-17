"""Masukkan output notebook Kaggle (E122/E142) ke pipeline lokal setelah mengecek urutan baris.

Taruh file output Kaggle di kaggle/output/ lalu jalankan:
  python kaggle/import_results.py
- e122_gru_v1[_varian].npz -> features/ (fitur nested, format e121.py)
- e122_gru_oof/test[_varian].npy -> oof|preds/E122_<varian>_s0.npy (model urutan tunggal, rata-rata seed)
- e122x_<v>_v1.npz / _oof|_test.npy -> features/, oof|preds/E122x_<v>_s0.npy
- e142_<m>_oof_s<s>.npy    -> oof/E142_<m>_s<s>.npy, preds/E142_<m>_s<s>.npy
Lanjutannya:
  python e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1.npz --basefs all --name E122
  python .claude/skills/run-experiment/scripts/compare_oof.py --base E130a --cand E142_realmlp --blend
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle" / "output"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    meta = pd.read_parquet(ROOT / "kaggle" / "dataset" / "apps_meta.parquet")
    ids_tr = meta.application_id.values[meta.is_test.values == 0]
    ids_te = meta.application_id.values[meta.is_test.values == 1]
    train_csv = pd.read_csv(ROOT / "Dataset" / "train_applications.csv", usecols=["application_id"])
    assert (train_csv.application_id.values == ids_tr).all(), "urutan train beda dengan Dataset/"

    ref = np.load(ROOT / "features" / "e121_ordermodel_v1.npz", allow_pickle=True)
    for p in sorted(SRC.glob("e122_gru_v1*.npz")):  # varian: _bigru, _tf, _shuffle
        z = np.load(p, allow_pickle=True)
        assert (z["ids_train"] == ids_tr).all() and (z["ids_test"] == ids_te).all()
        for k in range(5):
            for part in ("tr", "va", "te"):
                assert len(z[f"{part}{k}"]) == len(ref[f"{part}{k}"]), f"{p.name} {part}{k} panjang beda"
        shutil.copy2(p, ROOT / "features" / p.name)
        print(f"{p.name} -> features/")
    for q in sorted(SRC.glob("e122_gru_*.npy")):
        m = re.fullmatch(r"e122_gru_(oof|test)(_\w+)?", q.stem)
        if not m:
            continue
        dst = "oof" if m[1] == "oof" else "preds"
        name = f"E122{m[2] or '_gru'}_s0.npy"
        shutil.copy2(q, ROOT / dst / name)
        print(f"{q.name} -> {dst}/{name}")
    for p in sorted(SRC.glob("e122x_*_v1.npz")):  # E122x: cicilan/rekening/pinjaman lama/semua
        z = np.load(p, allow_pickle=True)
        assert (z["ids_train"] == ids_tr).all() and (z["ids_test"] == ids_te).all()
        for k in range(5):
            for part in ("tr", "va", "te"):
                assert len(z[f"{part}{k}"]) == len(ref[f"{part}{k}"]), f"{p.name} {part}{k} panjang beda"
                assert not np.isnan(z[f"{part}{k}"]).any(), f"{p.name} {part}{k} berisi NaN"
        shutil.copy2(p, ROOT / "features" / p.name)
        print(f"{p.name} -> features/")
    for q in sorted(SRC.glob("e122x_*.npy")):
        m = re.fullmatch(r"e122x_(\w+?)_(oof|test)", q.stem)
        if not m:
            continue
        dst = "oof" if m[2] == "oof" else "preds"
        shutil.copy2(q, ROOT / dst / f"E122x_{m[1]}_s0.npy")
        print(f"{q.name} -> {dst}/E122x_{m[1]}_s0.npy")
    for q in sorted(SRC.glob("e142_*_*_s*.npy")):
        m = re.fullmatch(r"e142_(\w+?)_(oof|test)_s(\d+)", q.stem)
        if not m:
            continue
        arr = np.load(q)
        assert len(arr) == (len(ids_tr) if m[2] == "oof" else len(ids_te))
        assert not np.isnan(arr).any(), f"{q.name} berisi NaN (mode SMOKE?)"
        dst = "oof" if m[2] == "oof" else "preds"
        shutil.copy2(q, ROOT / dst / f"E142_{m[1]}_s{m[3]}.npy")
        print(f"{q.name} -> {dst}/E142_{m[1]}_s{m[3]}.npy")


if __name__ == "__main__":
    main()
