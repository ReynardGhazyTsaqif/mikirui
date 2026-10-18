"""Siapkan folder kaggle/dataset/ untuk diunggah sebagai Kaggle Dataset PRIVAT (nama: dsc-logika-exp).

Isi (semua turunan data panitia, tanpa data luar):
  apps_meta.parquet         application_id, customer_id, is_test, target, fold (urutan = load_apps: train lalu test)
  ec_orders_v1.parquet      tabel order e-commerce (fitur intrinsik order, cache e121.build_orders)
  X_all.parquet             matriks 528 kolom semua grup registry (urutan = apps_meta)
  e121_ordermodel_v1.npz    7 fitur ecm_* nested per fold (format e121.py)
  seq_rp|ab|pa_v1.parquet   tabel urutan E122x (dibuat terpisah: python kaggle/make_seq_tables.py)
  e122_gru_v1*r.npz         seq_gru/seq_bigru/seq_tf nested, persentil per blok (kaggle/rank_blocks.py)
  rp_events_v1.parquet      18 kolom rp_ev_* (E200), index application_id
  ab_nonin_v1.parquet       5 kolom ab_nonin_* (E204), index application_id

  python kaggle/make_dataset.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402
from src.config import FEATURES_DIR, ROOT  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load_apps  # noqa: E402
from src.train import make_folds  # noqa: E402

OUT = ROOT / "kaggle" / "dataset"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    apps = load_apps()
    m = apps["is_test"].eq(0).values
    meta = apps[["application_id", "customer_id", "is_test", "target"]].copy()
    meta["fold"] = -1
    meta.loc[m, "fold"] = make_folds(apps[m])["fold"].values
    meta.to_parquet(OUT / "apps_meta.parquet", index=False)
    X = run.build_matrix(list(GROUPS), apps)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat:  # kategori -> kode angka (NaN tetap NaN)
        X[c] = X[c].astype("category").cat.codes.astype("float32").replace(-1, np.nan)
    X = X.astype("float32")
    X.insert(0, "application_id", apps["application_id"].values)
    X.to_parquet(OUT / "X_all.parquet", index=False)
    for f in ("ec_orders_v1.parquet", "e121_ordermodel_v1.npz",
              "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",  # seq_* persentil (E122c)
              "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz",  # seqx_* persentil (E122x1)
              "rp_events_v1.parquet", "ab_nonin_v1.parquet"):  # rantai bayar event (E200), inflow non-gaji (E204)
        shutil.copy2(FEATURES_DIR / f, OUT / f)
    print(f"kolom kategori dikodekan: {cat}")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:28s} {p.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
