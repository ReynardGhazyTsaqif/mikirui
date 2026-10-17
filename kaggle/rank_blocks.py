"""Normalisasi persentil per blok untuk fitur nested (E122r).

Fitur bagian latih tr{k} berasal dari 4 model dalam (satu per fold j != k) yang skala logit-nya bisa bergeser,
sedangkan va{k} dan te{k} berasal dari satu model luar. Tiap blok yang diprediksi oleh model yang sama
(baris fold j di tr{k}, seluruh va{k}, seluruh te{k}) diubah ke persentil di dalam bloknya. Blok-blok itu sampel
acak dari populasi yang sama, jadi persentil menyamakan skala tanpa memakai label.

  python kaggle/rank_blocks.py features/e122_gru_v1.npz features/e122_gru_v1r.npz
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def pct(a):
    out = np.full(a.shape, np.nan, dtype="float32")
    for c in range(a.shape[1]):
        m = ~np.isnan(a[:, c])
        out[m, c] = (rankdata(a[m, c]) / m.sum()).astype("float32")
    return out


def main(src, dst):
    meta = pd.read_parquet(ROOT / "kaggle" / "dataset" / "apps_meta.parquet")
    fold = meta.fold.values[meta.is_test.values == 0]
    z = dict(np.load(src, allow_pickle=True))
    for k in range(5):
        fj = fold[fold != k]  # urutan baris tr{k} = baris train dengan fold != k
        tr = z[f"tr{k}"].astype("float64")
        out = np.empty(tr.shape, dtype="float32")
        for j in np.unique(fj):
            out[fj == j] = pct(tr[fj == j])
        z[f"tr{k}"] = out
        z[f"va{k}"] = pct(z[f"va{k}"].astype("float64"))
        z[f"te{k}"] = pct(z[f"te{k}"].astype("float64"))
    np.savez(dst, **z)
    print(dst, "kolom", list(z["cols"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
