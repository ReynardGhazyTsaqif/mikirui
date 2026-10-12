"""Seberapa besar selisih skor public antar-model yang murni karena noise subset public?
Simulasi: ambil subset acak OOF (ukuran = porsi public dari 30k test), hitung KS tiap model,
lalu sd KS tunggal dan sd selisih KS antar pasangan model."""
import sys
sys.path.insert(0, r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika")
import numpy as np, pandas as pd
from src.train import ks_score
from src.io_utils import load
R = r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
y = load("train_applications", usecols=["target"]).target.values
m = {n: np.load(fr"{R}\oof\{n}.npy") for n in ["T_lgb_b3_4f3c7f", "T_cat_b3_a6abab", "E71b", "E82"]}
rng = np.random.default_rng(0)
pairs = [("T_lgb_b3_4f3c7f", "T_cat_b3_a6abab"), ("T_lgb_b3_4f3c7f", "E71b"), ("T_lgb_b3_4f3c7f", "E82")]
for size in [9000, 15000]:
    ks = {n: [] for n in m}
    for _ in range(1000):
        idx = rng.choice(len(y), size, replace=False)
        for n, p in m.items():
            ks[n].append(ks_score(y[idx], p[idx]))
    ks = {n: np.array(v) for n, v in ks.items()}
    print(f"subset {size}: sd KS tunggal T_lgb {ks['T_lgb_b3_4f3c7f'].std():.4f}")
    for a, b in pairs:
        d = ks[a] - ks[b]
        print(f"   selisih {a} - {b}: rata-rata {d.mean():+.4f}, sd {d.std():.4f}, korelasi pred {np.corrcoef(m[a], m[b])[0, 1]:.3f}")
