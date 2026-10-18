"""Seberapa besar noise selisih KS di subset seukuran public (9.000 baris) dan private (21.000 baris)?

Memakai OOF bag 3 seed (rank rata-rata rank per seed, sama seperti pembuat submission) untuk model yang sudah punya
skor public. Untuk 2000 subset acak: KS tiap model, selisih berpasangan terhadap rekor public (E122x1 + E141).
Selisih public yang teramati dibandingkan dengan sebaran simulasi (z = (teramati - rata-rata) / sd).
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import MAIN, apps

A = apps()
tr = A[A.is_test == 0].reset_index(drop=True)
y = tr.target.values.astype(int)


def bag(n):
    fs = sorted(glob.glob(str(MAIN / "oof" / f"{n}_s4*.npy")))
    assert len(fs) == 3, n
    return rankdata(np.mean([rankdata(np.load(f)) for f in fs], 0))


def ks(p, yy):
    o = np.argsort(-p, kind="stable")
    yy = yy[o]
    tpr = np.cumsum(yy) / yy.sum()
    fpr = np.cumsum(1 - yy) / (1 - yy).sum()
    return np.max(tpr - fpr)


G1, D0, G2, Gc, Ga = bag("E122x1"), bag("E141"), bag("E202"), bag("E122c"), bag("E130a")
M = {"E122x1+E141": (0.5 * G1 + 0.5 * D0, 0.55951), "E202": (G2, 0.55754), "E202+E141": (0.5 * G2 + 0.5 * D0, 0.55852),
     "E130a+E141": (0.5 * Ga + 0.5 * D0, 0.55858), "E122c": (Gc, 0.55732)}
ref = "E122x1+E141"
print("KS OOF penuh:", {k: round(ks(v[0], y), 4) for k, v in M.items()})
rng = np.random.default_rng(42)
for size in (9000, 21000):
    R = {k: [] for k in M}
    for _ in range(2000):
        idx = rng.choice(len(y), size, replace=False)
        for k, (p, _) in M.items():
            R[k].append(ks(p[idx], y[idx]))
    R = pd.DataFrame(R)
    print(f"\nsubset {size} baris: sd KS per model {R.std().round(4).to_dict()}")
    for k in M:
        if k == ref:
            continue
        d = R[k] - R[ref]
        line = f"  {k} - {ref}: rata-rata {d.mean():+.4f}, sd {d.std():.4f}, P(>0) {(d > 0).mean():.2f}"
        if size == 9000:
            obs = M[k][1] - M[ref][1]
            line += f" | public teramati {obs:+.4f}, z {(obs - d.mean()) / d.std():+.2f}"
        print(line)
