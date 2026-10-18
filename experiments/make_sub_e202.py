"""Submission 18 Okt: E202 (set E200 = E122x1 + rantai bayar event, 10-fold bersarang, bag 3 seed) dan, sebagai
diagnostik public, blend rank 50:50 dengan DART E141 (resep public terbaik lama; CV-nya lebih rendah).

  python experiments/make_sub_e202.py
"""
import glob
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
meta = pd.read_parquet("kaggle/dataset/apps_meta.parquet")
y = meta[meta.is_test == 0].target.values
fold = meta[meta.is_test == 0].fold.values
te = meta[meta.is_test == 1].application_id.values
ss = pd.read_csv("Dataset/sample_submission.csv")


def pf(p):
    return np.array([roc_auc_score(y[fold == k], p[fold == k]) for k in range(5)])


def ks(p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def bag(d, n):
    fs = sorted(glob.glob(f"{d}/{n}_s4*.npy"))
    assert len(fs) == 3, f"{d}/{n}_s4*.npy: {fs}"
    return rankdata(np.mean([rankdata(np.load(f)) for f in fs], 0))


def write(o, t, tag):
    r = pf(o)
    t = (rankdata(t) / len(t)).round(6)
    sub = ss[["application_id"]].merge(pd.DataFrame({"application_id": te, "default_probability": t}),
                                       on="application_id", how="left")
    assert len(sub) == 30000 and sub.default_probability.notna().all()
    fn = f"submissions/{tag}_auc{r.mean():.5f}.csv"
    sub.to_csv(fn, index=False, lineterminator="\n")
    print(f"{fn} | mean-fold {r.mean():.5f} {np.round(r, 4)} KS {ks(o):.4f} | md5 {hashlib.md5(open(fn, 'rb').read()).hexdigest()}")


G, D = bag("oof", "E202"), bag("oof", "E141")
Gt, Dt = bag("preds", "E202"), bag("preds", "E141")
B = bag("preds", "E122x1")
print(f"E202 {pf(G).mean():.5f} | E141 {pf(D).mean():.5f} | Spearman E202~E141 OOF {spearmanr(G, D)[0]:.4f} "
      f"test {spearmanr(Gt, Dt)[0]:.4f} | Spearman test E202~E122x1 {spearmanr(Gt, B)[0]:.4f}")
write(G, Gt, "E202_single")
write(0.5 * G + 0.5 * D, 0.5 * Gt + 0.5 * Dt, "E202_E141_rank50")
