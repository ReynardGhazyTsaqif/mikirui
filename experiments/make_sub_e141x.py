"""Submission tes setelah E141x (DART pada set E122x1): blend rank 50:50 dengan LGB E122x1, dan DART saja.

  python experiments/make_sub_e141x.py
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
    assert fs, f"{d}/{n}_s4*.npy tidak ada"
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


G, D, D0 = bag("oof", "E122x1"), bag("oof", "E141x"), bag("oof", "E141")
Gt, Dt = bag("preds", "E122x1"), bag("preds", "E141x")
print(f"E122x1 {pf(G).mean():.5f} | E141 (set lama) {pf(D0).mean():.5f} | E141x {pf(D).mean():.5f} "
      f"({(pf(D) > pf(D0)).sum()}/5 vs E141) | Spearman E122x1~E141x OOF {spearmanr(G, D)[0]:.4f} test {spearmanr(Gt, Dt)[0]:.4f}")
print(f"E122x1+E141 (lama) rank50: {pf(0.5 * G + 0.5 * D0).mean():.5f}")
write(0.5 * G + 0.5 * D, 0.5 * Gt + 0.5 * Dt, "E122x1_E141x_rank50")
write(D, Dt, "E141x_single")
