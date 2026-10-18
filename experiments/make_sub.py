"""Pembuat submission generik: blend rank berbobot dari bag OOF/preds beberapa eksperimen (semua seed per eksperimen).

Tiap komponen = rank dari rata-rata rank antar seed (sama dengan make_sub_e141x.py / make_sub_e202.py). Mencetak
mean-fold AUC dan KS OOF blend, Spearman antar komponen, menulis submissions/<tag>_auc<cv>.csv (akhir baris LF,
round 6, urutan sample_submission) dan MD5. Validasi + log tetap lewat verify_submission.py --log.

  python experiments/make_sub.py --parts E205:0.5 E207:0.5 --tag E205_E207_rank50
"""
import argparse
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
meta = pd.read_parquet(ROOT / "kaggle" / "dataset" / "apps_meta.parquet")
y = meta[meta.is_test == 0].target.values
fold = meta[meta.is_test == 0].fold.values
te = meta[meta.is_test == 1].application_id.values


def bag(d, n):
    fs = sorted(f for f in glob.glob(str(ROOT / d / f"{n}_s*.npy")) if Path(f).stem[len(n) + 2:].isdigit())
    assert len(fs) >= 3, f"{d}/{n}: {len(fs)} seed"
    return rankdata(np.mean([rankdata(np.load(f)) for f in fs], 0)), len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="NAMA:BOBOT")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    parts = [(p.split(":")[0], float(p.split(":")[1])) for p in a.parts]
    W = sum(w for _, w in parts)
    O, T = 0, 0
    comps = []
    for n, w in parts:
        o, ns = bag("oof", n)
        t, nt = bag("preds", n)
        assert ns == nt, f"{n}: oof {ns} seed, preds {nt} seed"
        comps.append((n, o, t))
        O, T = O + w / W * o / len(o), T + w / W * t / len(t)
        r = [roc_auc_score(y[fold == k], o[fold == k]) for k in range(5)]
        print(f"komponen {n} ({ns} seed, bobot {w / W:.2f}): mean-fold {np.mean(r):.5f}")
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            print(f"Spearman {comps[i][0]}~{comps[j][0]}: OOF {spearmanr(comps[i][1], comps[j][1])[0]:.4f} "
                  f"test {spearmanr(comps[i][2], comps[j][2])[0]:.4f}")
    r = np.array([roc_auc_score(y[fold == k], O[fold == k]) for k in range(5)])
    ks = ks_2samp(O[y == 1], O[y == 0]).statistic
    t = (rankdata(T) / len(T)).round(6)
    ss = pd.read_csv(ROOT / "Dataset" / "sample_submission.csv")
    sub = ss[["application_id"]].merge(pd.DataFrame({"application_id": te, "default_probability": t}),
                                       on="application_id", how="left")
    assert len(sub) == 30000 and sub.default_probability.notna().all()
    fn = ROOT / "submissions" / f"{a.tag}_auc{r.mean():.5f}.csv"
    sub.to_csv(fn, index=False, lineterminator="\n")
    print(f"{fn.relative_to(ROOT)} | mean-fold {r.mean():.5f} {np.round(r, 4)} KS OOF {ks:.4f} | "
          f"md5 {hashlib.md5(fn.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
