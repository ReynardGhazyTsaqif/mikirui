"""PROMPT 3 Langkah 5: ensemble sesama GBDT dengan rank average, OOF rata-rata >= 3 seed.

Empat pilihan dibandingkan (mean-fold AUC dan KS): model tunggal terbaik, rata-rata sederhana, bobot nested (dicari
di 4 fold, dinilai di fold ke-5; Nelder-Mead pada bobot softmax), Caruana nested (seleksi maju dengan pengembalian,
20 langkah, di 4 fold). NN opsional (--nn) dengan bobot total dibatasi <= --nn-cap (default 0,2).
Bobot akhir untuk test = bobot yang dicari di 5 fold penuh. Blend OOF terbaik ditulis ke experiments/prompt3/ untuk
compare_oof (--min-seeds 1 karena tiap komponen sudah bag >= 3 seed).

  python experiments/prompt3/p3_ensemble.py --models E210 E212 E213 [--nn E142_tabm_seqx_iv] --tag gbdt
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
meta = pd.read_parquet(ROOT / "kaggle" / "dataset" / "apps_meta.parquet")
y = meta[meta.is_test == 0].target.values.astype(int)
fold = meta[meta.is_test == 0].fold.values


def bag(name, kind="oof"):
    fs = sorted(f for f in glob.glob(str(ROOT / kind / f"{name}_s*.npy")) if Path(f).stem[len(name) + 2:].isdigit())
    assert len(fs) >= 3 or kind == "preds", f"{name}: {len(fs)} seed"
    return np.mean([np.load(f) for f in fs], 0), len(fs)


def ks(s, yy):
    o = np.argsort(-s, kind="stable")
    yy = yy[o]
    return np.max(np.cumsum(yy) / yy.sum() - np.cumsum(1 - yy) / (1 - yy).sum())


def mf(s, folds=range(5)):
    return np.mean([roc_auc_score(y[fold == k], s[fold == k]) for k in folds])


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def fit_weights(R, folds, cap=None, nn_idx=()):
    def loss(v):
        w = softmax(v)
        if cap is not None and len(nn_idx) and w[list(nn_idx)].sum() > cap:
            return 1.0
        return -mf(R @ w, folds)
    best = None
    for start in (np.zeros(R.shape[1]), np.random.default_rng(42).normal(0, 0.5, R.shape[1])):
        r = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    return softmax(best.x)


def caruana(R, folds, steps=20):
    chosen = []
    for _ in range(steps):
        scores = [mf(R[:, chosen + [j]].mean(1), folds) for j in range(R.shape[1])]
        chosen.append(int(np.argmax(scores)))
    return np.bincount(chosen, minlength=R.shape[1]) / len(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--nn", nargs="*", default=[])
    ap.add_argument("--nn-cap", type=float, default=0.2)
    ap.add_argument("--tag", default="gbdt")
    a = ap.parse_args()
    names = a.models + a.nn
    O = [bag(n) for n in names]
    R = np.column_stack([rankdata(o) / len(o) for o, _ in O])
    nn_idx = tuple(range(len(a.models), len(names)))
    rows = []
    for j, n in enumerate(names):
        rows.append(dict(pilihan=f"tunggal {n} ({O[j][1]} seed)", auc=mf(R[:, j]), ks=ks(R[:, j], y)))
    gb = list(range(len(a.models)))
    simple = R[:, gb].mean(1)
    rows.append(dict(pilihan="rata-rata sederhana GBDT", auc=mf(simple), ks=ks(simple, y)))
    for label, use_nn in (("GBDT", False), (f"GBDT + NN (cap {a.nn_cap})", True)):
        if use_nn and not a.nn:
            continue
        cols = list(range(len(names))) if use_nn else gb
        Rs = R[:, cols]
        idx_nn = tuple(i for i, c in enumerate(cols) if c in nn_idx)
        nested_w, nested_c = np.zeros(len(y)), np.zeros(len(y))
        for k in range(5):
            others = [j for j in range(5) if j != k]
            w = fit_weights(Rs, others, a.nn_cap if use_nn else None, idx_nn)
            nested_w[fold == k] = Rs[fold == k] @ w
            c = caruana(Rs, others) if not use_nn else w
            nested_c[fold == k] = Rs[fold == k] @ c
        w_full = fit_weights(Rs, range(5), a.nn_cap if use_nn else None, idx_nn)
        rows.append(dict(pilihan=f"bobot nested {label}", auc=mf(nested_w), ks=ks(nested_w, y),
                         bobot_penuh=dict(zip([names[c] for c in cols], np.round(w_full, 3)))))
        if not use_nn:
            c_full = caruana(Rs, range(5))
            rows.append(dict(pilihan="Caruana nested GBDT", auc=mf(nested_c), ks=ks(nested_c, y),
                             bobot_penuh=dict(zip([names[c] for c in cols], np.round(c_full, 3)))))
            np.save(ROOT / "experiments" / "prompt3" / f"blend_{a.tag}_nestedw.npy", nested_w)
            np.save(ROOT / "experiments" / "prompt3" / f"blend_{a.tag}_simple.npy", simple)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)
    print(T.round(5).to_string(index=False))
    best_single = T[T.pilihan.str.startswith("tunggal")].auc.max()
    print(f"selisih terhadap tunggal terbaik: {(T.auc - best_single).round(5).tolist()} (aturan: < 0,0003 -> pilih paling sederhana)")


if __name__ == "__main__":
    main()
