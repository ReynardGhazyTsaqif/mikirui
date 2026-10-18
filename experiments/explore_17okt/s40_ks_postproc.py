"""H27 (mentor ide 19): pascaproses yang bisa mengubah KS = urutan ANTAR kelompok. Nested: dipelajari di 4 fold OOF,
dinilai di fold ke-5. Basis = OOF E202 (bag 3 seed).

Varian:
  A. kalibrasi ulang logistik per segmen: logit' = a_s + b_s * logit(p), segmen = punya riwayat x tier (10 segmen)
  B. isotonik per segmen (segmen kecil -> isotonik global)
  C. geser per segmen yang memaksimalkan KS langsung (koordinat menurun pada grid geser logit, 2 putaran)
  D. seperti A dengan segmen punya riwayat saja / tier saja
Keluaran: mean-fold AUC dan KS per varian + bootstrap berpasangan berstrata per fold (1000x) vs basis.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import apps, load
from resid import reference

ref = reference("E202")
y, fold = ref.target.values, ref.fold.values
lg = ref.lg.values
A = apps()
tr = A[A.is_test == 0].reset_index(drop=True)
pa_c = set(load("previous_applications", usecols=["customer_id"]).customer_id)
hist = tr.customer_id.isin(pa_c).values.astype(int)
tier = np.searchsorted(np.sort(tr.annual_interest_rate.unique()), tr.annual_interest_rate.values)
SEGS = {"hist_x_tier": hist * 10 + tier, "hist": hist, "tier": tier}


def ks(s, yy):
    o = np.argsort(-s, kind="stable")
    yy = yy[o]
    return np.max(np.cumsum(yy) / yy.sum() - np.cumsum(1 - yy) / (1 - yy).sum())


def logistic(seg):
    out = np.zeros(len(y))
    for k in range(5):
        trm, va = fold != k, fold == k
        for s in np.unique(seg):
            mt, mv = trm & (seg == s), va & (seg == s)
            if mv.sum() == 0:
                continue
            lr = LogisticRegression(C=1e4, max_iter=1000).fit(lg[mt].reshape(-1, 1), y[mt])
            out[mv] = lr.decision_function(lg[mv].reshape(-1, 1))
    return out


def isotonic(seg):
    out = np.zeros(len(y))
    for k in range(5):
        trm, va = fold != k, fold == k
        for s in np.unique(seg):
            mt, mv = trm & (seg == s), va & (seg == s)
            if mv.sum() == 0:
                continue
            if mt.sum() < 2000:
                mt = trm
            iso = IsotonicRegression(out_of_bounds="clip").fit(lg[mt], y[mt])
            out[mv] = iso.predict(lg[mv]) + 1e-9 * lg[mv]  # pemecah seri: urutan dalam segmen tetap
    return out


def ks_shift(seg, grid=np.linspace(-0.6, 0.6, 25), rounds=2):
    out = np.zeros(len(y))
    segs = np.unique(seg)
    for k in range(5):
        trm, va = fold != k, fold == k
        sh = {s: 0.0 for s in segs}
        for _ in range(rounds):
            for s in segs:
                best, bv = sh[s], -1
                for g in grid:
                    sh[s] = g
                    v = np.mean([ks(lg[trm & (fold == j)] + np.vectorize(sh.get)(seg[trm & (fold == j)]),
                                    y[trm & (fold == j)]) for j in range(5) if j != k])
                    if v > bv:
                        best, bv = g, v
                sh[s] = best
        out[va] = lg[va] + np.vectorize(sh.get)(seg[va])
    return out


def per_fold(s):
    return (np.array([roc_auc_score(y[fold == k], s[fold == k]) for k in range(5)]),
            np.array([ks(s[fold == k], y[fold == k]) for k in range(5)]))


def boot(s, metric="ks", n=1000):
    rng = np.random.default_rng(42)
    idx = [np.where(fold == k)[0] for k in range(5)]
    d = []
    for _ in range(n):
        dk = []
        for ii in idx:
            b = rng.choice(ii, len(ii))
            if metric == "ks":
                dk.append(ks(s[b], y[b]) - ks(lg[b], y[b]))
            else:
                dk.append(roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], lg[b]))
        d.append(np.mean(dk))
    d = np.array(d)
    return np.mean(d > 0), np.percentile(d, [2.5, 97.5])


a0, k0 = per_fold(lg)
print(f"basis E202: mean-fold AUC {a0.mean():.5f}, KS {k0.mean():.4f}")
V = {"A_logistik_hist_x_tier": logistic(SEGS["hist_x_tier"]), "D_logistik_hist": logistic(SEGS["hist"]),
     "D_logistik_tier": logistic(SEGS["tier"]), "B_isotonik_hist_x_tier": isotonic(SEGS["hist_x_tier"]),
     "C_geser_KS_hist": ks_shift(SEGS["hist"]), "C_geser_KS_tier": ks_shift(SEGS["tier"])}
for name, s in V.items():
    a, kk = per_fold(s)
    pk, cik = boot(s, "ks", 300)
    print(f"{name:<24} AUC {a.mean():.5f} ({a.mean() - a0.mean():+.5f}) | KS {kk.mean():.4f} ({kk.mean() - k0.mean():+.4f}, "
          f"naik {(kk > k0).sum()}/5, P(KS naik) {pk:.2f}, CI [{cik[0]:+.4f}; {cik[1]:+.4f}])", flush=True)
