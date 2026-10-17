"""Langkah 2s: kelayakan 'pinjaman lama sebagai baris latih'. Apakah bukti customer TANPA pinjaman itu sendiri
memprediksi default pinjaman lama sama baiknya dengan memprediksi target sekarang?

Bukti sederhana (tanpa target sekarang): hasil pinjaman lengkap lain (k, n), laju lewat-bayar event pinjaman lain,
skor biro rata-rata semua laporan, tier pinjaman itu sendiri, tenor. Model: regresi logistik, CV dikelompokkan per customer.
Bandingkan AUC untuk (a) pinjaman lama lengkap tenor 12, (b) target sekarang (customer train punya riwayat).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import apps, load

A = apps()
EV = pd.read_parquet("cache_events_full.parquet")
PA = load("previous_applications").dropna(subset=["loan_id"]).set_index("loan_id")
L = EV.groupby("loan_id").agg(customer_id=("customer_id", "first"), dis=("dis", "first"), cutoff=("cutoff", "first"),
                              term=("term", "first"))
L["lab"] = (EV[EV.j <= 9].groupby("loan_id").A.max().reindex(L.index) >= 3).astype(int)
L["complete"] = (L.dis + pd.Timedelta(days=365)) <= L.cutoff
L["rate"] = PA.annual_interest_rate.reindex(L.index)
ev_loan = EV.groupby("loan_id").agg(n_ev=("miss", "size"), n_miss=("miss", "sum"))
L = L.join(ev_loan)
cust = L.groupby("customer_id").agg(K=("lab", lambda s: s[L.loc[s.index, "complete"]].sum()),
                                    N=("complete", "sum"), EVN=("n_ev", "sum"), EVM=("n_miss", "sum"))
BU = load("bureau_observations")
score = BU.groupby("customer_id").score.mean()


def design(df):
    X = pd.DataFrame(index=df.index)
    X["k"], X["n"] = df.k, df.n
    X["post"] = (df.k + 0.186) / (df.n + 1.516)
    X["miss_rate"] = (df.m + 0.95) / (df.e + 5.57)
    X["log_e"] = np.log1p(df.e)
    X["score"] = df.score.fillna(score.mean())
    X["rate"] = df.rate
    return X


# (a) pinjaman lama lengkap tenor 12: bukti dari pinjaman lain
a = L[L.complete & (L.term == 12)].copy()
c = cust.reindex(a.customer_id).values
a["k"] = c[:, 0] - a.lab.values
a["n"] = c[:, 1] - 1
a["e"] = c[:, 2] - a.n_ev.values
a["m"] = c[:, 3] - a.n_miss.values
a["score"] = a.customer_id.map(score)
# (b) target sekarang (train, punya riwayat)
tr = A[A.is_test == 0].set_index("customer_id")
b = pd.DataFrame(index=tr.index.intersection(cust.index))
b["lab"] = tr.target.reindex(b.index).astype(int)
cc = cust.reindex(b.index)
b["k"], b["n"], b["e"], b["m"] = cc.K, cc.N, cc.EVN, cc.EVM
b["score"] = score.reindex(b.index)
b["rate"] = tr.annual_interest_rate.reindex(b.index)
b["customer_id"] = b.index
for nm, d in [("pinjaman lama lengkap (tenor 12)", a), ("target sekarang (punya riwayat)", b)]:
    X, y, g = design(d), d.lab.values, d.customer_id.values
    oof = np.zeros(len(y))
    for trn, val in GroupKFold(5).split(X, y, g):
        mdl = LogisticRegression(max_iter=2000).fit(X.iloc[trn], y[trn])
        oof[val] = mdl.predict_proba(X.iloc[val])[:, 1]
    print(f"{nm}: n {len(y)}, rate {y.mean():.3f}, AUC bukti sederhana {roc_auc_score(y, oof):.4f}")
    # tanpa tier (tier = penilaian saat itu)
    X2 = X.drop(columns="rate")
    oof2 = np.zeros(len(y))
    for trn, val in GroupKFold(5).split(X2, y, g):
        oof2[val] = LogisticRegression(max_iter=2000).fit(X2.iloc[trn], y[trn]).predict_proba(X2.iloc[val])[:, 1]
    print(f"   tanpa tier: AUC {roc_auc_score(y, oof2):.4f}")
# silang: latih di pinjaman lama, nilai di target sekarang (dan sebaliknya)
Xa, Xb = design(a), design(b)
m_ab = LogisticRegression(max_iter=2000).fit(Xa, a.lab.values)
print("latih di pinjaman lama -> AUC target sekarang: %.4f" % roc_auc_score(b.lab.values, m_ab.predict_proba(Xb)[:, 1]))
m_ba = LogisticRegression(max_iter=2000).fit(Xb, b.lab.values)
print("latih di target sekarang -> AUC pinjaman lama: %.4f" % roc_auc_score(a.lab.values, m_ba.predict_proba(Xa)[:, 1]))
print("koefisien (pinjaman lama):", dict(zip(Xa.columns, np.round(m_ab.coef_[0], 3))))
print("koefisien (target sekarang):", dict(zip(Xb.columns, np.round(m_ba.coef_[0], 3))))
