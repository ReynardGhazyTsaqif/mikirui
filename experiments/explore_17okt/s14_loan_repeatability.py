"""Langkah 2m: seberapa 'acak' label default? Pinjaman lama berjendela lengkap sebagai pengukuran berulang.

Label pinjaman lama: A >= 3 pada event 1..9 (setara DPD >= 90 dalam 12 bulan). Hanya pinjaman yang
disbursement + 365 hari <= cutoff (jendela lengkap).
1. ICC beta-binomial antar pinjaman milik customer yang sama -> batas AUC jika pi customer diketahui sempurna.
2. P(target sekarang | porsi default pinjaman lama) vs P(default pinjaman j | porsi default pinjaman lain).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import apps

pd.set_option("display.width", 220)
A = apps()
EV = pd.read_parquet("cache_events_full.parquet")
L = EV.groupby("loan_id").agg(customer_id=("customer_id", "first"), dis=("dis", "first"), cutoff=("cutoff", "first"),
                              term=("term", "first"))
early = EV[EV.j <= 9].groupby("loan_id").A.max()
L["lab"] = (early.reindex(L.index) >= 3).astype(int)
L = L[(L.dis + pd.Timedelta(days=365)) <= L.cutoff]
print("pinjaman lama berjendela lengkap:", len(L), "default rate:", L.lab.mean().round(4), "per tenor:", L.groupby("term").lab.mean().round(3).to_dict())
c = L.groupby("customer_id").agg(n=("lab", "size"), k=("lab", "sum"))
m = c[c.n >= 2]
print("customer dengan >=2 pinjaman lengkap:", len(m), "rata n:", m.n.mean().round(2))
# ICC via pasangan
pairs = L.merge(L, on="customer_id")
pairs = pairs[pairs.index.isin(pairs.index)]
pairs = L.reset_index().merge(L.reset_index(), on="customer_id")
pairs = pairs[pairs.loan_id_x < pairs.loan_id_y]
p = L.lab.mean()
cov = ((pairs.lab_x - p) * (pairs.lab_y - p)).mean()
icc = cov / (p * (1 - p))
print(f"pasangan pinjaman: {len(pairs)}, korelasi label antar pinjaman customer sama (ICC) = {icc:.4f}")
print("P(default j | pinjaman lain default) = %.4f vs P(default j | lain tidak) = %.4f" % (
    pairs[pairs.lab_x == 1].lab_y.mean(), pairs[pairs.lab_x == 0].lab_y.mean()))
# batas AUC: pi ~ Beta(a,b) dengan mean p dan ICC -> a+b = 1/icc - 1
s = 1 / icc - 1
a, b = p * s, (1 - p) * s
rng = np.random.default_rng(42)
pi = rng.beta(a, b, 400000)
y = rng.random(400000) < pi
print(f"Beta(a={a:.3f}, b={b:.3f}) -> AUC maksimum jika pi diketahui: {roc_auc_score(y, pi):.4f}")
# target sekarang vs porsi default pinjaman lama
tr = A[A.is_test == 0].set_index("customer_id")
c["share"] = c.k / c.n
j = tr.join(c, how="inner")
print("P(target | jumlah pinjaman lama default k, n>=1):")
print(j.groupby([j.n.clip(upper=4), j.k.clip(upper=3)]).target.agg(["mean", "size"]).round(3).unstack().to_string())
# pembanding: perlakukan satu pinjaman lama (yang paling baru) sebagai 'target'
L2 = L.reset_index().sort_values("dis")
last = L2.groupby("customer_id").tail(1).set_index("customer_id")
others = L2[~L2.loan_id.isin(last.loan_id)].groupby("customer_id").agg(n=("lab", "size"), k=("lab", "sum"))
jj = last.join(others, how="inner")
print("P(default pinjaman lama terbaru | k default dari n pinjaman lama lain):")
print(jj.groupby([jj.n.clip(upper=4), jj.k.clip(upper=3)]).lab.agg(["mean", "size"]).round(3).unstack().to_string())
