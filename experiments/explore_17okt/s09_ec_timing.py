"""Langkah 2i: e-commerce sebagai proses titik. N order tidak terkait target, bentuk sebaran waktunya terkait.

Uji (cross-fit per fold, tanpa memakai label fold sendiri):
  f_y(tau) = densitas waktu order (hari sebelum keputusan, log-bin) untuk customer target y, dari fold lain.
  LLR order = log f1/f0; per customer: jumlah, rata-rata, jumlah 30 hari terakhir, dsb.
Juga: apakah atribut order lain (nominal, kategori, metode, status) bergeser menjelang keputusan khusus untuk target 1?
"""
import numpy as np
import pandas as pd

from resid import reference, table

pd.set_option("display.width", 220)
od = pd.read_parquet("cache_ec_orders.parquet")
ref = reference("E122x1")
fold = ref.set_index("customer_id").fold
od["fold"] = od.customer_id.map(fold).fillna(-1).astype(int)
od["tau"] = od.rec_h / 24
edges = np.unique(np.concatenate([np.arange(1, 8, 0.5), np.geomspace(8, 732, 40)]))
od["b"] = np.clip(np.searchsorted(edges, od.tau, side="right") - 1, 0, len(edges) - 2)
nb = len(edges) - 1
llr = np.zeros(len(od))
for k in range(5):
    src = od[(od.fold >= 0) & (od.fold != k)]
    c1 = np.bincount(src.b[src.target == 1], minlength=nb) + 1.0
    c0 = np.bincount(src.b[src.target == 0], minlength=nb) + 1.0
    l = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
    m = od.fold == k
    llr[m.values] = l[od.b[m].values]
# test: pakai semua train
c1 = np.bincount(od.b[(od.fold >= 0) & (od.target == 1)], minlength=nb) + 1.0
c0 = np.bincount(od.b[(od.fold >= 0) & (od.target == 0)], minlength=nb) + 1.0
lall = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
llr[(od.fold == -1).values] = lall[od.b[od.fold == -1].values]
od["llr"] = llr
print("LLR per bin (hari sebelum keputusan -> llr):")
print(pd.Series(lall, index=np.round(edges[:-1], 1)).round(3).to_string())
g = od.groupby("customer_id")
F = {"llr_sum": g.llr.sum(), "llr_mean": g.llr.mean(), "n_orders": g.size(),
     "llr_sum_30d": od[od.tau <= 30].groupby("customer_id").llr.sum(),
     "llr_max": g.llr.max()}
print(table(ref, F).to_string(index=False))
# atribut order menjelang keputusan, per target (customer train)
t = od[od.fold >= 0].copy()
t["win"] = pd.cut(t.tau, [1, 3, 7, 30, 90, 365, 732])
t["canc"] = (t.st == "cancelled").astype(int)
t["cod"] = (t.pm == "COD").astype(int)
t["cc"] = (t.pm == "credit_card").astype(int)
t["lt"] = np.log(t.total)
print("rata-rata log nominal order per jendela x target:\n", t.groupby(["win", "target"], observed=True).lt.mean().unstack().round(3))
print("porsi cancel:\n", t.groupby(["win", "target"], observed=True).canc.mean().unstack().round(4))
print("porsi COD:\n", t.groupby(["win", "target"], observed=True).cod.mean().unstack().round(4))
print("porsi kartu kredit:\n", t.groupby(["win", "target"], observed=True).cc.mean().unstack().round(4))
print("n item per order:\n", t.groupby(["win", "target"], observed=True).n_items.mean().unstack().round(3))
