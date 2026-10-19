"""Paket mekanisme 19 Okt: simpan fitur ramp change-point (logika s43, tanpa target) untuk train + test
-> features/cand_ramp_v1.parquet (index application_id).

- LLR change-point Poisson jumlah order (w 3/7/14/30/60 hari terakhir vs baseline customer sendiri), bertanda; maks.
- log rasio laju belanja w hari terakhir vs baseline; maks.
- z-score rekening 1 dan 3 bulan terakhir terhadap rata-rata/sd bulanan customer sendiri (outflow/income, arus bersih,
  hari saldo rendah, saldo akhir).
"""
import numpy as np
import pandas as pd

from common import MAIN, apps, load

A = apps()
od = pd.read_parquet("cache_ec_orders.parquet", columns=["customer_id", "rec_h", "total", "st"])
od = od[od.st.astype(str) != "cancelled"]
od["d"] = od.rec_h / 24.0
g = od.groupby("customer_id")
n = g.size()
T = g.d.max().clip(lower=30)


def xlogy(a, b):
    return np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0.0)


F = {}
best, best_s = pd.Series(-np.inf, index=n.index), pd.Series(-np.inf, index=n.index)
spend_tot = g.total.sum()
for w in (3, 7, 14, 30, 60):
    nw = od[od.d <= w + 1].groupby("customer_id").size().reindex(n.index).fillna(0)
    llr = xlogy(nw, np.full(len(nw), float(w))) + xlogy(n - nw, (T - w).clip(lower=1)) - xlogy(n, T)
    sign = np.sign(nw / w - (n - nw) / (T - w).clip(lower=1))
    F[f"cp_llr_orders_w{w}"] = pd.Series(sign * llr, index=n.index)
    best = np.maximum(best, sign * llr)
    sw = od[od.d <= w + 1].groupby("customer_id").total.sum().reindex(n.index).fillna(0)
    rr = np.log((sw / w + 1) / ((spend_tot - sw) / (T - w).clip(lower=1) + 1))
    F[f"cp_logratio_spend_w{w}"] = rr
    best_s = np.maximum(best_s, rr)
F["cp_llr_orders_max"] = best
F["cp_spend_ratio_max"] = best_s
ab = load("account_behavior").sort_values(["customer_id", "month"])
ab["oi"] = ab.outflows / ab.income_like_credits.replace(0, np.nan)
ab["nf"] = (ab.inflows - ab.outflows) / ab.income_like_credits.replace(0, np.nan)
ab["rk"] = ab.groupby("customer_id").cumcount(ascending=False)
for c in ["oi", "nf", "low_balance_days", "closing_balance"]:
    base = ab[ab.rk >= 3].groupby("customer_id")[c].agg(["mean", "std"])
    for k in (1, 3):
        last = ab[ab.rk < k].groupby("customer_id")[c].mean()
        F[f"cp_z_{c}_last{k}"] = (last - base["mean"]) / base["std"].replace(0, np.nan)
out = pd.DataFrame({k: A.customer_id.map(v).astype("float32").values for k, v in F.items()}, index=A.application_id.values)
out = out.replace([np.inf, -np.inf], np.nan)
out.index.name = "application_id"
out[sorted(out.columns)].to_parquet(MAIN / "features" / "cand_ramp_v1.parquet")
print(out.shape, out.notna().mean().round(3).to_dict())
