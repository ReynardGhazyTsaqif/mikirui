"""H24 putaran CAAFE 2 (umpan balik putaran 1: konsep keterjangkauan/penumpukan/kredit eksternal sudah jenuh).
Fokus ke satu-satunya keluarga sinyal yang bergantung langsung pada target: ramp sebelum keputusan, diukur relatif
terhadap baseline customer sendiri dengan statistik yang tepat.

- LLR change-point Poisson order: H0 laju tetap pada [1 hari, T]; H1 laju beda di w hari terakhir;
  LLR(w) = n_w ln(n_w/w) + (n-n_w) ln((n-n_w)/(T-w)) - n ln(n/T); bertanda (+ jika laju naik); maks atas w.
- versi nominal belanja (total per hari) dengan statistik rasio laju yang sama (pendekatan Gamma-Poisson kasar).
- z-score rekening 1/3 bulan terakhir terhadap rata-rata dan sd bulanan customer sendiri (bulan sebelumnya).
- skor ramp gabungan = z(LLR order) + z(outflow/income) - z(arus bersih) + z(hari saldo rendah).
Saring: z residual OOF E202 fold 0-2 (seleksi) dan 3-4 (konfirmasi), seperti s42.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import load
from resid import reference

ref = reference("E202")
fold, r, y = ref.fold.values, ref.r.values, ref.target.values
od = pd.read_parquet("cache_ec_orders.parquet", columns=["customer_id", "rec_h", "total", "st"])
od = od[od.st.astype(str) != "cancelled"]
od["d"] = od.rec_h / 24.0  # hari sebelum keputusan (>= 1 karena dipotong 24 jam)
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
    rate_ratio = (sw / w + 1) / ((spend_tot - sw) / (T - w).clip(lower=1) + 1)
    F[f"cp_logratio_spend_w{w}"] = np.log(rate_ratio)
    best_s = np.maximum(best_s, np.log(rate_ratio))
F["cp_llr_orders_max"] = best
F["cp_spend_ratio_max"] = best_s
ab = load("account_behavior").sort_values(["customer_id", "month"])
ab["oi"] = ab.outflows / ab.income_like_credits.replace(0, np.nan)
ab["nf"] = (ab.inflows - ab.outflows) / ab.income_like_credits.replace(0, np.nan)
ab["rk"] = ab.groupby("customer_id").cumcount(ascending=False)  # 0 = bulan terakhir
for c in ["oi", "nf", "low_balance_days", "closing_balance"]:
    base = ab[ab.rk >= 3].groupby("customer_id")[c].agg(["mean", "std"])
    for k in (1, 3):
        last = ab[ab.rk < k].groupby("customer_id")[c].mean()
        F[f"cp_z_{c}_last{k}"] = (last - base["mean"]) / base["std"].replace(0, np.nan)


def zs(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    return (s - s.mean()) / s.std()


F["cp_joint_ramp"] = (zs(F["cp_llr_orders_max"]).reindex(ref.customer_id).fillna(0).values
                      + zs(F["cp_z_oi_last3"]).reindex(ref.customer_id).fillna(0).values
                      - zs(F["cp_z_nf_last3"]).reindex(ref.customer_id).fillna(0).values
                      + zs(F["cp_z_low_balance_days_last3"]).reindex(ref.customer_id).fillna(0).values)
F["cp_joint_ramp"] = pd.Series(F["cp_joint_ramp"], index=ref.customer_id.values)
gap = od.sort_values(["customer_id", "d"]).groupby("customer_id").d.diff()
F["cp_last_gap_vs_median"] = gap.groupby(od.customer_id).first() / gap.groupby(od.customer_id).median()
sel, val = np.isin(fold, [0, 1, 2]), np.isin(fold, [3, 4])
rows = []
for k, s in F.items():
    x = ref.customer_id.map(s).astype(float).values
    zz = []
    for msk in (sel, val):
        ok = msk & np.isfinite(x)
        zz.append(np.corrcoef(rankdata(x[ok]), r[ok])[0, 1] * np.sqrt(ok.sum()) if ok.sum() > 1000 and np.nanstd(x[ok]) > 0 else np.nan)
    ok = np.isfinite(x)
    auc = roc_auc_score(y[ok], x[ok])
    rows.append((k, round(max(auc, 1 - auc), 4), round(ok.mean(), 3), round(zz[0], 2), round(zz[1], 2)))
Tb = pd.DataFrame(rows, columns=["fitur", "auc", "cakupan", "z_sel", "z_val"])
Tb["lolos"] = (Tb.z_sel.abs() >= 3) & (np.sign(Tb.z_sel) == np.sign(Tb.z_val)) & (Tb.z_val.abs() >= 2)
print(Tb.sort_values("z_sel", key=np.abs, ascending=False).to_string(index=False))
