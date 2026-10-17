"""Langkah 2p: kasus salah E122x1 diukur dengan besaran pembangkit.

Kelompok (customer train, OOF bag 3 seed):
  FN  = target 1 dengan persentil skor < 50%      TP_hi = target 1 dengan persentil > 90%
  FP  = target 0 dengan persentil > 95%            TN_lo = target 0 dengan persentil < 50%
  Pembanding sebanding: target 0 dengan persentil < 50% (untuk FN), target 1 persentil > 95% (untuk FP).
Juga: apakah ramp e-commerce terkait target langsung atau lewat risiko customer (pi)?
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import apps, load
from resid import reference

pd.set_option("display.width", 220)
ref = reference("E122x1")
ref["pct"] = rankdata(ref.p) / len(ref)
c = ref.set_index("customer_id")
LP = pd.read_parquet("cache_loan_posterior.parquet")
ST = pd.read_parquet("cache_struct.parquet")
EC = pd.read_parquet("cache_ec_customer.parquet")
od = pd.read_parquet("cache_ec_orders.parquet")
n7 = (od.rec_h <= 7 * 24).groupby(od.customer_id).sum()
A = apps().set_index("customer_id")
BU = load("bureau_observations")
bu = BU[BU.application_id.isin(set(A.application_id))].groupby("customer_id").score.mean()
AB = load("account_behavior").merge(A[["decision_datetime"]].reset_index(), on="customer_id")
AB["mago"] = (AB.decision_datetime.dt.year * 12 + AB.decision_datetime.dt.month) - (AB.month.dt.year * 12 + AB.month.dt.month)
last_out = AB[AB.mago == 1].set_index("customer_id").outflows
base_out = AB[AB.mago.between(4, 12)].groupby("customer_id").outflows.mean()
P = load("customer_profiles").set_index("customer_id")
X = pd.DataFrame({
    "has_hist": c.index.isin(LP.index),
    "pi_post": LP.lo_post_mean.reindex(c.index),
    "n_default_prev": LP.lo_n_default.reindex(c.index),
    "q0": ST.q0.reindex(c.index), "catch_c": ST.c.reindex(c.index),
    "ec_last_h": EC.last_h.reindex(c.index), "ec_n7": n7.reindex(c.index).fillna(0),
    "ec_n_orders": EC.n.reindex(c.index),
    "ab_out_ramp": (last_out / base_out).reindex(c.index),
    "bu_score": bu.reindex(c.index), "tier": A.annual_interest_rate.reindex(c.index),
    "income": P.declared_monthly_income.reindex(c.index), "pti": (A.proposed_monthly_installment / P.declared_monthly_income).reindex(c.index),
}, index=c.index)
X["target"], X["pct"] = c.target, c.pct
grp = {
    "FN (y=1, pct<0.5)": (X.target == 1) & (X.pct < 0.5),
    "TN_lo (y=0, pct<0.5)": (X.target == 0) & (X.pct < 0.5),
    "TP_hi (y=1, pct>0.9)": (X.target == 1) & (X.pct > 0.9),
    "FP (y=0, pct>0.95)": (X.target == 0) & (X.pct > 0.95),
    "TP_vhi (y=1, pct>0.95)": (X.target == 1) & (X.pct > 0.95),
}
rows = []
for k, m in grp.items():
    s = X[m]
    rows.append({"kelompok": k, "n": int(m.sum()), "punya_riwayat": s.has_hist.mean(), "pi_post": s.pi_post.mean(),
                 "n_def_prev": s.n_default_prev.mean(), "q0": s.q0.mean(), "ec_last_h_med": s.ec_last_h.median(),
                 "ec_n7": s.ec_n7.mean(), "ab_out_ramp_med": s.ab_out_ramp.median(), "bu_score": s.bu_score.mean(),
                 "tier": s.tier.mean(), "pti": s.pti.mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("porsi defaulter yang skornya < median:", ((X.target == 1) & (X.pct < 0.5)).sum() / (X.target == 1).sum())
# ramp e-commerce: terkait target langsung atau lewat pi?
h = X[X.has_hist].copy()
h["pi_bin"] = pd.qcut(h.pi_post.rank(method="first"), 5, labels=False)
print("median jam order terakhir per kuintil pi_post (baris) x target:")
print(h.groupby(["pi_bin", "target"]).ec_last_h.median().unstack().round(1).to_string())
print("rata n order 7 hari per kuintil pi_post x target:")
print(h.groupby(["pi_bin", "target"]).ec_n7.mean().unstack().round(3).to_string())
print("median ramp outflow rekening per kuintil pi_post x target:")
print(h.groupby(["pi_bin", "target"]).ab_out_ramp.median().unstack().round(3).to_string())
X.to_parquet("cache_error_frame.parquet")
