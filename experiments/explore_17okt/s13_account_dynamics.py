"""Langkah 2l: dinamika rekening per target dan keterkaitan bulanan dengan lewat-bayar cicilan."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 220)
A = apps()
P = load("customer_profiles")
AB = load("account_behavior").merge(A[["customer_id", "decision_datetime", "target", "is_test"]], on="customer_id")
AB = AB.merge(P[["customer_id", "declared_monthly_income"]], on="customer_id")
AB["mago"] = (AB.decision_datetime.dt.year * 12 + AB.decision_datetime.dt.month) - (AB.month.dt.year * 12 + AB.month.dt.month)
AB["lbal"] = np.log1p(AB.closing_balance)
AB["inc_ratio"] = AB.income_like_credits / AB.declared_monthly_income
AB["out_ratio"] = AB.outflows / AB.declared_monthly_income
AB["net_ratio"] = (AB.inflows - AB.outflows) / AB.declared_monthly_income
AB["low"] = (AB.low_balance_days > 0).astype(int)
t = AB[AB.is_test == 0]
for c in ["lbal", "inc_ratio", "out_ratio", "net_ratio", "low", "low_balance_days"]:
    m = t[t.mago <= 12].groupby(["mago", "target"])[c].mean().unstack()
    print(f"{c} per bulan sebelum keputusan (baris) x target:\n", m.round(3).T.to_string())
# kaitan bulanan dengan lewat-bayar
EV = pd.read_parquet("cache_events_full.parquet")
EV["cal"] = EV.cal.astype(int)
AB["cal"] = AB.month.dt.year * 12 + AB.month.dt.month - 1
em = EV.groupby(["customer_id", "cal"]).miss.max().rename("miss_any").reset_index()
j = AB.merge(em, on=["customer_id", "cal"], how="inner")
print("bulan dengan event bayar & data rekening:", len(j))
for c in ["low", "net_ratio", "inc_ratio", "out_ratio"]:
    print(f"P(miss) per kuintil {c} (dalam bulan sama):", j.groupby(pd.qcut(j[c].rank(method='first'), 5, labels=False)).miss_any.mean().round(3).tolist())
# within-customer: deviasi bulanan dari rata-rata customer
for c in ["inc_ratio", "net_ratio", "lbal"]:
    j[c + "_dev"] = j[c] - j.groupby("customer_id")[c].transform("mean")
    print(f"P(miss) per kuintil deviasi {c} dari rata-rata customer:",
          j.groupby(pd.qcut(j[c + "_dev"].rank(method='first'), 5, labels=False)).miss_any.mean().round(3).tolist())
# kandidat fitur rekening relatif terhadap income deklarasi (belum ada versi ini?)
g = AB.groupby("customer_id")
ref = reference("E122x1")
F = {"ab_inc_ratio_mean": g.inc_ratio.mean(), "ab_inc_ratio_std": g.inc_ratio.std(), "ab_out_ratio_mean": g.out_ratio.mean(),
     "ab_net_ratio_mean": g.net_ratio.mean(), "ab_low_share": g.low.mean(),
     "ab_lbal_slope": g.apply(lambda d: np.polyfit(-d.mago, d.lbal, 1)[0] if len(d) > 2 else np.nan),
     "ab_inc_last3": AB[AB.mago <= 3].groupby("customer_id").inc_ratio.mean(),
     "ab_bal_to_inc_last": AB[AB.mago == 1].set_index("customer_id").closing_balance / AB[AB.mago == 1].set_index("customer_id").declared_monthly_income}
print(table(ref, F).to_string(index=False))
