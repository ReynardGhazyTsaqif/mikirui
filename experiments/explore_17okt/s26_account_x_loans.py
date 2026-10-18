"""Putaran 2 penemuan: rekening x pinjaman.

1. Apakah pencairan pinjaman internal terlihat sebagai lonjakan inflow non-gaji di bulan pencairan?
2. Lonjakan inflow non-gaji yang TIDAK dijelaskan pinjaman internal (indikasi utang lembaga lain / loan stacking).
3. Ramp relatif baseline sendiri: outflow & arus bersih bulan terakhir, belanja e-commerce 30 hari terakhir.
Semua tanpa target; dinilai dengan z residual di atas OOF E200.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import apps, load  # noqa: E402
from resid import reference, table  # noqa: E402

pd.set_option("display.width", 220)
A = apps()
P = load("customer_profiles").set_index("customer_id")
AB = load("account_behavior", usecols=["customer_id", "month", "inflows", "outflows", "income_like_credits", "closing_balance", "low_balance_days"]).merge(A[["customer_id", "decision_datetime"]], on="customer_id")
AB["inc"] = AB.customer_id.map(P.declared_monthly_income)
AB["mago"] = (AB.decision_datetime.dt.year * 12 + AB.decision_datetime.dt.month) - (AB.month.dt.year * 12 + AB.month.dt.month)
AB["cal"] = AB.month.dt.year * 12 + AB.month.dt.month - 1
AB["nonincome_in"] = AB.inflows - AB.income_like_credits
PA = load("previous_applications")
ap = PA[PA.decision == "approved"].copy()
ap["cal"] = ap.decision_datetime.dt.year * 12 + ap.decision_datetime.dt.month - 1
dis = ap.groupby(["customer_id", "cal"]).approved_amount.sum().rename("disbursed").reset_index()
AB = AB.merge(dis, on=["customer_id", "cal"], how="left")
AB["disbursed"] = AB.disbursed.fillna(0)
# 1. apakah pencairan terlihat di rekening?
has = AB.disbursed > 0
print("bulan dengan pencairan internal:", int(has.sum()), "dari", len(AB))
print("rata-rata inflow non-gaji / income: bulan pencairan %.3f vs bulan lain %.3f" % (
    (AB.nonincome_in / AB.inc)[has].mean(), (AB.nonincome_in / AB.inc)[~has].mean()))
print("korelasi inflow non-gaji dengan nominal pencairan (bulan pencairan):",
      np.corrcoef(AB.nonincome_in[has], AB.disbursed[has])[0, 1].round(3))
print("rasio median inflow non-gaji bulan pencairan / nominal pencairan:", (AB.nonincome_in[has] / AB.disbursed[has]).median().round(3))
# 2. lonjakan non-gaji tak terjelaskan
AB["nonin_ratio"] = AB.nonincome_in / AB.inc
med = AB.groupby("customer_id").nonin_ratio.transform("median")
AB["spike"] = (AB.nonin_ratio > med + 1.0) & (AB.disbursed == 0)
g = AB.groupby("customer_id")
F = {"ab_unexpl_spikes": g.spike.sum(), "ab_unexpl_spikes_6m": AB[AB.mago <= 6].groupby("customer_id").spike.sum(),
     "ab_nonin_ratio_mean": g.nonin_ratio.mean(),
     "ab_nonin_ratio_last3": AB[AB.mago <= 3].groupby("customer_id").nonin_ratio.mean()}
# 3. ramp relatif baseline sendiri
base = AB[AB.mago.between(4, 12)].groupby("customer_id")
last = AB[AB.mago == 1].set_index("customer_id")
F["ab_out_last_vs_base"] = last.outflows / base.outflows.mean()
F["ab_net_last_vs_inc"] = (last.inflows - last.outflows) / last.inc
F["ab_bal_last_vs_base"] = last.closing_balance / base.closing_balance.mean().replace(0, np.nan)
F["ab_low_last_minus_base"] = last.low_balance_days - base.low_balance_days.mean()
od = pd.read_parquet(HERE / "cache_ec_orders.parquet", columns=["customer_id", "rec_h", "total", "st"])
od = od[od.st == "completed"]
recent = od[od.rec_h <= 30 * 24].groupby("customer_id").total.sum()
hist = od[od.rec_h > 30 * 24]
span = hist.groupby("customer_id").rec_h.max().clip(lower=60 * 24) / 24 - 30
F["ec_spend30_vs_base"] = recent / (hist.groupby("customer_id").total.sum() / span * 30)
ref = reference("E200")
print(table(ref, F).to_string(index=False))
