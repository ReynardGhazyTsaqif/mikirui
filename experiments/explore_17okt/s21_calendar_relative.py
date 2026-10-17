"""Langkah 2t: (a) guncangan kalender pada lewat-bayar & default; (b) ciri pinjaman sekarang relatif pinjaman lama."""
import numpy as np
import pandas as pd

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 220)
EV = pd.read_parquet("cache_events_full.parquet")
EV["cal_m"] = (EV.cal % 12) + 1
EV["cal_y"] = EV.cal // 12
pm = EV.groupby(["cal_y", "cal_m"]).miss.agg(["mean", "size"])
pm = pm[pm["size"] > 5000]
print("P(miss) per bulan kalender event (tahun, bulan):\n", pm["mean"].unstack().round(3).to_string())
print("sd antar bulan %.4f vs sd sampling ~%.4f" % (pm["mean"].std(), np.sqrt(0.18 * 0.82 / pm["size"].median())))
L = EV.groupby("loan_id").agg(dis=("dis", "first"), cutoff=("cutoff", "first"))
L["lab"] = (EV[EV.j <= 9].groupby("loan_id").A.max().reindex(L.index) >= 3).astype(int)
L = L[(L.dis + pd.Timedelta(days=365)) <= L.cutoff]
print("default pinjaman lama per bulan pencairan:", L.groupby(L.dis.dt.to_period("Q")).lab.agg(["mean", "size"]).round(3).T.to_string())
# (b) ciri pinjaman sekarang relatif pinjaman lama
A = apps()
P = load("customer_profiles").set_index("customer_id")
PA = load("previous_applications")
ap = PA[PA.decision == "approved"].copy()
ap["inc"] = ap.customer_id.map(P.declared_monthly_income)
ap["lti"] = ap.approved_amount / ap.inc
ap["pti"] = ap.scheduled_monthly_installment / ap.inc
g = ap.groupby("customer_id")
cur = A.set_index("customer_id")
inc = P.declared_monthly_income.reindex(cur.index)
lti_c = cur.loan_amount / inc
pti_c = cur.proposed_monthly_installment / inc
ref = reference("E122x1")
F = {"lti_rel_mean": lti_c / g.lti.mean(), "lti_rel_max": lti_c / g.lti.max(), "pti_rel_mean": pti_c / g.pti.mean(),
     "rate_rel_mean": cur.annual_interest_rate - g.annual_interest_rate.mean(),
     "amount_rel_last": cur.loan_amount / ap.sort_values("decision_datetime").groupby("customer_id").approved_amount.last(),
     "pti_total_active": None}
del F["pti_total_active"]
print(table(ref, F).to_string(index=False))
