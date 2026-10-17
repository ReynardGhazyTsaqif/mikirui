"""Langkah 2o: apa yang menentukan default suatu pinjaman, selain customer-nya?

1. Default pinjaman lama lengkap per tier bunga, tenor, PTI (cicilan/income), nominal/income, approved/requested.
2. Dalam customer yang sama (efek tetap customer): apakah pinjaman dengan tier/PTI lebih tinggi lebih sering default?
   (logit bersyarat sederhana: pasangan pinjaman customer sama dengan label berbeda)
3. Ramp e-commerce sebelum keputusan pinjaman LAMA: laju order 0-7 hari sebelum keputusan lama vs 30-90 hari,
   per label pinjaman lama.
"""
import numpy as np
import pandas as pd

from common import DATA, apps, load

pd.set_option("display.width", 220)
A = apps()
P = load("customer_profiles")
PA = load("previous_applications")
EV = pd.read_parquet("cache_events_full.parquet")
L = EV.groupby("loan_id").agg(customer_id=("customer_id", "first"), dis=("dis", "first"), cutoff=("cutoff", "first"))
L["lab"] = (EV[EV.j <= 9].groupby("loan_id").A.max().reindex(L.index) >= 3).astype(int)
L = L[(L.dis + pd.Timedelta(days=365)) <= L.cutoff].join(PA.set_index("loan_id")[["requested_amount", "approved_amount", "term_months",
                                                                                   "annual_interest_rate", "scheduled_monthly_installment", "application_id"]])
L = L.reset_index().merge(P[["customer_id", "declared_monthly_income", "age"]], on="customer_id")
L["pti"] = L.scheduled_monthly_installment / L.declared_monthly_income
L["lti"] = L.approved_amount / L.declared_monthly_income
L["cut_ratio"] = L.approved_amount / L.requested_amount
print("default per tier:", L.groupby("annual_interest_rate").lab.agg(["mean", "size"]).round(3).T.to_string())
print("default per tenor:", L.groupby("term_months").lab.mean().round(3).to_dict())
print("default per desil PTI:", L.groupby(pd.qcut(L.pti, 10)).lab.mean().round(3).tolist())
print("default per kuintil LTI:", L.groupby(pd.qcut(L.lti, 5)).lab.mean().round(3).tolist())
print("default: dipotong vs tidak:", L.groupby(L.cut_ratio < 1).lab.mean().round(3).to_dict())
# dalam customer: pasangan pinjaman dengan label berbeda
pr = L.merge(L, on="customer_id")
pr = pr[(pr.lab_x == 1) & (pr.lab_y == 0)]
for c in ["annual_interest_rate", "pti", "lti", "term_months", "cut_ratio"]:
    d = pr[f"{c}_x"] - pr[f"{c}_y"]
    print(f"dalam customer: pinjaman default vs tidak, {c}: P(lebih tinggi) {np.mean(d > 0):.3f}, P(lebih rendah) {np.mean(d < 0):.3f}, n {len(d)}")
# urutan waktu: pinjaman lebih baru lebih sering default?
d = (pr.dis_x - pr.dis_y).dt.days
print("dalam customer: pinjaman default lebih baru? P =", np.mean(d > 0).round(3))
# ramp e-commerce sebelum keputusan lama
od = pd.read_parquet("cache_ec_orders.parquet")[["customer_id", "t"]]
la = L[["loan_id", "customer_id", "dis", "lab"]]
# hanya pinjaman lama yang keputusannya >= 120 hari setelah awal riwayat e-commerce customer
first = od.groupby("customer_id").t.min()
la = la[la.dis - pd.Timedelta(days=120) > la.customer_id.map(first)]
x = la.merge(od, on="customer_id")
x["d"] = (x.dis - x.t).dt.total_seconds() / 86400
x = x[(x.d > 0) & (x.d <= 120)]
r = x.groupby("loan_id").agg(n7=("d", lambda s: (s <= 7).sum()), n30_120=("d", lambda s: (s > 30).sum()))
r = la.set_index("loan_id").join(r).fillna(0)
r["ratio"] = r.n7 / 7 / ((r.n30_120 + 0.5) / 90)
print("pinjaman lama: rata laju order 0-7 hari relatif 30-120 hari, per label:", r.groupby("lab").ratio.mean().round(3).to_dict(),
      "| n:", r.lab.value_counts().to_dict())
print("pinjaman lama: P(order dalam 7 hari sebelum keputusan lama) per label:", r.groupby("lab").n7.apply(lambda s: (s > 0).mean()).round(3).to_dict())
