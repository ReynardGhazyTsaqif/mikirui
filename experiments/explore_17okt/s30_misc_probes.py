"""Putaran 2: probe murah untuk sinyal yang belum pernah diuji, terhadap residual OOF E202 (acuan kerja terbaik).

(a) Fase aplikasi dalam siklus jatuh tempo pinjaman aktif: apply menjelang due date berikutnya = butuh dana?
    Pembayaran hanya terjadi di ulang bulan pencairan, jadi due berikutnya = pencairan + (j_terakhir + 1) bulan.
(b) Inkonsistensi profil: masa kerja vs umur kerja maksimum (data "janggal" kadang khas customer buruk di data sintetis).
"""
import numpy as np
import pandas as pd

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 220)
ref = reference("E202")
A = apps().set_index("customer_id")
ev = pd.read_parquet("cache_events_full.parquet", columns=["loan_id", "j", "customer_id", "dis", "cutoff", "term", "A"])
last = ev.sort_values(["loan_id", "j"]).groupby("loan_id").tail(1).copy()
last = last[(last.j < last.term) | (last.A > 0)]  # pinjaman masih punya kewajiban di cutoff
last["app_t"] = last.customer_id.map(A.application_datetime)
nxt = [d + pd.DateOffset(months=int(j) + 1) for d, j in zip(last.dis, last.j)]
last["next_due"] = pd.to_datetime(nxt)
last["d_app"] = (last.next_due - last.app_t).dt.total_seconds() / 86400
last["d_dec"] = (last.next_due - last.cutoff).dt.total_seconds() / 86400
last["since_last_due_app"] = (last.app_t - (last.dis + pd.to_timedelta(0, "D"))).dt.total_seconds() / 86400
last = last[last.d_app > -40]
g = last.groupby("customer_id")
F = {
    "a_days_to_next_due_app_min": g.d_app.min(),
    "a_days_to_next_due_dec_min": g.d_dec.min(),
    "a_n_due_within7_app": (last.d_app.between(0, 7)).groupby(last.customer_id).sum(),
    "a_n_due_within3_app": (last.d_app.between(0, 3)).groupby(last.customer_id).sum(),
    "a_due_before_dec": ((last.d_app >= 0) & (last.d_dec < 0)).groupby(last.customer_id).sum(),
    "a_n_active_loans": g.size(),
    "a_arrears_active": g.A.sum(),
}
prof = load("customer_profiles").set_index("customer_id")
work = ((prof.age - 18).clip(lower=1) * 12)
F["b_tenure_ratio"] = prof.employment_tenure_months / work
F["b_tenure_gt_worklife"] = (prof.employment_tenure_months > work).astype(float)
F["b_tenure_months"] = prof.employment_tenure_months
F["b_income_per_dependent"] = prof.declared_monthly_income / (1 + prof.dependents)
print(table(ref, F).to_string(index=False))
print("porsi tenure > umur kerja:", round(F["b_tenure_gt_worklife"].mean(), 4))
x = ref.customer_id.map(F["a_days_to_next_due_app_min"])
ok = x.notna()
print("target per kuintil hari ke due berikutnya (dari aplikasi), customer berpinjaman aktif:")
print(ref[ok].groupby(pd.qcut(x[ok], 5)).agg(n=("target", "size"), tgt=("target", "mean"), resid=("r", "mean")).round(4))
