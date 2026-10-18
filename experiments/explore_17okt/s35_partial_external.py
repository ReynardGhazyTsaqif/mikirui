"""Putaran 2: (G) pembayaran parsial cicilan dan (F) utang eksternal = saldo biro - sisa kewajiban internal.

(G) 2,0% baris cicilan dibayar sebagian (0 < bayar < tagihan) dan rantai event E200 hanya menghitung cicilan lunas
    penuh. Fitur: porsi cicilan parsial, nilai parsial terakhir, parsial pada cicilan terbaru, total kekurangan bayar
    relatif cicilan.
(F) Laporan biro aplikasi sekarang (outstanding_balance, active_accounts) dikurangi sisa kewajiban pinjaman internal
    yang masih berjalan di cutoff (cicilan belum lunas + cicilan belum jatuh tempo x nilai cicilan).
Uji: AUC univariat dan z residual OOF E202 (resid.py).
"""
import numpy as np
import pandas as pd

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 220)
ref = reference("E202")
A = apps()
rp = load("repayment_history", usecols=["loan_id", "customer_id", "due_date", "amount_due", "amount_paid"])
rp = rp.sort_values(["loan_id", "due_date"], kind="stable")
rp["part"] = ((rp.amount_paid > 0) & (rp.amount_paid < rp.amount_due)).astype(float)
rp["short"] = (rp.amount_due - rp.amount_paid).clip(lower=0)
rp["frac"] = np.where(rp.part == 1, rp.amount_paid / rp.amount_due, np.nan)
last = rp.groupby("loan_id").tail(1)
g = rp.groupby("customer_id")
F = {"g_part_share": g.part.mean(), "g_n_part": g.part.sum(),
     "g_last_frac": rp.dropna(subset=["frac"]).groupby("customer_id").frac.last(),
     "g_part_on_latest": last.groupby("customer_id").part.max(),
     "g_short_vs_due": g.short.sum() / g.amount_due.mean()}
# (F) sisa kewajiban internal: cicilan belum lunas + cicilan yang belum jatuh tempo (tenor - jumlah baris)
pa = load("previous_applications", usecols=["loan_id", "customer_id", "term_months", "scheduled_monthly_installment"])
pa = pa.dropna(subset=["loan_id"])
L = rp.groupby("loan_id").agg(n_rows=("due_date", "size"), short=("short", "sum"))
pa = pa.merge(L, left_on="loan_id", right_index=True, how="left")
pa["n_rows"] = pa.n_rows.fillna(0)
pa["short"] = pa.short.fillna(0)
pa["remaining"] = pa.short + (pa.term_months - pa.n_rows).clip(lower=0) * pa.scheduled_monthly_installment
internal = pa.groupby("customer_id").remaining.sum()
n_active = (pa.remaining > 0).groupby(pa.customer_id).sum()
bu = load("bureau_observations", usecols=["application_id", "outstanding_balance", "active_accounts"])
bu = bu[bu.application_id.isin(A.application_id)].groupby("application_id").agg(
    out=("outstanding_balance", "mean"), acc=("active_accounts", "mean"))
cur = A.set_index("customer_id").application_id
bout = cur.map(bu.out)
bacc = cur.map(bu.acc)
inc = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"]).set_index("customer_id").declared_monthly_income
intl = internal.reindex(cur.index).fillna(0)
F["f_internal_remaining"] = intl
F["f_bureau_minus_internal"] = bout - intl
F["f_external_vs_income"] = (bout - intl) / inc.reindex(cur.index)
F["f_internal_share_of_bureau"] = intl / bout.replace(0, np.nan)
F["f_ext_accounts"] = bacc - n_active.reindex(cur.index).fillna(0)
ok = intl > 0
print("customer dengan sisa kewajiban internal > 0:", int(ok.sum()), "| korelasi saldo biro ~ internal (yang > 0):",
      round(np.corrcoef(bout[ok], intl[ok])[0, 1], 3), "| median rasio internal/biro:", round((intl[ok] / bout[ok]).median(), 3))
print(table(ref, F).to_string(index=False))
