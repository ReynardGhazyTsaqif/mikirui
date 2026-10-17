"""Langkah 2k: baterai kandidat sinyal baru dari data mentah vs residual E122x1 (+ cek kebocoran urutan baris/ID).

Kelompok 'LEAKCHECK_*' hanya untuk deteksi kebocoran data panitia: TIDAK boleh dipakai sebagai fitur.
"""
import numpy as np
import pandas as pd

from common import DATA, apps, load
from resid import reference, table

pd.set_option("display.width", 220)
A = apps()
ref = reference("E122x1")
F = {}
# ---------------------------------------------------------------- cek kebocoran: posisi baris & ID
for name in ["repayment_history", "bureau_observations", "account_behavior", "previous_applications", "customer_profiles"]:
    d = pd.read_csv(DATA / f"{name}.csv", usecols=["customer_id"])
    pos = pd.Series(np.arange(len(d)) / len(d), index=d.customer_id)
    F[f"LEAKCHECK_pos_{name}"] = pos.groupby(level=0).min()
d = pd.read_csv(DATA / "ecommerce_transactions.csv", usecols=["customer_id"])
F["LEAKCHECK_pos_ecommerce"] = pd.Series(np.arange(len(d)) / len(d), index=d.customer_id).groupby(level=0).min()
del d
# urutan order dalam blok customer e-commerce: apakah kronologis?
ec = pd.read_csv(DATA / "ecommerce_transactions.csv", usecols=["customer_id", "order_id", "transaction_datetime"],
                 parse_dates=["transaction_datetime"]).drop_duplicates("order_id")
ec["i"] = ec.groupby("customer_id").cumcount()
F["LEAKCHECK_ec_order_unsorted"] = ec.groupby("customer_id").transaction_datetime.apply(lambda s: float((s.diff().dt.total_seconds() < 0).mean()))
pa = load("previous_applications")
F["LEAKCHECK_pa_unsorted"] = pa.groupby("customer_id").decision_datetime.apply(lambda s: float((s.diff().dt.total_seconds() < 0).mean()))
for idc, tab in [("loan_id", pa.dropna(subset=["loan_id"])), ("application_id", pa)]:
    h = tab[idc].str[2:6].apply(lambda x: int(x, 16) / 65535.0)
    F[f"LEAKCHECK_hex_{idc}_mean"] = h.groupby(tab.customer_id).mean()
# ---------------------------------------------------------------- repayment: perilaku event
EV = pd.read_parquet("cache_events_full.parquet")
RP = load("repayment_history")
RP["partial"] = (RP.amount_paid > 0) & (RP.amount_paid < RP.amount_due)
F["rp_n_partial"] = RP.groupby("customer_id").partial.sum()
F["rp_partial_share"] = RP.groupby("customer_id").partial.mean()
S = pd.read_parquet("cache_struct.parquet")
F["ev_catch_c"] = S.c
F["ev_q0"] = S.q0
EV["post_term"] = EV.j > EV.term
F["ev_post_term_events"] = EV.groupby("customer_id").post_term.sum()
# beruntun miss terpanjang
EV = EV.sort_values(["loan_id", "j"])
grp = (EV.miss != EV.groupby("loan_id").miss.shift()).cumsum()
run = EV.groupby(grp).miss.transform("size") * EV.miss
F["ev_max_miss_run"] = run.groupby(EV.customer_id).max()
# jumlah episode default (A mencapai 3)
EV["hit3"] = (EV.A >= 3) & (EV.A_prev < 3)
F["ev_n_hit3"] = EV.groupby("customer_id").hit3.sum()
F["ev_months_since_last_miss"] = EV[EV.miss == 1].groupby("customer_id").m_before.min()
# ---------------------------------------------------------------- biro: waktu
BU = load("bureau_observations")
cur = BU[BU.application_id.isin(set(A.application_id))].merge(A[["application_id", "application_datetime", "decision_datetime"]], on="application_id")
cur["dec_minus_av_h"] = (cur.decision_datetime - cur.available_datetime).dt.total_seconds() / 3600
cur["av_minus_app_h"] = (cur.available_datetime - cur.application_datetime).dt.total_seconds() / 3600
cur["rep_after_app"] = cur.report_datetime > cur.application_datetime
cur["lag_d"] = (cur.available_datetime - cur.report_datetime).dt.total_seconds() / 86400
g = cur.groupby("customer_id")
F["bu_dec_minus_av_h_min"] = g.dec_minus_av_h.min()
F["bu_av_minus_app_h_max"] = g.av_minus_app_h.max()
F["bu_rep_after_app"] = g.rep_after_app.max().astype(float)
F["bu_lag_d_mean"] = g.lag_d.mean()
# ---------------------------------------------------------------- aplikasi: waktu
A2 = A.set_index("customer_id")
F["app_proc_h"] = (A2.decision_datetime - A2.application_datetime).dt.total_seconds() / 3600
F["app_dec_dow"] = A2.decision_datetime.dt.dayofweek.astype(float)
F["app_dec_day"] = A2.decision_datetime.dt.day.astype(float)
F["app_app_hour"] = A2.application_datetime.dt.hour.astype(float)
# ---------------------------------------------------------------- e-commerce: order setelah aplikasi
od = pd.read_parquet("cache_ec_orders.parquet")
F["ec_n_after_app"] = (od.rec_app_h < 0).groupby(od.customer_id).sum()
F["ec_last_vs_app_h"] = od.groupby("customer_id").rec_app_h.min()
# ---------------------------------------------------------------- rekening
AB = load("account_behavior")
ga = AB.groupby("customer_id")
F["ab_n_months"] = ga.size()
F["ab_min0_months"] = (AB.minimum_balance == 0).groupby(AB.customer_id).sum()
F["ab_zero_income_months"] = (AB.income_like_credits == 0).groupby(AB.customer_id).sum()
F["ab_first_open"] = AB.sort_values("month").groupby("customer_id").opening_balance.first()
print(table(ref, F).to_string(index=False))
