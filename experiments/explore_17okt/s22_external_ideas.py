"""Bukti data untuk ide eksternal yang belum punya bukti (LAMPIRAN no. 2, 6, 7, 8, 9).

LEAKCHECK_* hanya untuk deteksi kebocoran data panitia (dilaporkan, tidak dipakai sebagai fitur).
"""
import numpy as np
import pandas as pd

from common import DATA, apps, load
from resid import reference, table

pd.set_option("display.width", 220)
A = apps()
ref = reference("E122x1")
cust_of_app = A.set_index("application_id").customer_id
F = {}


def hexval(s, a=2, b=8):
    return s.str[a:b].apply(lambda x: int(x, 16) / 16 ** (b - a))


# ---------------------------------------------------------------- ide 2: ID & urutan baris tabel pendukung
rp = pd.read_csv(DATA / "repayment_history.csv", usecols=["loan_id", "installment_id", "customer_id", "due_date"],
                 parse_dates=["due_date"])
rp["pos"] = np.arange(len(rp))
F["LEAKCHECK_installment_id_hex"] = hexval(rp.installment_id).groupby(rp.customer_id).mean()
F["LEAKCHECK_loan_id_hex_rp"] = hexval(rp.loan_id).groupby(rp.customer_id).mean()
# apakah baris cicilan dalam file urut due_date per pinjaman?
rp["unsorted"] = rp.groupby("loan_id").due_date.diff().dt.total_seconds() < 0
F["LEAKCHECK_rp_unsorted"] = rp.groupby("customer_id").unsorted.mean().astype(float)
# urutan pinjaman dalam blok customer vs urutan waktu
first = rp.groupby("loan_id").agg(c=("customer_id", "first"), p=("pos", "min"), d=("due_date", "min"))
first["rank_pos"] = first.groupby("c").p.rank()
first["rank_time"] = first.groupby("c").d.rank()
F["LEAKCHECK_rp_loanorder_mismatch"] = (first.rank_pos != first.rank_time).groupby(first.c).mean().astype(float)
del rp
bo = pd.read_csv(DATA / "bureau_observations.csv", usecols=["bureau_observation_id", "application_id", "customer_id", "bureau_provider"])
F["LEAKCHECK_bureau_obs_id_hex"] = hexval(bo.bureau_observation_id).groupby(bo.customer_id).mean()
cur = bo[bo.application_id.isin(set(A.application_id))].copy()
cur["k"] = cur.groupby("application_id").cumcount()
cur["prov_code"] = cur.bureau_provider.str[-1].map({"A": 0, "B": 1, "C": 2})
cur["out_of_order"] = cur.groupby("application_id").prov_code.diff() < 0
F["LEAKCHECK_bu_provider_order"] = cur.groupby("customer_id").out_of_order.mean().astype(float)
del bo
ec = pd.read_csv(DATA / "ecommerce_transactions.csv", usecols=["order_id", "order_item_id", "customer_id"])
F["LEAKCHECK_order_id_hex"] = hexval(ec.order_id).groupby(ec.customer_id).mean()
F["LEAKCHECK_order_item_id_hex"] = hexval(ec.order_item_id).groupby(ec.customer_id).mean()
del ec
# ---------------------------------------------------------------- ide 6: artefak angka
P = load("customer_profiles").set_index("customer_id")
inc = P.declared_monthly_income
print("income kelipatan 1.000:", (inc % 1000 == 0).mean().round(4), "| 10.000:", (inc % 10000 == 0).mean().round(5),
      "| digit terakhir seragam? top:", (inc % 10).value_counts(normalize=True).round(3).head(3).to_dict())
PA = load("previous_applications")
print("requested kelipatan 100 rb:", (PA.requested_amount % 100000 == 0).mean(), "| approved kelipatan 100 rb:",
      (PA.approved_amount.dropna() % 100000 == 0).mean().round(4), "| approved kelipatan 1 rb:", (PA.approved_amount.dropna() % 1000 == 0).mean().round(4))
cut = PA[(PA.decision == "approved") & (PA.approved_amount < PA.requested_amount)]
print("approved dipotong: rasio unik (5 teratas):", (cut.approved_amount / cut.requested_amount).round(3).value_counts().head(5).to_dict())
F["inc_last3_digits"] = (inc % 1000).astype(float)
F["inc_round_10k"] = (inc % 10000 == 0).astype(float)
loan = A.set_index("customer_id").loan_amount
F["loan_round_1m"] = (loan % 1_000_000 == 0).astype(float)
F["loan_round_5m"] = (loan % 5_000_000 == 0).astype(float)
# ---------------------------------------------------------------- ide 7: tujuan pinjaman vs belanja
ecc = pd.read_csv(DATA / "ecommerce_transactions.csv", usecols=["customer_id", "transaction_datetime", "product_category", "item_total", "order_status"],
                  parse_dates=["transaction_datetime"])
ecc = ecc[ecc.order_status == "completed"].merge(A[["customer_id", "decision_datetime", "loan_purpose"]], on="customer_id")
ecc["age_d"] = (ecc.decision_datetime - ecc.transaction_datetime).dt.total_seconds() / 86400
match = {"medical": "health", "education": "education", "home_improvement": "home", "household": "groceries", "working_capital": None}
ecc["match"] = ecc.product_category == ecc.loan_purpose.map(match)
for w in (30, 90, 730):
    e = ecc[ecc.age_d <= w]
    tot = e.groupby("customer_id").item_total.sum()
    m = e[e.match].groupby("customer_id").item_total.sum().reindex(tot.index).fillna(0)
    F[f"purpose_match_share_{w}d"] = (m / tot).where(tot > 0)
    F[f"purpose_match_spend_{w}d"] = m
print("target per tujuan x ada belanja kategori cocok 90 hari (train):")
t = ref.set_index("customer_id").join(pd.DataFrame({"share": F["purpose_match_share_90d"]}))
t["purpose"] = A.set_index("customer_id").loan_purpose
t["has"] = t.share.fillna(0) > 0
print(t.groupby(["purpose", "has"]).target.agg(["mean", "size"]).round(4).unstack().to_string())
del ecc
# ---------------------------------------------------------------- ide 8: pinjam untuk menutup utang
EV = pd.read_parquet("cache_events_full.parquet")
PAa = PA[PA.decision == "approved"].set_index("loan_id")
last = EV.sort_values("j").groupby("loan_id").tail(1).set_index("loan_id")
L = last[["customer_id", "C", "term", "A"]].join(PAa[["approved_amount", "annual_interest_rate", "scheduled_monthly_installment", "term_months"]])
r = L.annual_interest_rate / 12
paid_k = L.C.clip(upper=L.term)  # jumlah cicilan yang sudah lunas
# sisa pokok anuitas setelah k cicilan dibayar
L["rem_principal"] = L.approved_amount * (1 + r) ** paid_k - L.scheduled_monthly_installment * (((1 + r) ** paid_k - 1) / r)
L["rem_principal"] = L.rem_principal.clip(lower=0)
g = L.groupby("customer_id")
rem = g.rem_principal.sum()
F["rem_principal_sum"] = rem
F["loan_vs_rem_principal"] = (loan / rem.replace(0, np.nan))
F["arrears_installments_now"] = g.A.sum()
# hari ke due berikutnya dari pinjaman aktif
ap = PA[PA.decision == "approved"].merge(A[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"}), on="customer_id")
months = (ap.cut.dt.year - ap.decision_datetime.dt.year) * 12 + (ap.cut.dt.month - ap.decision_datetime.dt.month)
nxt = ap.decision_datetime + pd.to_timedelta(0, unit="D")
nd = []
for mm in (0, 1):
    cand = ap.decision_datetime + (months + mm).apply(lambda x: pd.DateOffset(months=int(x)) if x >= 0 else pd.DateOffset(months=0))
    nd.append(cand)
ap["next_due"] = np.where(nd[0] > ap.cut, nd[0], nd[1])
ap["next_due"] = pd.to_datetime(ap.next_due)
ap["active"] = months < ap.term_months
ap["days_to_next_due"] = (ap.next_due - ap.cut).dt.total_seconds() / 86400
F["days_to_next_due_min"] = ap[ap.active].groupby("customer_id").days_to_next_due.min()
# ---------------------------------------------------------------- ide 9: kalender
A2 = A.set_index("customer_id")
F["app_day_of_month"] = A2.application_datetime.dt.day.astype(float)
F["dec_days_to_month_end"] = (A2.decision_datetime.dt.days_in_month - A2.decision_datetime.dt.day).astype(float)
F["dis_day_vs_payday25"] = ((A2.decision_datetime.dt.day - 25) % 31).astype(float)
print(table(ref, F).to_string(index=False))
