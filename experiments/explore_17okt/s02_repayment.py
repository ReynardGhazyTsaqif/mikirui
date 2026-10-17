"""Langkah 2b: struktur baris repayment_history dan hubungannya dengan pinjaman lama dan cutoff."""
import numpy as np
import pandas as pd

from common import apps, load

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

A = apps()
cut = A[["customer_id", "decision_datetime", "target", "is_test"]].rename(columns={"decision_datetime": "cutoff"})
RP = load("repayment_history")
PA = load("previous_applications")
print("== repayment", RP.shape)
print(RP.head(14).to_string())
RP = RP.merge(cut, on="customer_id").merge(
    PA[["loan_id", "decision_datetime", "approved_amount", "term_months", "annual_interest_rate",
        "scheduled_monthly_installment"]], on="loan_id", how="left")
RP = RP.sort_values(["loan_id", "due_date"]).reset_index(drop=True)
RP["k"] = RP.groupby("loan_id").cumcount() + 1
RP["n_inst"] = RP.groupby("loan_id").k.transform("max")
print("loan tanpa induk:", RP.decision_datetime.isna().sum())
# jadwal jatuh tempo: due_k - disbursement
RP["due_off_d"] = (RP.due_date - RP.decision_datetime).dt.total_seconds() / 86400
print("offset due pertama (hari) quantiles:", RP[RP.k == 1].due_off_d.quantile([0, .01, .5, .99, 1]).round(2).tolist())
print("jam due_date:", RP.due_date.dt.hour.value_counts().head(5).to_dict(), "| hari dlm bulan due vs hari disbursement sama:",
      (RP.due_date.dt.day == RP.decision_datetime.dt.day).mean().round(4))
d = RP.groupby("loan_id").due_date.diff().dt.days
print("jarak antar due (hari):", d.value_counts().head(8).to_dict())
# amount_due vs jadwal
RP["due_minus_sched"] = RP.amount_due - RP.scheduled_monthly_installment
print("amount_due - scheduled (bukan cicilan terakhir):", RP.loc[RP.k < RP.term_months, "due_minus_sched"].describe().round(2).to_dict())
print("amount_due - scheduled (cicilan terakhir):", RP.loc[RP.k == RP.term_months, "due_minus_sched"].describe().round(2).to_dict())
# jumlah cicilan yang tercatat vs yang seharusnya sudah jatuh tempo sebelum cutoff
exp_n = np.floor(((RP.cutoff - RP.decision_datetime).dt.days) / 30.44).clip(upper=RP.term_months)
L = RP.groupby("loan_id").agg(n=("k", "max"), term=("term_months", "first"), dis=("decision_datetime", "first"),
                              cutoff=("cutoff", "first"), last_due=("due_date", "max"))
L["months_elapsed"] = (L.cutoff - L.dis).dt.days / 30.44
L["all_due_before_cut"] = True
print("n cicilan tercatat == min(term, cicilan jatuh tempo < cutoff)?")
# hitung jadwal: due_k = dis + k bulan (cek dengan offset)
L["n_expected"] = np.minimum(L.term, np.floor(L.months_elapsed + 1e-9)).astype(int)
print((L.n - L.n_expected).value_counts().head(10).to_dict())
# semantik DPD
RP["paid_full"] = RP.amount_paid >= RP.amount_due
RP["age_d"] = (RP.cutoff - RP.due_date).dt.total_seconds() / 86400
RP["delay_d"] = (RP.last_payment_datetime - RP.due_date).dt.total_seconds() / 86400
print("paid_full share:", RP.paid_full.mean().round(4), "| amount_paid==0:", (RP.amount_paid == 0).mean().round(4),
      "| partial:", ((RP.amount_paid > 0) & ~RP.paid_full).mean().round(4), "| overpaid:", (RP.amount_paid > RP.amount_due).mean())
u = RP[~RP.paid_full]
print("unpaid: dpd == floor(age)?", (u.days_past_due == np.floor(u.age_d)).mean().round(4), "| dpd==ceil:", (u.days_past_due == np.ceil(u.age_d)).mean().round(4))
print("unpaid: max_dpd == dpd?", (u.max_days_past_due == u.days_past_due).mean().round(4))
p = RP[RP.paid_full]
print("paid: dpd==0?", (p.days_past_due == 0).mean().round(4))
dd = np.ceil(p.delay_d.clip(lower=0))
print("paid: max_dpd == ceil(delay)+?:", pd.Series(p.max_days_past_due - dd).value_counts().head(6).to_dict())
print("paid: max_dpd vs floor(delay):", pd.Series(p.max_days_past_due - np.floor(p.delay_d.clip(lower=0))).value_counts().head(6).to_dict())
print("delay (hari) quantiles bayar penuh:", p.delay_d.quantile([0, .01, .1, .25, .5, .75, .9, .99, 1]).round(2).tolist())
print("max_dpd value counts top:", RP.max_days_past_due.value_counts().head(25).to_dict())
print("jam last_payment:", RP.last_payment_datetime.dt.hour.value_counts().sort_index().to_dict())
print("hari last_payment - due (bayar sebelum due, hari ke-):", np.floor(-p.delay_d[p.delay_d < 0]).value_counts().head(10).to_dict())
# parsial: pola amount_paid/amount_due
pp = RP[(RP.amount_paid > 0) & ~RP.paid_full]
print("rasio bayar parsial quantiles:", (pp.amount_paid / pp.amount_due).quantile([0, .1, .25, .5, .75, .9, 1]).round(3).tolist())
print("contoh pinjaman lengkap (12 cicilan) dengan DPD:")
ex = RP[RP.loan_id.isin(RP[(RP.max_days_past_due >= 90)].loan_id.drop_duplicates().sample(3, random_state=1))]
print(ex[["loan_id", "k", "due_date", "amount_due", "amount_paid", "last_payment_datetime", "days_past_due", "max_days_past_due", "cutoff"]].to_string())
RP[["loan_id", "customer_id", "k", "n_inst", "term_months", "due_off_d", "age_d", "delay_d", "paid_full", "amount_due", "amount_paid",
    "days_past_due", "max_days_past_due"]].to_parquet("cache_rp_rows.parquet")
