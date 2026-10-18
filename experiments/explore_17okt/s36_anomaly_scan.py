"""Masukan user 18 Okt butir 2: data aneh / missing mungkin berpola. Pindai anomali per tabel (kecuali e-commerce,
lihat s37) dan uji tiap pola sebagai fitur per customer terhadap target dan residual OOF E202.

Pola yang dicari: digit/pembulatan nominal, detik/menit timestamp, inkonsistensi antar kolom, nilai di luar aturan
pembangkit (bayar bukan di ulang bulan due, tunggakan tidak konsisten), duplikat, missing per kolom.
Keluaran: tabel prevalensi + target rate + AUC + z residual (|z| > 3 = sinyal di luar model).
"""
import numpy as np
import pandas as pd

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
ref = reference("E202")
A = apps()
cut = A.set_index("customer_id").decision_datetime
F = {}


def add(name, s):
    F[name] = s.astype(float)


# ---------- aplikasi sekarang
a = A.set_index("customer_id")
add("app_sec", a.application_datetime.dt.second)
add("app_min", a.application_datetime.dt.minute)
add("app_sec0", a.application_datetime.dt.second == 0)
add("dec_min", a.decision_datetime.dt.minute)
add("dec_min0", a.decision_datetime.dt.minute == 0)
add("app_day", a.application_datetime.dt.day)
add("amt_mod1m", (a.loan_amount % 1_000_000) == 0)
add("amt_mod5m", (a.loan_amount % 5_000_000) == 0)
add("amt_last_digits", (a.loan_amount // 100_000) % 10)
r, n = a.annual_interest_rate / 12, a.term_months
ann = a.loan_amount * r / (1 - (1 + r) ** (-n))
add("inst_round_resid", a.proposed_monthly_installment - ann)
add("inst_mod1000", (a.proposed_monthly_installment % 1000) == 0)
# ---------- profil
p = load("customer_profiles").set_index("customer_id")
inc = p.declared_monthly_income
add("inc_mod100k", (inc % 100_000) == 0)
add("inc_mod10k", (inc % 10_000) == 0)
add("inc_mod1k", (inc % 1_000) == 0)
add("inc_last3", inc % 1000)
add("inc_digit_sum_mod9", inc.astype(np.int64).astype(str).str.split("").apply(lambda d: sum(int(x) for x in d if x)) % 9)
add("tenure_mod12", (p.employment_tenure_months % 12) == 0)
add("tenure_zero", p.employment_tenure_months == 0)
add("age_minus_tenure_y", p.age - p.employment_tenure_months / 12)
# ---------- aplikasi lama
pa = load("previous_applications")
add("pa_n_cancel", (pa.decision == "cancelled").groupby(pa.customer_id).sum())
for dcs in pa.decision.dropna().unique():
    add(f"pa_share_{dcs}", (pa.decision == dcs).groupby(pa.customer_id).mean())
appr = pa[pa.approved_amount.notna()].copy()
appr["cut"] = appr.approved_amount / appr.requested_amount
add("pa_cut_mean", appr.groupby("customer_id").cut.mean())
add("pa_cut_min", appr.groupby("customer_id").cut.min())
add("pa_cut_share", (appr.cut < 1).groupby(appr.customer_id).mean())
add("pa_req_mod1m", ((pa.requested_amount % 1_000_000) == 0).groupby(pa.customer_id).mean())
add("pa_appr_mod100k", ((appr.approved_amount % 100_000) == 0).groupby(appr.customer_id).mean())
add("pa_app_sec0", (pa.application_datetime.dt.second == 0).groupby(pa.customer_id).mean())
add("pa_dec_min0", (pa.decision_datetime.dt.minute == 0).groupby(pa.customer_id).mean())
add("pa_lag_h_mean", ((pa.decision_datetime - pa.application_datetime).dt.total_seconds() / 3600).groupby(pa.customer_id).mean())
dup = pa.duplicated(["customer_id", "application_datetime"], keep=False)
add("pa_dup_time", dup.groupby(pa.customer_id).sum())
r2 = appr.annual_interest_rate / 12
ann2 = appr.approved_amount * r2 / (1 - (1 + r2) ** (-appr.term_months))
add("pa_inst_resid_abs", (appr.scheduled_monthly_installment - ann2).abs().groupby(appr.customer_id).max())
gap = pa.sort_values(["customer_id", "application_datetime"]).groupby("customer_id").application_datetime.diff().dt.days
add("pa_gap_min_d", gap.groupby(pa.customer_id).min())
# ---------- cicilan
rp = load("repayment_history")
rp["cut"] = rp.customer_id.map(cut)
paid_any = rp.amount_paid > 0
add("rp_paid_no_dt", (paid_any & rp.last_payment_datetime.isna()).groupby(rp.customer_id).sum())
add("rp_dt_no_paid", (~paid_any & rp.last_payment_datetime.notna()).groupby(rp.customer_id).sum())
late = (rp.last_payment_datetime - rp.due_date).dt.total_seconds() / 86400
add("rp_paid_before_due", (late < -0.001).groupby(rp.customer_id).sum())
off = (rp.last_payment_datetime.dt.day != rp.due_date.dt.day) | (rp.last_payment_datetime.dt.hour != rp.due_date.dt.hour)
add("rp_pay_off_anniv", (off & rp.last_payment_datetime.notna()).groupby(rp.customer_id).sum())
add("rp_dpd_gt_max", (rp.days_past_due > rp.max_days_past_due).groupby(rp.customer_id).sum())
exp_dpd = np.where(rp.amount_paid >= rp.amount_due, 0, (rp.cut - rp.due_date).dt.days)
add("rp_dpd_mismatch", (np.abs(rp.days_past_due - exp_dpd) > 1).groupby(rp.customer_id).sum())
add("rp_due_mod1000", ((rp.amount_due % 1000) == 0).groupby(rp.customer_id).mean())
add("rp_paid_frac_part", ((rp.amount_paid > 0) & (rp.amount_paid < rp.amount_due)).groupby(rp.customer_id).mean())
add("rp_paid_mod1000", (((rp.amount_paid % 1000) == 0) & (rp.amount_paid > 0)).groupby(rp.customer_id).mean())
# ---------- biro (laporan aplikasi sekarang dan lama dipisah)
bu = load("bureau_observations")
bu["cur"] = bu.application_id.isin(A.application_id)
lim, ro = bu.revolving_credit_limit, bu.revolving_outstanding_balance
util_calc = ro / lim.where(lim > 0)
bu["util_mismatch"] = (bu.utilization.notna() & util_calc.notna() & ((bu.utilization - util_calc).abs() > 0.01))
bu["util_missing_with_lim"] = bu.utilization.isna() & (lim > 0)
bu["ro_gt_out"] = ro > bu.outstanding_balance
bu["ro_gt_lim"] = ro > lim
bu["acc0_bal"] = (bu.active_accounts == 0) & (bu.outstanding_balance > 0)
bu["delinq_gt_acc"] = bu.delinquent_accounts_last_12m > bu.active_accounts
bu["score_na"] = bu.score.isna()
bu["lag_d"] = (bu.available_datetime - bu.report_datetime).dt.total_seconds() / 86400
bu["rep_min0"] = bu.report_datetime.dt.minute == 0
bu["score_mod10"] = (bu.score % 10) == 0
for flag in ["util_mismatch", "util_missing_with_lim", "ro_gt_out", "ro_gt_lim", "acc0_bal", "delinq_gt_acc",
             "score_na", "rep_min0", "score_mod10"]:
    for lab, part in [("cur", bu[bu.cur]), ("prev", bu[~bu.cur])]:
        add(f"bu_{lab}_{flag}", part[flag].groupby(part.customer_id).mean())
add("bu_cur_lag_mean", bu[bu.cur].groupby("customer_id").lag_d.mean())
add("bu_cur_nprov", bu[bu.cur].groupby("customer_id").size())
# ---------- rekening
ab = load("account_behavior").sort_values(["customer_id", "month"])
ab["prev_close"] = ab.groupby("customer_id").closing_balance.shift()
ab["open_mismatch"] = ab.prev_close.notna() & (ab.opening_balance != ab.prev_close)
ab["flow_mismatch"] = (ab.closing_balance - ab.opening_balance - ab.inflows + ab.outflows).abs() > 1
ab["inc_gt_inflow"] = ab.income_like_credits > ab.inflows
ab["min_gt_avg"] = ab.minimum_balance > ab.average_balance
ab["neg_bal"] = (ab.minimum_balance < 0) | (ab.closing_balance < 0)
ab["inc_zero"] = ab.income_like_credits == 0
ab["gap"] = ab.groupby("customer_id").month.diff().dt.days > 31
for flag in ["open_mismatch", "flow_mismatch", "inc_gt_inflow", "min_gt_avg", "neg_bal", "inc_zero", "gap"]:
    add(f"ab_{flag}", ab[flag].groupby(ab.customer_id).mean())
add("ab_inc_cv", ab.groupby("customer_id").income_like_credits.std() / ab.groupby("customer_id").income_like_credits.mean())
add("ab_inc_vs_declared", ab.groupby("customer_id").income_like_credits.median() / inc)
T = table(ref, {k: v for k, v in F.items()})
prev = {k: float(ref.customer_id.map(v).notna().mean()) for k, v in F.items()}
T["cakupan"] = T.fitur.map(prev).round(3)
print(T.to_string(index=False))
# ringkasan prevalensi flag biner yang jarang (agar pola langka tidak terlewat oleh z)
print("\nprevalensi flag (porsi customer train dengan nilai > 0) dan target rate-nya:")
y = ref.target.values
for k, v in F.items():
    x = ref.customer_id.map(v).values
    m = np.nan_to_num(x) > 0
    if 0 < m.mean() < 0.2:
        print(f"  {k:<28} porsi {m.mean():.4f}  target {y[m].mean():.3f} vs lainnya {y[~m].mean():.3f}  resid {ref.r.values[m].mean():+.4f}")
