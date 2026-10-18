"""H24 (mentor ide 16), putaran CAAFE 1: fitur domain kredit yang belum ada di 528 kolom + rp_ev + ab_nonin.

Saring sama seperti s38: z residual OOF E202 di fold 0-2 (seleksi) dan fold 3-4 (konfirmasi); lolos jika |z_sel| >= 3
dan z_val searah dengan |z_val| >= 2 (lebih longgar dari s38 karena kandidat sedikit dan berbasis domain).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import apps, load
from resid import reference

ref = reference("E202")
fold, r = ref.fold.values, ref.r.values
A = apps().set_index("customer_id")
cut = A.decision_datetime
F = {}
pa = load("previous_applications")
prof = load("customer_profiles").set_index("customer_id")
inc = prof.declared_monthly_income
# 1 guncangan cicilan: cicilan baru / cicilan terjadwal terbesar sebelumnya
F["payment_shock_vs_max"] = A.proposed_monthly_installment / pa.groupby("customer_id").scheduled_monthly_installment.max()
# 2 jarak hari dari lunasnya pinjaman terakhir (pembayaran terakhir pinjaman yang semua cicilannya lunas) ke aplikasi
rp = load("repayment_history", usecols=["loan_id", "customer_id", "due_date", "amount_due", "amount_paid", "last_payment_datetime"])
pa_l = pa.dropna(subset=["loan_id"])[["loan_id", "term_months"]]
L = rp.groupby("loan_id").agg(cid=("customer_id", "first"), n=("due_date", "size"),
                              paid_all=("amount_paid", lambda s: 0), last_pay=("last_payment_datetime", "max"))
full = rp.assign(ok=rp.amount_paid >= rp.amount_due).groupby("loan_id").ok.all()
L = L.join(pa_l.set_index("loan_id")).join(full.rename("all_paid"))
done = L[(L.n >= L.term_months) & L.all_paid]
F["days_since_loan_paidoff"] = (A.application_datetime - done.groupby("cid").last_pay.max().reindex(A.index)).dt.days
# 3 porsi tenor berjalan yang sudah lewat, minimum atas pinjaman aktif (baru mulai pinjaman lain = penumpukan)
act = L[(L.n < L.term_months)]
F["min_term_elapsed_active"] = (act.n / act.term_months).groupby(act.cid).min()
F["n_active_young_loans"] = ((act.n / act.term_months) < 0.34).groupby(act.cid).sum()
# 4 inquiry biro di luar aplikasi internal 6 bulan terakhir; 5 tunggakan biro yang tak terlihat internal
bu = load("bureau_observations")
cur = bu[bu.application_id.isin(A.application_id)]
inq = cur.groupby("customer_id").inquiries_last_6m.max()
pa_t = pa.merge(cut.rename("cut"), left_on="customer_id", right_index=True)
n6 = ((pa_t.cut - pa_t.application_datetime).dt.days <= 183).groupby(pa_t.customer_id).sum().reindex(A.index).fillna(0)
F["ext_inquiries_6m"] = inq - n6
rp_t = rp.merge(cut.rename("cut"), left_on="customer_id", right_index=True)
late30 = ((rp_t.last_payment_datetime.fillna(rp_t.cut) - rp_t.due_date).dt.days >= 30) & ((rp_t.cut - rp_t.due_date).dt.days <= 365)
int_del = late30.groupby(rp_t.loan_id).any().groupby(L.cid).sum().reindex(A.index).fillna(0)
F["ext_delinq_minus_internal"] = cur.groupby("customer_id").delinquent_accounts_last_12m.max() - int_del
# 6 runway saldo (rata-rata saldo / rata-rata outflow) dan 7 belanja online / outflow
ab = load("account_behavior")
g = ab.groupby("customer_id")
F["runway_months"] = g.average_balance.mean() / g.outflows.mean().replace(0, np.nan)
F["runway_months_last3"] = ab[ab.groupby("customer_id").cumcount(ascending=False) < 3].groupby("customer_id").pipe(
    lambda h: h.average_balance.mean() / h.outflows.mean().replace(0, np.nan))
od = pd.read_parquet("cache_ec_orders.parquet", columns=["customer_id", "t", "rec_h", "total", "pm", "st"])
od_ok = od[od.st.astype(str) != "cancelled"]
spend_m = od_ok.groupby("customer_id").total.sum() / 24.0
F["ec_spend_vs_outflow"] = spend_m / g.outflows.mean()
# 8 ketergantungan kartu kredit x utilisasi revolving biro
cc = (od.pm.astype(str) == "credit_card").groupby(od.customer_id).mean()
F["cc_share_x_util"] = cc * cur.groupby("customer_id").utilization.mean().fillna(0)
# 9 usia saat aplikasi kredit pertama
first = pa.groupby("customer_id").application_datetime.min()
F["age_at_first_credit"] = prof.age - (cut - first.reindex(cut.index)).dt.days / 365.25
# 10 DTI total dengan estimasi cicilan utang eksternal (3% saldo biro + 5% saldo revolving)
act_inst = pa.dropna(subset=["loan_id"]).set_index("loan_id").scheduled_monthly_installment.reindex(act.index).groupby(act.cid).sum()
ext = cur.groupby("customer_id").outstanding_balance.mean() * 0.03 + cur.groupby("customer_id").revolving_outstanding_balance.mean() * 0.05
F["dti_total_est"] = (A.proposed_monthly_installment + act_inst.reindex(A.index).fillna(0) + ext) / inc
# 11 nominal relatif median per tujuan dan per tier
F["amount_vs_purpose_median"] = A.loan_amount / A.groupby("loan_purpose", observed=True).loan_amount.transform("median")
F["amount_vs_tier_median"] = A.loan_amount / A.groupby("annual_interest_rate").loan_amount.transform("median")
# 12 total bunga terhadap income tahunan
F["interest_to_annual_income"] = (A.proposed_monthly_installment * A.term_months - A.loan_amount) / (12 * inc)
# 13 beruntun penolakan terakhir sebelum aplikasi sekarang
pa_s = pa.sort_values(["customer_id", "application_datetime"])
rej = (pa_s.decision == "rejected").astype(int)
streak = rej.groupby(pa_s.customer_id).apply(lambda s: int((s[::-1].cumprod()).sum()))
F["last_rejection_streak"] = streak
# 14 porsi order malam (22-05) 7 hari terakhir vs seluruh riwayat
night = od.t.dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5])
w7 = od.rec_h <= 24 * 7
F["night_share_7d_minus_all"] = (night & w7).groupby(od.customer_id).sum() / w7.groupby(od.customer_id).sum().replace(0, np.nan) - night.groupby(od.customer_id).mean()
sel, val = np.isin(fold, [0, 1, 2]), np.isin(fold, [3, 4])
rows = []
y = ref.target.values
for k, s in F.items():
    x = ref.customer_id.map(s).astype(float).values
    zz = []
    for msk in (sel, val):
        ok = msk & np.isfinite(x)
        zz.append(np.corrcoef(rankdata(x[ok]), r[ok])[0, 1] * np.sqrt(ok.sum()) if ok.sum() > 1000 and np.nanstd(x[ok]) > 0 else np.nan)
    ok = np.isfinite(x)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[ok], x[ok]) if ok.sum() > 1000 else np.nan
    rows.append((k, round(max(auc, 1 - auc), 4), round(ok.mean(), 3), round(zz[0], 2), round(zz[1], 2)))
T = pd.DataFrame(rows, columns=["fitur", "auc", "cakupan", "z_sel", "z_val"])
T["lolos"] = (T.z_sel.abs() >= 3) & (np.sign(T.z_sel) == np.sign(T.z_val)) & (T.z_val.abs() >= 2)
print(T.sort_values("z_sel", key=np.abs, ascending=False).to_string(index=False))
