"""Langkah 2f: apakah ada status 'stres' customer yang berubah menurut waktu dan dibagi lintas pinjaman?

Uji: untuk pasangan pinjaman berbeda milik customer yang sama, bandingkan
  P(miss di pinjaman 1 bulan m DAN miss di pinjaman 2 bulan m)        (bulan sama)
  P(miss di pinjaman 1 bulan m DAN miss di pinjaman 2 bulan m + lag)  (lag 1..12)
Jika hanya heterogenitas customer (p tetap), peluang bersama tidak bergantung lag.
Jika ada status bersama yang berubah, peluang bersama turun dengan lag.
Event lengkap: termasuk event setelah tenor selama masih ada tunggakan.
"""
import numpy as np
import pandas as pd

from common import apps, load

A = apps()
RP = load("repayment_history")
PA = load("previous_applications", usecols=["loan_id", "decision_datetime", "term_months"])
RP = RP.merge(PA, on="loan_id").merge(A[["customer_id", "decision_datetime", "target", "is_test"]].rename(
    columns={"decision_datetime": "cutoff"}), on="customer_id")
RP = RP.sort_values(["loan_id", "due_date"]).reset_index(drop=True)
RP["k"] = RP.groupby("loan_id").cumcount() + 1
paid = RP.amount_paid >= RP.amount_due
lp = RP.last_payment_datetime
RP["e_paid"] = np.where(paid, (lp.dt.year - RP.decision_datetime.dt.year) * 12 + (lp.dt.month - RP.decision_datetime.dt.month), np.nan)
# event anniversaries strictly before cutoff: j = 1..J, J = jumlah ulang bulan < cutoff
L = RP.groupby("loan_id").agg(customer_id=("customer_id", "first"), dis=("decision_datetime", "first"),
                              cutoff=("cutoff", "first"), n=("k", "max"), term=("term_months", "first"),
                              last_e=("e_paid", "max"), all_paid=("e_paid", lambda s: s.notna().all()),
                              target=("target", "first"), is_test=("is_test", "first"))
mdiff = (L.cutoff.dt.year - L.dis.dt.year) * 12 + (L.cutoff.dt.month - L.dis.dt.month)
# anniversary di bulan cutoff terjadi sebelum cutoff jika (hari, jam) dis < (hari, jam) cutoff
same_month_before = (L.dis.dt.day * 1440 + L.dis.dt.hour * 60 + L.dis.dt.minute) < (
    L.cutoff.dt.day * 1440 + L.cutoff.dt.hour * 60 + L.cutoff.dt.minute)
L["J_all"] = mdiff - 1 + same_month_before.astype(int)
print("cek n cicilan tercatat == min(term, J_all):", (L.n == np.minimum(L.term, L.J_all)).mean().round(5))
# event aktif: j <= term, atau j > term selama tunggakan > 0 (sampai lunas atau cutoff)
L["J"] = np.where(L.all_paid, np.maximum(L.n, L.last_e.fillna(0)), L.J_all).astype(int)
L["J"] = np.minimum(L.J, L.J_all)
idx = np.repeat(L.index.values, L.J.values)
j = np.concatenate([np.arange(1, x + 1) for x in L.J.values])
EV = pd.DataFrame({"loan_id": idx, "j": j})
X = RP[RP.e_paid.notna()].groupby(["loan_id", "e_paid"]).size().rename("X").reset_index().rename(columns={"e_paid": "j"})
X["j"] = X.j.astype(int)
EV = EV.merge(X, on=["loan_id", "j"], how="left")
EV["X"] = EV.X.fillna(0).astype(int)
EV = EV.merge(L[["customer_id", "dis", "cutoff", "term", "target", "is_test"]], left_on="loan_id", right_index=True)
EV["due_cum"] = np.minimum(EV.j, EV.term)
EV["C"] = EV.groupby("loan_id").X.cumsum()
EV["A"] = EV.due_cum - EV.C
EV["A_prev"] = EV.groupby("loan_id").A.shift().fillna(0).astype(int)
EV["due_now"] = EV.A_prev + (EV.j <= EV.term).astype(int)
EV = EV[EV.due_now > 0].copy()  # event tanpa kewajiban tidak dihitung
EV["miss"] = (EV.X == 0).astype("int8")
EV["cal"] = EV.dis.dt.year * 12 + EV.dis.dt.month - 1 + EV.j
EV["m_before"] = (EV.cutoff.dt.year * 12 + EV.cutoff.dt.month - 1) - EV.cal
print("A < 0:", (EV.A < 0).sum(), "| events:", len(EV), "| post-term events:", int((EV.j > EV.term).sum()))
print("P(X | A_prev) dengan event lengkap:\n", pd.crosstab(EV.A_prev.clip(upper=6), EV.X.clip(upper=4), normalize="index").round(3))
EV.to_parquet("cache_events_full.parquet")
# uji status bersama: pasangan pinjaman beda milik customer yang sama
cm = EV[["customer_id", "loan_id", "cal", "miss"]]
cust_loans = cm.groupby("customer_id").loan_id.nunique()
multi_c = cust_loans[cust_loans >= 2].index
cm = cm[cm.customer_id.isin(multi_c)]
res = []
for lag in [0, 1, 2, 3, 6, 9, 12]:
    b = cm.copy()
    b["cal"] = b.cal - lag
    m = cm.merge(b, on=["customer_id", "cal"], suffixes=("1", "2"))
    m = m[m.loan_id1 != m.loan_id2]
    res.append(dict(lag=lag, n=len(m), p_both=(m.miss1 & m.miss2).mean(), p1=m.miss1.mean(), p2=m.miss2.mean()))
r = pd.DataFrame(res)
r["ratio_vs_indep"] = r.p_both / (r.p1 * r.p2)
print("co-miss lintas pinjaman (customer sama) per lag bulan:\n", r.round(4).to_string(index=False))
# hal yang sama di dalam satu pinjaman (autokorelasi menurut lag)
res = []
for lag in [1, 2, 3, 6]:
    b = EV[["loan_id", "j", "miss"]].copy()
    b["j"] = b.j - lag
    m = EV[["loan_id", "j", "miss"]].merge(b, on=["loan_id", "j"], suffixes=("1", "2"))
    res.append(dict(lag=lag, n=len(m), p_both=(m.miss1 & m.miss2).mean(), p1=m.miss1.mean(), p2=m.miss2.mean()))
r = pd.DataFrame(res)
r["ratio_vs_indep"] = r.p_both / (r.p1 * r.p2)
print("autokorelasi miss dalam pinjaman:\n", r.round(4).to_string(index=False))
# miss rate per bulan-sebelum-cutoff, dipisah target (customer train)
t = EV[EV.is_test == 0]
mm = t.groupby([t.m_before.clip(upper=24), "target"]).miss.mean().unstack()
print("P(miss) per bulan sebelum keputusan (0 = bulan keputusan) per target:\n", mm.round(3).T.to_string())
