"""Langkah 2e: rekonstruksi proses bayar bulanan per pinjaman.

Temuan s02: pembayaran hanya terjadi pada tanggal-jam ulang bulan jatuh tempo, alokasi FIFO ke cicilan tertua.
Event j (j = 1..n) = due date cicilan ke-j (n = cicilan yang sudah jatuh tempo sebelum cutoff).
X_j = jumlah cicilan yang lunas pada event j; karena e_k >= k (tidak ada bayar dimuka),
tunggakan setelah event j: A_j = j - sum_{i<=j} X_i.
"""
import numpy as np
import pandas as pd

from common import apps, load

pd.set_option("display.width", 220)
A = apps()
RP = load("repayment_history")
PA = load("previous_applications", usecols=["loan_id", "decision_datetime", "term_months"])
RP = RP.merge(PA, on="loan_id").merge(A[["customer_id", "decision_datetime", "target", "is_test"]].rename(
    columns={"decision_datetime": "cutoff"}), on="customer_id")
RP = RP.sort_values(["loan_id", "due_date"]).reset_index(drop=True)
RP["k"] = RP.groupby("loan_id").cumcount() + 1
paid = RP.amount_paid >= RP.amount_due
lp = RP.last_payment_datetime
mi = (lp.dt.year - RP.decision_datetime.dt.year) * 12 + (lp.dt.month - RP.decision_datetime.dt.month)
RP["e_paid"] = np.where(paid, mi, np.nan)
ok = RP.e_paid.notna()
print("e_paid >= k:", (RP.e_paid[ok] >= RP.k[ok]).mean(), "| FIFO:",
      (RP[ok].groupby("loan_id").e_paid.diff().dropna() >= 0).mean().round(5))
print("partial (0<paid<due):", ((RP.amount_paid > 0) & ~paid).sum(), "| partial hanya di cicilan tertua yang belum lunas?")
# skeleton event = baris cicilan (loan, k) -> (loan, j=k)
X = RP[ok].groupby(["loan_id", "e_paid"]).size().rename("X").reset_index().rename(columns={"e_paid": "j"})
X["j"] = X.j.astype(int)
EV = RP[["loan_id", "customer_id", "k", "decision_datetime", "cutoff", "term_months", "target", "is_test"]].rename(columns={"k": "j"})
EV = EV.merge(X, on=["loan_id", "j"], how="left")
EV["X"] = EV.X.fillna(0).astype(int)
EV["C"] = EV.groupby("loan_id").X.cumsum()
EV["A"] = EV.j - EV.C
EV["A_prev"] = EV.groupby("loan_id").A.shift().fillna(0).astype(int)
print("pembayaran pada event > n (setelah event terakhir yang tercatat):",
      int(RP[ok].e_paid.gt(RP[ok].groupby("loan_id").k.transform("max")).sum()))
print("A < 0:", (EV.A < 0).sum())
print("events:", len(EV))
ct = pd.crosstab(EV.A_prev.clip(upper=6), EV.X.clip(upper=7), normalize="index")
print("P(X | A_prev):\n", ct.round(3))
print("n event per A_prev:", EV.A_prev.clip(upper=6).value_counts().sort_index().to_dict())
EV["due_now"] = EV.A_prev + 1  # jumlah cicilan jatuh tempo pada event
EV["miss"] = (EV.X == 0).astype("int8")
EV["full_catch"] = (EV.X >= EV.due_now).astype("int8")
EV["cal"] = EV.decision_datetime.dt.year * 12 + EV.decision_datetime.dt.month - 1 + EV.j  # bulan kalender event
EV["cut_cal"] = EV.cutoff.dt.year * 12 + EV.cutoff.dt.month - 1
EV["m_before"] = EV.cut_cal - EV.cal  # 0 = bulan keputusan, 1 = bulan sebelumnya
EV.to_parquet("cache_events.parquet")
print("m_before q:", EV.m_before.quantile([0, .01, .5, .99, 1]).tolist())
# ketergantungan antarpinjaman pada bulan kalender sama
cm = EV.groupby(["customer_id", "cal"]).agg(n=("loan_id", "size"), ms=("miss", "sum")).reset_index()
m2 = cm[cm.n == 2]
pm = EV.miss.mean()
print("P(miss) event: %.4f | bulan dgn 2 pinjaman: %d, P(keduanya miss) %.4f, P(satu) %.4f, P(tidak ada) %.4f" % (
    pm, len(m2), (m2.ms == 2).mean(), (m2.ms == 1).mean(), (m2.ms == 0).mean()))
q = (m2.ms.mean() / 2)
print("  jika independen dengan p=%.4f: P(keduanya)=%.4f P(satu)=%.4f" % (q, q * q, 2 * q * (1 - q)))
# ketergantungan pada customer yang sama, beda waktu: korelasi miss rate paruh pertama vs paruh kedua riwayat
cust = EV.groupby("customer_id").agg(n=("miss", "size"), miss=("miss", "mean"), target=("target", "first"), is_test=("is_test", "first"))
print("miss rate per customer q:", cust.miss.quantile([.1, .25, .5, .75, .9, .95, .99]).round(3).tolist())
# autokorelasi: miss pada event j vs miss pada event j-1 (pinjaman sama)
EV["miss_prev"] = EV.groupby("loan_id").miss.shift()
e2 = EV.dropna(subset=["miss_prev"])
print("P(miss | miss_prev=1) %.4f vs P(miss | miss_prev=0) %.4f" % (e2[e2.miss_prev == 1].miss.mean(), e2[e2.miss_prev == 0].miss.mean()))
