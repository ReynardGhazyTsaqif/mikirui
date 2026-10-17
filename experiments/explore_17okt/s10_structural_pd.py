"""Langkah 2j: PD struktural dari parameter rantai bayar per customer.

Model per customer: di A=0 lewat bayar dengan peluang q0; di A>=1 lewat bayar q1, kalau bayar: mengejar 1 cicilan
tambahan dengan peluang c. Default 12 bulan ~ A mencapai 3 pada event 1..9 (DPD 90 sebelum bulan 12).
Parameter: posterior mean Beta (prior method of moments). PD dihitung dengan DP 9 langkah dari A=0.
Juga dicek pada pinjaman lama dengan jendela lengkap: apakah label 'A>=3 di event 1..9' = maxDPD>=90 (<=275 hari)?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from resid import reference, table

pd.set_option("display.width", 220)
EV = pd.read_parquet("cache_events_full.parquet")
EV["at0"] = EV.A_prev == 0
g0 = EV[EV.at0].groupby("customer_id")
g1 = EV[~EV.at0].groupby("customer_id")
S = pd.DataFrame({"n0": g0.size(), "m0": g0.miss.sum()}).join(
    pd.DataFrame({"n1": g1.size(), "m1": g1.miss.sum(), "c1": (EV[~EV.at0].X >= 2).groupby(EV[~EV.at0].customer_id).sum()}),
    how="outer").fillna(0)
S["p1"] = S.n1 - S.m1  # event bayar saat menunggak


def mom_beta(k, n, min_n):
    m = n >= min_n
    r = k[m] / n[m]
    mu, var = r.mean(), r.var()
    # koreksi varians binomial rata-rata
    var_b = np.mean(mu * (1 - mu) / n[m])
    v = max(var - var_b, 1e-4)
    s = mu * (1 - mu) / v - 1
    return mu * s, (1 - mu) * s


a0, b0 = mom_beta(S.m0, S.n0, 6)
a1, b1 = mom_beta(S.m1, S.n1, 4)
ac, bc = mom_beta(S.c1, S.p1, 4)
print(f"prior q0 Beta({a0:.2f},{b0:.2f}) mean {a0/(a0+b0):.3f} | q1 Beta({a1:.2f},{b1:.2f}) mean {a1/(a1+b1):.3f} | "
      f"c Beta({ac:.2f},{bc:.2f}) mean {ac/(ac+bc):.3f}")
S["q0"] = (S.m0 + a0) / (S.n0 + a0 + b0)
S["q1"] = (S.m1 + a1) / (S.n1 + a1 + b1)
S["c"] = (S.c1 + ac) / (S.p1 + ac + bc)


def pd_dp(q0, q1, c, steps=9):
    """Peluang A mencapai 3 dalam `steps` event, mulai A=0."""
    p = np.zeros((len(q0), 4))
    p[:, 0] = 1
    for _ in range(steps):
        n = np.zeros_like(p)
        n[:, 3] = p[:, 3]
        n[:, 1] += p[:, 0] * q0
        n[:, 0] += p[:, 0] * (1 - q0)
        for a in (1, 2):
            n[:, a + 1] += p[:, a] * q1
            n[:, a] += p[:, a] * (1 - q1) * (1 - c)
            n[:, a - 1] += p[:, a] * (1 - q1) * c
        p = n
    return p[:, 3]


S["pd_struct"] = pd_dp(S.q0.values, S.q1.values, S.c.values)
S["pd_q0only"] = pd_dp(S.q0.values, S.q0.values, np.full(len(S), ac / (ac + bc)))
S["miss_post"] = pd.read_parquet("cache_cust_event_stats.parquet").miss_post.reindex(S.index)
ref = reference("E122x1")
print(table(ref, {c: S[c] for c in ["pd_struct", "pd_q0only", "miss_post", "q0", "q1", "c"]}).to_string(index=False))
# kalibrasi: target rate per desil pd_struct (customer train)
tt = ref.set_index("customer_id").join(S[["pd_struct"]], how="inner")
tt["dec"] = pd.qcut(tt.pd_struct, 10, labels=False)
print("desil pd_struct: rata pd, target rate, rata p model")
print(tt.groupby("dec").agg(pd=("pd_struct", "mean"), target=("target", "mean"), p_model=("p", "mean")).round(4).T.to_string())
# AUC model di segmen punya riwayat vs pd_struct
print("AUC E122x1 pada customer punya riwayat: %.4f | AUC pd_struct: %.4f" % (
    roc_auc_score(tt.target, tt.p), roc_auc_score(tt.target, tt.pd_struct)))
S.to_parquet("cache_struct.parquet")
