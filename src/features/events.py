"""Fitur rantai bayar bulanan level event (rp_ev_), H01 eksplorasi 17 Okt.

Pembangkit cicilan (experiments/explore_17okt s02/s05/s06): pembayaran hanya terjadi tepat di ulang bulan due
(tanggal-jam pencairan + j bulan), dialokasikan FIFO ke cicilan tertua, tidak pernah dibayar di muka, 0-3 cicilan
per event. Event j = 1..J dengan J = jumlah ulang bulan pencairan yang < cutoff (= decision_datetime aplikasi
sekarang); setelah tenor, event masih dihitung selama ada tunggakan. X_j = jumlah cicilan yang lunas penuh di event j,
A_j = min(j, tenor) - sum X (tunggakan setelah event j). Tidak memakai target; prior Beta dari semua customer.
"""
import numpy as np
import pandas as pd

from ..io_utils import cutoffs, load
from .common import by_customer


def _events(apps):
    rp = load("repayment_history", usecols=["loan_id", "customer_id", "due_date", "amount_due", "amount_paid",
                                            "last_payment_datetime"])
    pa = load("previous_applications", usecols=["loan_id", "decision_datetime", "term_months"]).dropna(subset=["loan_id"])
    rp = rp.merge(pa, on="loan_id").merge(cutoffs(apps)[["customer_id", "cutoff"]], on="customer_id")
    rp = rp.sort_values(["loan_id", "due_date"], kind="stable").reset_index(drop=True)
    rp["k"] = rp.groupby("loan_id").cumcount() + 1
    paid = rp["amount_paid"] >= rp["amount_due"]
    lp, dis = rp["last_payment_datetime"], rp["decision_datetime"]
    rp["e_paid"] = np.where(paid, (lp.dt.year - dis.dt.year) * 12 + (lp.dt.month - dis.dt.month), np.nan)
    L = rp.groupby("loan_id", sort=True).agg(customer_id=("customer_id", "first"), dis=("decision_datetime", "first"),
                                             cutoff=("cutoff", "first"), n=("k", "max"), term=("term_months", "first"),
                                             last_e=("e_paid", "max"), n_unpaid=("e_paid", lambda s: s.isna().sum()))
    mdiff = (L.cutoff.dt.year - L.dis.dt.year) * 12 + (L.cutoff.dt.month - L.dis.dt.month)
    before = (L.dis.dt.day * 1440 + L.dis.dt.hour * 60 + L.dis.dt.minute) < (
        L.cutoff.dt.day * 1440 + L.cutoff.dt.hour * 60 + L.cutoff.dt.minute)
    j_all = (mdiff - 1 + before.astype(int)).clip(lower=0)
    J = np.where(L.n_unpaid == 0, np.maximum(L.n, L.last_e.fillna(0)), j_all).astype(int)
    L["J"] = np.minimum(J, j_all.values)
    L = L[L.J > 0]
    ev = pd.DataFrame({"loan_id": np.repeat(L.index.values, L.J.values),
                       "j": np.concatenate([np.arange(1, x + 1) for x in L.J.values])})
    x = rp[rp.e_paid.notna()].groupby(["loan_id", "e_paid"]).size().rename("X").reset_index()
    x["j"] = x.e_paid.astype(int)
    ev = ev.merge(x[["loan_id", "j", "X"]], on=["loan_id", "j"], how="left")
    ev["X"] = ev["X"].fillna(0).astype(int)
    ev = ev.merge(L[["customer_id", "dis", "cutoff", "term"]], left_on="loan_id", right_index=True)
    ev["A"] = np.minimum(ev.j, ev.term) - ev.groupby("loan_id").X.cumsum()
    ev["A_prev"] = ev.groupby("loan_id").A.shift().fillna(0).astype(int)
    ev = ev[ev.A_prev + (ev.j <= ev.term).astype(int) > 0].copy()  # event tanpa kewajiban tidak dihitung
    ev["miss"] = (ev.X == 0).astype("int8")
    cal = ev.dis.dt.year * 12 + ev.dis.dt.month - 1 + ev.j
    ev["m_before"] = (ev.cutoff.dt.year * 12 + ev.cutoff.dt.month - 1) - cal
    return ev


def _beta_prior(k, n, min_n):
    """Prior Beta method of moments (varians dikoreksi binomial) dari customer dengan n >= min_n."""
    m = n >= min_n
    r = k[m] / n[m]
    mu = r.mean()
    v = max(r.var() - np.mean(mu * (1 - mu) / n[m]), 1e-4)
    s = mu * (1 - mu) / v - 1
    return mu * s, (1 - mu) * s


def _pd_dp(q0, q1, c, steps=9):
    """P(tunggakan mencapai 3 dalam `steps` event) dari A=0; rantai: lewat bayar q, bayar dan mengejar c."""
    p = np.zeros((len(q0), 4))
    p[:, 0] = 1
    for _ in range(steps):
        nx = np.zeros_like(p)
        nx[:, 3] = p[:, 3]
        nx[:, 1] += p[:, 0] * q0
        nx[:, 0] += p[:, 0] * (1 - q0)
        for a in (1, 2):
            nx[:, a + 1] += p[:, a] * q1
            nx[:, a] += p[:, a] * (1 - q1) * (1 - c)
            nx[:, a - 1] += p[:, a] * (1 - q1) * c
        p = nx
    return p[:, 3]


def build_rp_events(apps):
    """H01: statistik rantai bayar bulanan per customer (tanpa target)."""
    ev = _events(apps)
    at0 = ev.A_prev == 0
    g = ev.groupby("customer_id")
    g0, g1 = ev[at0].groupby("customer_id"), ev[~at0].groupby("customer_id")
    S = pd.DataFrame({"n_ev": g.size(), "n_miss": g.miss.sum()})
    S = S.join(pd.DataFrame({"n0": g0.size(), "m0": g0.miss.sum()}), how="left")
    paid1 = ev[~at0 & (ev.X >= 1)]
    S = S.join(pd.DataFrame({"n1": g1.size(), "m1": g1.miss.sum()}), how="left")
    S = S.join(pd.DataFrame({"p1": paid1.groupby("customer_id").size(),
                             "c1": (paid1.X >= 2).groupby(paid1.customer_id).sum()}), how="left")
    S = S.fillna(0)
    a0, b0 = _beta_prior(S.m0.values, S.n0.values, 6)
    a1, b1 = _beta_prior(S.m1.values, S.n1.values, 4)
    am, bm = _beta_prior(S.n_miss.values, S.n_ev.values, 6)
    out = pd.DataFrame(index=S.index)
    out["n_ev"], out["n_ev_at0"], out["n_ev_behind"] = S.n_ev, S.n0, S.n1
    out["miss_post"] = (S.n_miss + am) / (S.n_ev + am + bm)
    out["q0_post"] = (S.m0 + a0) / (S.n0 + a0 + b0)
    out["q1_post"] = (S.m1 + a1) / (S.n1 + a1 + b1)
    out["catch_rate"] = (S.c1 / S.p1).where(S.p1 > 0)
    # laju mengejar dengan prior lemah (rata-rata populasi, kekuatan 4 event) supaya tetap bervariasi
    c_mu = S.c1.sum() / max(S.p1.sum(), 1)
    out["catch_post"] = (S.c1 + 4 * c_mu) / (S.p1 + 4)
    out["pd_struct"] = _pd_dp(out.q0_post.values, out.q1_post.values, out.catch_post.values)
    out["max_arrears"] = g.A.max()
    out["months_since_last_miss"] = ev[ev.miss == 1].groupby("customer_id").m_before.min()
    out["n_post_term_ev"] = (ev.j > ev.term).groupby(ev.customer_id).sum()
    for w in (3, 6, 12):
        e = ev[ev.m_before < w]
        out[f"miss_rate_{w}m"] = e.groupby("customer_id").miss.mean()
        out[f"n_ev_{w}m"] = e.groupby("customer_id").size()
    return by_customer(out, apps, "rp_ev_")
