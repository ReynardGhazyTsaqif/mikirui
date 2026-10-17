"""Fitur riwayat cicilan (rp_). Sinyal terkuat di EDA.

Semantik kolom (diverifikasi di data):
- days_past_due     = DPD cicilan saat cutoff (0 jika sudah lunas)
- max_days_past_due = DPD terburuk cicilan itu (= keterlambatan bayar untuk cicilan lunas)
"""
import numpy as np
import pandas as pd

from ..io_utils import cutoffs, load
from .common import by_customer, days_between

WINDOWS = [90, 180, 365]


def _load_rp(apps):
    rp = load("repayment_history").merge(cutoffs(apps)[["customer_id", "cutoff"]],
                                         on="customer_id", how="inner")
    rp["age_days"] = days_between(rp["cutoff"], rp["due_date"])
    rp["unpaid"] = rp["amount_due"] - rp["amount_paid"]
    rp["paid_ratio"] = rp["amount_paid"] / rp["amount_due"]
    rp["pay_delay_days"] = days_between(rp["last_payment_datetime"], rp["due_date"])
    rp.loc[rp["unpaid"] > 0, "pay_delay_days"] = np.nan  # belum lunas: keterlambatan belum final
    for t in [1, 30, 60, 90]:
        rp[f"dpd{t}"] = (rp["max_days_past_due"] >= t).astype("int8")
    rp["late_now"] = (rp["days_past_due"] > 0).astype("int8")
    return rp


def build_rp_core(apps):
    """E10: DPD dan pembayaran all-time + kondisi saat cutoff."""
    rp = _load_rp(apps)
    g = rp.groupby("customer_id")
    agg = pd.DataFrame(index=g.size().index)
    for t in [1, 30, 60, 90]:
        agg[f"n_dpd{t}"] = g[f"dpd{t}"].sum()
        agg[f"share_dpd{t}"] = g[f"dpd{t}"].mean()
    agg["maxdpd_median"] = g["max_days_past_due"].median()
    agg["maxdpd_q75"] = g["max_days_past_due"].quantile(0.75)
    agg["maxdpd_sum"] = g["max_days_past_due"].sum()
    agg["n_late_now"] = g["late_now"].sum()
    agg["dpd_now_max"] = g["days_past_due"].max()
    agg["dpd_now_sum"] = g["days_past_due"].sum()
    agg["arrears_now"] = g["unpaid"].sum()
    agg["arrears_ratio"] = agg["arrears_now"] / g["amount_due"].sum()
    agg["delay_mean"] = g["pay_delay_days"].mean()
    agg["delay_max"] = g["pay_delay_days"].max()
    agg["delay_std"] = g["pay_delay_days"].std()
    agg["n_unpaid_inst"] = (rp["unpaid"] > 0).groupby(rp["customer_id"]).sum()
    agg["hist_len_days"] = g["age_days"].max()
    return by_customer(agg, apps, "rp_")


def build_rp_window(apps):
    """E11: jendela 90/180/365 hari (berdasarkan due_date) + recency kejadian buruk."""
    rp = _load_rp(apps)
    agg = pd.DataFrame(index=rp["customer_id"].unique())
    for w in WINDOWS:
        sub = rp[rp["age_days"] <= w]
        g = sub.groupby("customer_id")
        agg[f"w{w}_n"] = g.size()
        agg[f"w{w}_maxdpd_max"] = g["max_days_past_due"].max()
        agg[f"w{w}_maxdpd_mean"] = g["max_days_past_due"].mean()
        agg[f"w{w}_share_dpd1"] = g["dpd1"].mean()
        agg[f"w{w}_share_dpd30"] = g["dpd30"].mean()
        agg[f"w{w}_paid_ratio"] = g["amount_paid"].sum() / g["amount_due"].sum()
        agg[f"w{w}_delay_mean"] = g["pay_delay_days"].mean()
    for t in [1, 30, 90]:
        last = rp[rp[f"dpd{t}"] == 1].groupby("customer_id")["age_days"].min()
        agg[f"days_since_dpd{t}"] = last
    agg["days_since_last_payment"] = days_between(
        rp.groupby("customer_id")["cutoff"].first(),
        rp.groupby("customer_id")["last_payment_datetime"].max())
    agg["days_since_last_due"] = rp.groupby("customer_id")["age_days"].min()
    return by_customer(agg, apps, "rp_")


def build_rp_loan(apps):
    """E12: dua tingkat cicilan -> pinjaman -> customer. ever90 per pinjaman = analog target."""
    rp = _load_rp(apps)
    lg = rp.groupby(["customer_id", "loan_id"])
    loan = pd.DataFrame({
        "maxdpd": lg["max_days_past_due"].max(),
        "paid_ratio": lg["amount_paid"].sum() / lg["amount_due"].sum(),
        "n_inst": lg.size(),
        "first_due_age": lg["age_days"].max(),
        "last_due_age": lg["age_days"].min(),
        "share_dpd30": lg["dpd30"].mean(),
        "principal_proxy": lg["amount_due"].first(),
    }).reset_index()
    for t in [30, 60, 90]:
        loan[f"ever{t}"] = (loan["maxdpd"] >= t).astype("int8")
    g = loan.groupby("customer_id")
    agg = pd.DataFrame({"n_loans_obs": g.size()})
    for t in [30, 60, 90]:
        agg[f"n_loans_ever{t}"] = g[f"ever{t}"].sum()
        agg[f"share_loans_ever{t}"] = g[f"ever{t}"].mean()
    agg["loan_maxdpd_mean"] = g["maxdpd"].mean()
    agg["loan_maxdpd_min"] = g["maxdpd"].min()
    agg["loan_paid_ratio_min"] = g["paid_ratio"].min()
    agg["loan_share_dpd30_max"] = g["share_dpd30"].max()
    # Pinjaman paling baru (jatuh tempo pertama paling dekat ke cutoff).
    recent = loan.sort_values("first_due_age").groupby("customer_id").first()
    for c in ["maxdpd", "paid_ratio", "n_inst", "share_dpd30", "ever90"]:
        agg[f"recent_loan_{c}"] = recent[c]
    # Pinjaman ever90 paling baru: seberapa lama sejak kejadian default terakhir.
    agg["days_since_ever90_loan"] = loan[loan["ever90"] == 1].groupby("customer_id")[
        "last_due_age"].min()
    return by_customer(agg, apps, "rp_")


def _longest_streak(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def build_rp_trend(apps):
    """E13: arah perubahan perilaku bayar, DPD berbobot waktu, streak terlambat terpanjang."""
    rp = _load_rp(apps)
    rp["w"] = np.exp(-rp["age_days"] / 180)
    rp["w_maxdpd"] = rp["w"] * rp["max_days_past_due"]
    rp["w_dpd30"] = rp["w"] * rp["dpd30"]
    g = rp.groupby("customer_id")
    agg = pd.DataFrame({
        "wdpd_sum": g["w_maxdpd"].sum(),
        "wdpd_mean": g["w_maxdpd"].sum() / g["w"].sum(),
        "wdpd30_share": g["w_dpd30"].sum() / g["w"].sum(),
    })
    recent = rp[rp["age_days"] <= 180].groupby("customer_id")["max_days_past_due"]
    prior = rp[(rp["age_days"] > 180) & (rp["age_days"] <= 360)].groupby("customer_id")[
        "max_days_past_due"]
    agg["maxdpd_6m_minus_prev6m"] = recent.max() - prior.max()
    agg["meandpd_6m_minus_prev6m"] = recent.mean() - prior.mean()
    # Kemiringan max DPD terhadap umur cicilan (negatif umur = makin baru).
    x = -rp["age_days"]
    rp["_x"], rp["_xy"], rp["_xx"] = x, x * rp["max_days_past_due"], x * x
    s = rp.groupby("customer_id")[["_x", "_xy", "_xx", "max_days_past_due"]].sum()
    n = g.size()
    den = n * s["_xx"] - s["_x"] ** 2
    agg["maxdpd_slope"] = (n * s["_xy"] - s["_x"] * s["max_days_past_due"]) / den.replace(0, np.nan)
    rp = rp.sort_values(["customer_id", "due_date"])
    agg["longest_late_streak"] = rp.groupby("customer_id")["dpd1"].agg(_longest_streak)
    return by_customer(agg, apps, "rp_")
