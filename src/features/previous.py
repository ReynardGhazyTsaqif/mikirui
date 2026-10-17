"""Fitur aplikasi sebelumnya (pa_)."""
import numpy as np
import pandas as pd

from ..io_utils import cutoffs, load
from .common import by_customer, days_between

WINDOWS = [30, 90, 180, 365]


def _load_pa(apps):
    pa = load("previous_applications").merge(
        apps[["customer_id", "decision_datetime", "loan_amount", "annual_interest_rate",
              "proposed_monthly_installment"]].rename(columns={
                  "decision_datetime": "cutoff", "loan_amount": "cur_amount",
                  "annual_interest_rate": "cur_rate",
                  "proposed_monthly_installment": "cur_inst"}),
        on="customer_id", how="inner")
    pa["age_days"] = days_between(pa["cutoff"], pa["decision_datetime"])
    pa["app_age_days"] = days_between(pa["cutoff"], pa["application_datetime"])
    return pa


def build_pa_window(apps):
    """E30: hitungan per jendela, recency per keputusan, kecepatan, pinjaman baru belum jatuh tempo."""
    pa = _load_pa(apps)
    agg = pd.DataFrame(index=pa["customer_id"].unique())
    for w in WINDOWS:
        sub = pa[pa["app_age_days"] <= w]
        ct = pd.crosstab(sub["customer_id"], sub["decision"])
        agg[f"w{w}_n_apps"] = sub.groupby("customer_id").size()
        for d in ["approved", "rejected", "cancelled"]:
            agg[f"w{w}_n_{d}"] = ct[d] if d in ct else 0
    wcols = [c for c in agg.columns if c.startswith("w")]
    agg[wcols] = agg[wcols].fillna(0)
    for d in ["approved", "rejected", "cancelled"]:
        agg[f"days_since_last_{d}"] = pa[pa["decision"] == d].groupby("customer_id")[
            "age_days"].min()
    agg["days_since_last_app"] = pa.groupby("customer_id")["app_age_days"].min()
    agg["hist_len_days"] = pa.groupby("customer_id")["app_age_days"].max()
    agg["apps_per_year"] = pa.groupby("customer_id").size() / (agg["hist_len_days"] / 365).clip(
        lower=1 / 12)
    pa_s = pa.sort_values(["customer_id", "application_datetime"])
    pa_s["gap"] = pa_s.groupby("customer_id")["app_age_days"].diff().abs()
    agg["app_gap_mean"] = pa_s.groupby("customer_id")["gap"].mean()
    agg["app_gap_min"] = pa_s.groupby("customer_id")["gap"].min()
    agg["process_days_mean"] = (pa["app_age_days"] - pa["age_days"]).groupby(
        pa["customer_id"]).mean()
    # Pinjaman cair <= 31 hari sebelum cutoff belum punya cicilan jatuh tempo (target 19,1%).
    rp_loans = load("repayment_history", usecols=["loan_id"])["loan_id"].unique()
    appr = pa[pa["decision"] == "approved"]
    fresh = appr[~appr["loan_id"].isin(rp_loans)]
    agg["n_fresh_loans"] = fresh.groupby("customer_id").size()
    agg["n_fresh_loans"] = agg["n_fresh_loans"].fillna(0)
    agg["fresh_loan_amount"] = fresh.groupby("customer_id")["approved_amount"].sum()
    agg["fresh_loan_inst"] = fresh.groupby("customer_id")["scheduled_monthly_installment"].sum()
    last = pa.sort_values("application_datetime").groupby("customer_id").last()
    agg["last_decision_rejected"] = (last["decision"] == "rejected").astype("float32")
    agg["last_decision_cancelled"] = (last["decision"] == "cancelled").astype("float32")
    return by_customer(agg, apps, "pa_")


def build_pa_burden(apps):
    """E31: beban cicilan aktif, nominal & bunga dibanding pinjaman saat ini."""
    pa = _load_pa(apps)
    appr = pa[pa["decision"] == "approved"].copy()
    appr["end"] = appr["decision_datetime"] + pd.to_timedelta(appr["term_months"] * 30.44, unit="D")
    appr["active"] = appr["end"] > appr["cutoff"]
    act = appr[appr["active"]]
    g = act.groupby("customer_id")
    agg = pd.DataFrame(index=pa["customer_id"].unique())
    agg["n_active_loans"] = g.size()
    agg["active_inst_sum"] = g["scheduled_monthly_installment"].sum()
    agg["active_amount_sum"] = g["approved_amount"].sum()
    agg["active_months_left_sum"] = (days_between(act["end"], act["cutoff"]) / 30.44).groupby(
        act["customer_id"]).sum()
    agg[["n_active_loans", "active_inst_sum", "active_amount_sum"]] = agg[
        ["n_active_loans", "active_inst_sum", "active_amount_sum"]].fillna(0)
    cur = pa.groupby("customer_id")[["cur_amount", "cur_rate", "cur_inst"]].first()
    ga = appr.groupby("customer_id")
    agg["amount_vs_max_approved"] = cur["cur_amount"] / ga["approved_amount"].max()
    agg["amount_vs_mean_requested"] = cur["cur_amount"] / pa.groupby("customer_id")[
        "requested_amount"].mean()
    last_appr = appr.sort_values("decision_datetime").groupby("customer_id").last()
    agg["amount_vs_last_approved"] = cur["cur_amount"] / last_appr["approved_amount"]
    agg["rate_minus_last"] = cur["cur_rate"] - last_appr["annual_interest_rate"]
    agg["rate_minus_mean"] = cur["cur_rate"] - ga["annual_interest_rate"].mean()
    agg["rate_prev_max"] = ga["annual_interest_rate"].max()
    agg["inst_vs_last"] = cur["cur_inst"] / last_appr["scheduled_monthly_installment"]
    agg["approved_share_amt"] = ga["approved_amount"].sum() / pa.groupby("customer_id")[
        "requested_amount"].sum()
    agg["rejected_amount_max"] = pa[pa["decision"] == "rejected"].groupby("customer_id")[
        "requested_amount"].max()
    agg["total_inst_with_current"] = agg["active_inst_sum"] + cur["cur_inst"]
    return by_customer(agg, apps, "pa_")
