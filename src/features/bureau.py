"""Fitur biro kredit: bu_ (laporan aplikasi saat ini), bp_ (laporan previous applications).

Distribusi skor ketiga provider identik, jadi tidak ada normalisasi per provider.
"""
import numpy as np
import pandas as pd

from ..io_utils import cutoffs, load
from .common import by_customer, days_between, finalize


def _split(apps):
    bo = load("bureau_observations")
    cur_mask = bo["application_id"].isin(apps["application_id"])
    cur = bo[cur_mask].merge(cutoffs(apps)[["application_id", "cutoff"]], on="application_id")
    prev = bo[~cur_mask].merge(cutoffs(apps)[["customer_id", "cutoff"]], on="customer_id")
    for d in (cur, prev):
        d["report_age_days"] = days_between(d["cutoff"], d["report_datetime"])
        d["avail_age_days"] = days_between(d["cutoff"], d["available_datetime"])
    return cur, prev


def build_bu_extra(apps):
    """E20: utilisasi agregat, sebaran skor antar provider, laporan paling segar."""
    cur, _ = _split(apps)
    g = cur.groupby("application_id")
    agg = pd.DataFrame({
        "util_total": g["revolving_outstanding_balance"].sum()
        / g["revolving_credit_limit"].sum().replace(0, np.nan),
        "has_revolving": (g["revolving_credit_limit"].max() > 0).astype("int8"),
        "score_range": g["score"].max() - g["score"].min(),
        "outstanding_sum": g["outstanding_balance"].sum(),
        "revolving_headroom": g["revolving_credit_limit"].max()
        - g["revolving_outstanding_balance"].max(),
        "accounts_sum": g["active_accounts"].sum(),
        "inquiries_sum": g["inquiries_last_6m"].sum(),
        "delinq_sum": g["delinquent_accounts_last_12m"].sum(),
        "n_delinq_providers": (cur["delinquent_accounts_last_12m"] > 0).groupby(
            cur["application_id"]).sum(),
        "avail_age_min": g["avail_age_days"].min(),
    })
    fresh = cur.sort_values("report_age_days").groupby("application_id").first()
    for c in ["score", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m",
              "outstanding_balance", "report_age_days"]:
        agg[f"fresh_{c}"] = fresh[c]
    # Skor dengan bobot kesegaran laporan.
    w = np.exp(-cur["report_age_days"] / 60)
    sw = (cur["score"] * w).groupby(cur["application_id"]).sum()
    ww = w.where(cur["score"].notna(), 0).groupby(cur["application_id"]).sum()
    agg["score_fresh_weighted"] = sw / ww.replace(0, np.nan)
    return finalize(agg.reindex(apps["application_id"]), "bu_")


def build_bp_trend(apps):
    """E21: perubahan skor/utang/inquiry dari laporan previous applications ke laporan saat ini."""
    cur, prev = _split(apps)
    cur = cur.merge(apps[["application_id", "customer_id"]], on=["application_id", "customer_id"])
    num = ["score", "outstanding_balance", "inquiries_last_6m", "delinquent_accounts_last_12m",
           "active_accounts", "utilization"]
    # Laporan sebelumnya terakhir per (customer, provider) -> delta terhadap provider yang sama.
    last_prev = prev.sort_values("report_datetime").groupby(
        ["customer_id", "bureau_provider"], observed=True)[num + ["report_age_days"]].last()
    c = cur.set_index(["customer_id", "bureau_provider"])[num]
    delta = (c - last_prev[num].reindex(c.index)).groupby(level=0).mean()
    agg = delta.add_suffix("_delta_mean")
    agg["score_delta_min"] = (c["score"] - last_prev["score"].reindex(c.index)).groupby(
        level=0).min()
    first_prev = prev.sort_values("report_datetime").groupby("customer_id")["score"].first()
    agg["score_change_since_first"] = cur.groupby("customer_id")["score"].mean() - first_prev
    agg["bureau_hist_days"] = prev.groupby("customer_id")["report_age_days"].max()
    agg["n_prev_reports"] = prev.groupby("customer_id").size()
    agg["prev_delinq_max"] = prev.groupby("customer_id")["delinquent_accounts_last_12m"].max()
    agg["prev_score_min"] = prev.groupby("customer_id")["score"].min()
    # Kemiringan skor per tahun terhadap waktu laporan (prev + current).
    allr = pd.concat([prev[["customer_id", "report_age_days", "score"]],
                      cur[["customer_id", "report_age_days", "score"]]]).dropna()
    x = -allr["report_age_days"] / 365
    allr = allr.assign(_x=x, _xy=x * allr["score"], _xx=x * x)
    s = allr.groupby("customer_id")[["_x", "_xy", "_xx", "score"]].sum()
    n = allr.groupby("customer_id").size()
    den = n * s["_xx"] - s["_x"] ** 2
    agg["score_slope_per_year"] = (n * s["_xy"] - s["_x"] * s["score"]) / den.where(den > 1e-9)
    return by_customer(agg, apps, "bp_")
