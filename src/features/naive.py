"""E03: agregat naif (count/mean/max/min/std) untuk semua tabel pendukung.

Baseline B1. Fitur yang lebih cermat per tabel (E10 dst.) ditulis di modul per tabel.
Semua recency dihitung terhadap cutoff = decision_datetime aplikasi saat ini.
"""
import pandas as pd

from ..io_utils import cutoffs, load
from .common import by_customer, days_between, finalize

STATS = ["mean", "max", "min", "std"]


def _num_agg(df, cols, stats=STATS):
    agg = df.groupby("customer_id")[cols].agg(stats)
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg


def _with_cutoff(df, apps):
    return df.merge(cutoffs(apps)[["customer_id", "cutoff"]], on="customer_id", how="inner")


def build_pa_naive(apps):
    pa = _with_cutoff(load("previous_applications"), apps)
    pa["days_before_cutoff"] = days_between(pa["cutoff"], pa["decision_datetime"])
    pa["approved_ratio_amt"] = pa["approved_amount"] / pa["requested_amount"]
    num = ["requested_amount", "term_months", "approved_amount", "annual_interest_rate",
           "scheduled_monthly_installment", "days_before_cutoff", "approved_ratio_amt"]
    agg = _num_agg(pa, num)
    agg["count"] = pa.groupby("customer_id").size()
    dec = pd.crosstab(pa["customer_id"], pa["decision"])
    for d in ["approved", "rejected", "cancelled"]:
        agg[f"n_{d}"] = dec[d]
        agg[f"share_{d}"] = dec[d] / agg["count"]
    return by_customer(agg, apps, "pa_")


def build_rp_naive(apps):
    rp = _with_cutoff(load("repayment_history"), apps)
    rp["unpaid"] = rp["amount_due"] - rp["amount_paid"]
    rp["paid_ratio"] = rp["amount_paid"] / rp["amount_due"]
    rp["due_days_before_cutoff"] = days_between(rp["cutoff"], rp["due_date"])
    num = ["amount_due", "amount_paid", "unpaid", "paid_ratio", "days_past_due",
           "max_days_past_due", "due_days_before_cutoff"]
    agg = _num_agg(rp, num)
    g = rp.groupby("customer_id")
    agg["n_installments"] = g.size()
    agg["n_loans"] = g["loan_id"].nunique()
    agg["sum_due"] = g["amount_due"].sum()
    agg["sum_paid"] = g["amount_paid"].sum()
    agg["sum_unpaid"] = g["unpaid"].sum()
    agg["paid_ratio_total"] = agg["sum_paid"] / agg["sum_due"]
    return by_customer(agg, apps, "rp_")


_BU_NUM = ["score", "active_accounts", "outstanding_balance", "revolving_credit_limit",
           "revolving_outstanding_balance", "utilization", "inquiries_last_6m",
           "delinquent_accounts_last_12m"]


def build_bu_naive(apps):
    """Laporan biro untuk aplikasi saat ini: pivot per provider + agregat lintas provider."""
    bo = load("bureau_observations")
    cur = bo[bo["application_id"].isin(apps["application_id"])].merge(
        cutoffs(apps)[["application_id", "cutoff"]], on="application_id")
    cur["report_age_days"] = days_between(cur["cutoff"], cur["report_datetime"])
    num = _BU_NUM + ["report_age_days"]
    piv = cur.pivot(index="application_id", columns="bureau_provider", values=num)
    piv.columns = [f"{c}_{p[-1]}" for c, p in piv.columns]
    agg = cur.groupby("application_id")[num].agg(STATS)
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n_providers"] = cur.groupby("application_id").size()
    agg["n_unscored"] = cur["score"].isna().groupby(cur["application_id"]).sum()
    out = piv.join(agg).reindex(apps["application_id"])
    return finalize(out, "bu_")


def build_bp_naive(apps):
    """Laporan biro untuk previous applications, diagregasi per customer."""
    bo = load("bureau_observations")
    prev = _with_cutoff(bo[~bo["application_id"].isin(apps["application_id"])], apps)
    prev["report_age_days"] = days_between(prev["cutoff"], prev["report_datetime"])
    agg = _num_agg(prev, _BU_NUM + ["report_age_days"], ["mean", "max", "min"])
    agg["n_reports"] = prev.groupby("customer_id").size()
    return by_customer(agg, apps, "bp_")


def build_ab_naive(apps):
    ab = load("account_behavior")
    ab["net_flow"] = ab["inflows"] - ab["outflows"]
    num = ["opening_balance", "closing_balance", "inflows", "outflows", "income_like_credits",
           "average_balance", "minimum_balance", "low_balance_days", "net_flow"]
    agg = _num_agg(ab, num)
    g = ab.groupby("customer_id")
    agg["n_months"] = g.size()
    last = ab.sort_values("month").groupby("customer_id").last()
    for c in ["closing_balance", "average_balance", "income_like_credits", "low_balance_days"]:
        agg[f"{c}_last"] = last[c]
    cut = cutoffs(apps).set_index("customer_id")["cutoff"]
    agg["days_since_last_month"] = days_between(cut.reindex(agg.index), last["month"])
    return by_customer(agg, apps, "ab_")


def build_ec_naive(apps):
    cols = ["order_id", "customer_id", "transaction_datetime", "product_category", "quantity",
            "item_total", "order_total", "payment_method", "order_status"]
    ec = load("ecommerce_transactions", usecols=cols)
    ec["quantity"] = ec["quantity"].astype("int16")
    ec["item_total"] = ec["item_total"].astype("int64")
    completed = ec["order_status"] == "completed"

    # Level order: order_total berulang per item -> dedup per order_id.
    orders = ec.drop_duplicates("order_id")[["order_id", "customer_id", "transaction_datetime",
                                             "order_total", "payment_method", "order_status"]]
    orders = _with_cutoff(orders, apps)
    orders["completed"] = orders["order_status"] == "completed"
    done = orders[orders["completed"]]
    g = orders.groupby("customer_id")
    agg = pd.DataFrame({"n_orders": g.size(), "n_completed": g["completed"].sum()})
    agg["cancel_ratio"] = 1 - agg["n_completed"] / agg["n_orders"]
    gd = done.groupby("customer_id")["order_total"]
    agg["spend_sum"] = gd.sum()
    agg["spend_mean"] = gd.mean()
    agg["spend_max"] = gd.max()
    agg["spend_std"] = gd.std()
    agg["days_since_last_order"] = days_between(g["cutoff"].first(), g["transaction_datetime"].max())
    agg["days_since_first_order"] = days_between(g["cutoff"].first(), g["transaction_datetime"].min())
    agg["spend_per_30d"] = agg["spend_sum"] / (agg["days_since_first_order"].clip(lower=30) / 30)
    pm = pd.crosstab(orders["customer_id"], orders["payment_method"], normalize="index")
    agg = agg.join(pm.add_prefix("pm_share_"))

    # Level item: proporsi nilai belanja completed per kategori.
    items = ec[completed]
    cat = items.pivot_table(index="customer_id", columns="product_category", values="item_total",
                            aggfunc="sum", fill_value=0, observed=False)
    cat = cat.div(cat.sum(axis=1), axis=0).add_prefix("cat_share_")
    agg = agg.join(cat)
    agg["n_items"] = ec.groupby("customer_id").size()
    agg["qty_mean"] = items.groupby("customer_id")["quantity"].mean()
    return by_customer(agg, apps, "ec_")
