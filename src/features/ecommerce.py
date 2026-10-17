"""Fitur e-commerce (ec_). Recency belanja adalah fitur tunggal terkuat di E03 (AUC 0,689).

order_total berulang per item -> metrik level order memakai data yang di-dedup per order_id.
Order cancelled bukan spending.
"""
import numpy as np
import pandas as pd

from ..io_utils import cutoffs, load
from .common import by_customer, days_between

WINDOWS = [3, 7, 14, 30, 90, 180, 365]
_COLS = ["order_id", "customer_id", "transaction_datetime", "product_category", "item_total",
         "order_total", "payment_method", "order_status"]


def _load_ec(apps):
    ec = load("ecommerce_transactions", usecols=_COLS)
    ec = ec.merge(cutoffs(apps)[["customer_id", "cutoff"]], on="customer_id", how="inner")
    ec["age_days"] = days_between(ec["cutoff"], ec["transaction_datetime"]).astype("float32")
    ec["completed"] = (ec["order_status"] == "completed").astype("int8")
    orders = ec.drop_duplicates("order_id")[["customer_id", "transaction_datetime", "age_days",
                                             "order_total", "payment_method", "completed"]]
    return ec, orders


def build_ec_window(apps):
    """E50: RFM per jendela 3–365 hari, recency per status, jarak antarorder."""
    _, orders = _load_ec(apps)
    done = orders[orders["completed"] == 1]
    canc = orders[orders["completed"] == 0]
    agg = pd.DataFrame(index=orders["customer_id"].unique())
    tot_days = orders.groupby("customer_id")["age_days"].max().clip(lower=30)
    spend_all = done.groupby("customer_id")["order_total"].sum()
    n_all = orders.groupby("customer_id").size()
    for w in WINDOWS:
        o = orders[orders["age_days"] <= w]
        d = done[done["age_days"] <= w]
        agg[f"w{w}_n_orders"] = o.groupby("customer_id").size()
        agg[f"w{w}_spend"] = d.groupby("customer_id")["order_total"].sum()
        agg[f"w{w}_n_cancel"] = (o["completed"] == 0).groupby(o["customer_id"]).sum()
        # Intensitas relatif terhadap rata-rata historis customer (1 = normal).
        agg[f"w{w}_order_intensity"] = agg[f"w{w}_n_orders"] / (n_all * w / tot_days)
        agg[f"w{w}_spend_intensity"] = agg[f"w{w}_spend"] / (spend_all * w / tot_days)
    win_cols = [c for c in agg.columns if c.endswith(("_n_orders", "_spend", "_n_cancel"))]
    agg[win_cols] = agg[win_cols].fillna(0)
    agg["days_since_last_completed"] = done.groupby("customer_id")["age_days"].min()
    agg["days_since_last_cancel"] = canc.groupby("customer_id")["age_days"].min()
    # Jarak antarorder: gap terakhir dibanding gap rata-rata.
    o = orders.sort_values(["customer_id", "age_days"])
    o["gap"] = o.groupby("customer_id")["age_days"].diff()
    g = o.groupby("customer_id")["gap"]
    agg["gap_mean"] = g.mean()
    agg["gap_median"] = g.median()
    agg["gap_min"] = g.min()
    recency = orders.groupby("customer_id")["age_days"].min()
    agg["recency_over_gap"] = recency / agg["gap_mean"]
    agg["n_orders_last_day"] = orders[orders["age_days"] <= 1.5].groupby("customer_id").size()
    agg["n_orders_last_day"] = agg["n_orders_last_day"].fillna(0)
    agg["last_order_total"] = orders.sort_values("age_days").groupby("customer_id")[
        "order_total"].first()
    agg["last_order_completed"] = orders.sort_values("age_days").groupby("customer_id")[
        "completed"].first()
    return by_customer(agg, apps, "ec_")


def build_ec_mix(apps):
    """E51: pergeseran metode bayar & kategori menjelang aplikasi, pola waktu belanja."""
    ec, orders = _load_ec(apps)
    agg = pd.DataFrame(index=orders["customer_id"].unique())
    pm_all = pd.crosstab(orders["customer_id"], orders["payment_method"], normalize="index")
    for w in [30, 90]:
        o = orders[orders["age_days"] <= w]
        pm = pd.crosstab(o["customer_id"], o["payment_method"], normalize="index")
        pm = pm.reindex(columns=pm_all.columns)
        agg = agg.join((pm - pm_all.reindex(pm.index)).add_prefix(f"w{w}_pm_shift_"))
    done = orders[orders["completed"] == 1]
    agg = agg.join(pd.crosstab(done["customer_id"], done["payment_method"],
                               values=done["order_total"], aggfunc="sum",
                               normalize="index").add_prefix("pm_value_share_"))
    items = ec[ec["completed"] == 1]
    cat_all = items.pivot_table(index="customer_id", columns="product_category",
                                values="item_total", aggfunc="sum", fill_value=0, observed=False)
    cat_all = cat_all.div(cat_all.sum(axis=1), axis=0)
    it30 = items[items["age_days"] <= 30]
    cat30 = it30.pivot_table(index="customer_id", columns="product_category",
                             values="item_total", aggfunc="sum", fill_value=0, observed=False)
    cat30 = cat30.div(cat30.sum(axis=1), axis=0)
    agg = agg.join((cat30 - cat_all.reindex(cat30.index)).add_prefix("w30_cat_shift_"))
    # Belanja elektronik & kesehatan 30 hari terakhir (nilai absolut).
    for c in ["electronics", "health"]:
        agg[f"w30_spend_{c}"] = it30[it30["product_category"] == c].groupby("customer_id")[
            "item_total"].sum()
        agg[f"w30_spend_{c}"] = agg[f"w30_spend_{c}"].fillna(0)
    hour = orders["transaction_datetime"].dt.hour
    orders = orders.assign(night=((hour >= 0) & (hour < 5)).astype("int8"),
                           weekend=(orders["transaction_datetime"].dt.dayofweek >= 5).astype("int8"))
    g = orders.groupby("customer_id")
    agg["share_night"] = g["night"].mean()
    agg["share_weekend"] = g["weekend"].mean()
    agg["cancel_ratio_w90"] = 1 - orders[orders["age_days"] <= 90].groupby("customer_id")[
        "completed"].mean()
    return by_customer(agg, apps, "ec_")
