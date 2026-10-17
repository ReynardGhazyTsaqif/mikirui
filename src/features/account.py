"""Fitur perilaku rekening (ab_). Agregat mentahnya lemah, jadi fokus ke tren, volatilitas, rasio."""
import numpy as np
import pandas as pd

from ..io_utils import load
from .common import by_customer

_NUM = ["closing_balance", "inflows", "outflows", "income_like_credits", "average_balance",
        "minimum_balance", "low_balance_days", "net_flow"]


def _load_ab():
    ab = load("account_behavior")
    ab["net_flow"] = ab["inflows"] - ab["outflows"]
    ab = ab.sort_values(["customer_id", "month"])
    # Urutan bulan dari belakang: 1 = bulan observasi terakhir.
    ab["rev_idx"] = ab.groupby("customer_id").cumcount(ascending=False) + 1
    return ab


def _slope(df, col):
    """Kemiringan OLS col terhadap urutan bulan, dinormalisasi oleh rata-rata |col|."""
    x = -df["rev_idx"].astype(float)
    t = df.assign(_x=x, _y=df[col], _xy=x * df[col], _xx=x * x)
    s = t.groupby("customer_id")[["_x", "_y", "_xy", "_xx"]].sum()
    n = t.groupby("customer_id").size()
    den = n * s["_xx"] - s["_x"] ** 2
    slope = (n * s["_xy"] - s["_x"] * s["_y"]) / den.where(den > 0)
    return slope / (s["_y"].abs() / n).replace(0, np.nan)


def build_ab_window(apps):
    """E40: jendela 3/6/12 bulan terakhir, tren, volatilitas, rasio terbaru vs sebelumnya."""
    ab = _load_ab()
    agg = pd.DataFrame(index=ab["customer_id"].unique())
    for w in [3, 6, 12]:
        g = ab[ab["rev_idx"] <= w].groupby("customer_id")
        for c in ["average_balance", "minimum_balance", "income_like_credits", "net_flow"]:
            agg[f"w{w}_{c}_mean"] = g[c].mean()
        agg[f"w{w}_min_balance_min"] = g["minimum_balance"].min()
        agg[f"w{w}_low_days_sum"] = g["low_balance_days"].sum()
        agg[f"w{w}_n_neg_flow"] = (ab.loc[ab["rev_idx"] <= w, "net_flow"] < 0).groupby(
            ab.loc[ab["rev_idx"] <= w, "customer_id"]).sum()
        agg[f"w{w}_outflow_inflow"] = g["outflows"].sum() / g["inflows"].sum().replace(0, np.nan)
    prev = ab[(ab["rev_idx"] > 3) & (ab["rev_idx"] <= 6)].groupby("customer_id")
    for c in ["average_balance", "income_like_credits", "low_balance_days"]:
        agg[f"{c}_3m_vs_prev3m"] = agg.get(f"w3_{c}_mean", ab[ab["rev_idx"] <= 3].groupby(
            "customer_id")[c].mean()) - prev[c].mean()
    for c in ["average_balance", "income_like_credits", "closing_balance", "outflows"]:
        agg[f"{c}_slope"] = _slope(ab, c)
    g = ab.groupby("customer_id")
    agg["income_cv"] = g["income_like_credits"].std() / g["income_like_credits"].mean().replace(
        0, np.nan)
    agg["balance_cv"] = g["average_balance"].std() / g["average_balance"].mean().abs().replace(
        0, np.nan)
    agg["share_months_low"] = (ab["low_balance_days"] > 0).groupby(ab["customer_id"]).mean()
    agg["last_low_days"] = g["low_balance_days"].last()
    agg["last_min_balance"] = g["minimum_balance"].last()
    return by_customer(agg, apps, "ab_")


def build_ab_income(apps):
    """E41: verifikasi pendapatan dan bantalan likuiditas terhadap cicilan baru."""
    ab = _load_ab()
    prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
    cur = apps[["customer_id", "proposed_monthly_installment"]].merge(prof, on="customer_id")
    cur = cur.set_index("customer_id")
    agg = pd.DataFrame(index=ab["customer_id"].unique())
    for w, lab in [(3, "w3"), (100, "all")]:
        g = ab[ab["rev_idx"] <= w].groupby("customer_id")
        inc = g["income_like_credits"].mean()
        agg[f"{lab}_income_vs_declared"] = inc / cur["declared_monthly_income"].reindex(inc.index)
        agg[f"{lab}_inst_vs_income"] = cur["proposed_monthly_installment"].reindex(
            inc.index) / inc.replace(0, np.nan)
        agg[f"{lab}_avgbal_vs_inst"] = g["average_balance"].mean() / cur[
            "proposed_monthly_installment"].reindex(inc.index)
        agg[f"{lab}_minbal_vs_inst"] = g["minimum_balance"].mean() / cur[
            "proposed_monthly_installment"].reindex(inc.index)
        agg[f"{lab}_netflow_vs_inst"] = g["net_flow"].mean() / cur[
            "proposed_monthly_installment"].reindex(inc.index)
        agg[f"{lab}_outflow_vs_income"] = g["outflows"].mean() / inc.replace(0, np.nan)
    return by_customer(agg, apps, "ab_")
