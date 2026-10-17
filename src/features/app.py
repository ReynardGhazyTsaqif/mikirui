"""Fitur aplikasi saat ini (app_) dan profil customer (prof_)."""
import numpy as np
import pandas as pd

from ..io_utils import load
from .common import days_between, finalize

APP_CATS = {"loan_purpose": ["education", "home_improvement", "household", "medical",
                             "working_capital"]}
PROF_CATS = {
    "region": ["Region_A", "Region_B", "Region_C", "Region_D", "Region_E"],
    "employment_type": ["salaried", "self_employed", "variable_income"],
    "housing_status": ["family", "owned", "rented"],
}


def _to_cat(df, cats):
    for c, levels in cats.items():
        df[c] = pd.Categorical(df[c], categories=levels)
    return df


def build_app_raw(apps):
    """E01: kolom mentah aplikasi. term_months dibuang (konstan 12)."""
    cols = ["loan_amount", "annual_interest_rate", "proposed_monthly_installment", "loan_purpose"]
    df = apps.set_index("application_id")[cols].copy()
    return finalize(_to_cat(df, APP_CATS), "app_")


def build_prof_raw(apps):
    """E01: kolom mentah profil customer."""
    prof = load("customer_profiles")
    df = apps[["application_id", "customer_id"]].merge(prof, on="customer_id", how="left")
    df = df.set_index("application_id").drop(columns="customer_id")
    return finalize(_to_cat(df, PROF_CATS), "prof_")


def build_app_ratios(apps):
    """E02: keterjangkauan, stabilitas kerja, dan waktu proses aplikasi."""
    prof = load("customer_profiles")
    a = apps.merge(prof, on="customer_id", how="left").set_index("application_id")
    inc = a["declared_monthly_income"].replace(0, np.nan)
    out = pd.DataFrame(index=a.index)
    out["pti"] = a["proposed_monthly_installment"] / inc
    out["lti"] = a["loan_amount"] / inc
    out["income_per_member"] = a["declared_monthly_income"] / (a["dependents"] + 1)
    out["pti_per_member"] = a["proposed_monthly_installment"] / out["income_per_member"]
    out["residual_income"] = a["declared_monthly_income"] - a["proposed_monthly_installment"]
    out["tenure_share_of_age"] = a["employment_tenure_months"] / (a["age"] * 12)
    out["age_started_job"] = a["age"] - a["employment_tenure_months"] / 12
    out["process_hours"] = days_between(a["decision_datetime"], a["application_datetime"]) * 24
    out["app_hour"] = a["application_datetime"].dt.hour
    out["app_dayofweek"] = a["application_datetime"].dt.dayofweek
    out["app_month"] = a["application_datetime"].dt.month
    out["dec_hour"] = a["decision_datetime"].dt.hour
    out["total_interest"] = a["proposed_monthly_installment"] * 12 - a["loan_amount"]
    return finalize(out, "app_")
