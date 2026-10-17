"""Fitur lintas tabel (x_). Dibangun dari cache grup lain agar tidak menghitung ulang."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import N_THREADS, SEED
from .common import cached, finalize


def _get(apps, name):
    # Import lokal untuk menghindari import melingkar dengan registry.
    from .registry import GROUPS
    return cached(name, GROUPS[name], apps)


def build_x_cross(apps):
    """E60: total beban cicilan, utang biro vs pendapatan, tiga sumber pendapatan, skor stres."""
    prof = _get(apps, "prof_raw_v1")
    app_ = _get(apps, "app_raw_v1")
    bu = _get(apps, "bu_naive_v1")
    pab = _get(apps, "pa_burden_v1")
    abi = _get(apps, "ab_naive_v1")
    ec = _get(apps, "ec_naive_v1")
    rpw = _get(apps, "rp_window_v1")
    inc = prof["prof_declared_monthly_income"].replace(0, np.nan)
    inst = app_["app_proposed_monthly_installment"]
    out = pd.DataFrame(index=prof.index)
    total_inst = inst + pab["pa_active_inst_sum"].fillna(0)
    out["total_pti"] = total_inst / inc
    out["bureau_debt_to_annual_income_max"] = bu["bu_outstanding_balance_max"] / (inc * 12)
    out["bureau_debt_plus_loan_to_income"] = (bu["bu_outstanding_balance_max"]
                                              + app_["app_loan_amount"]) / (inc * 12)
    acct_inc = abi["ab_income_like_credits_mean"]
    out["acct_income_vs_declared"] = acct_inc / inc
    out["ec_monthly_spend_vs_income"] = ec["ec_spend_per_30d"] / inc
    out["ec_monthly_spend_vs_acct_income"] = ec["ec_spend_per_30d"] / acct_inc.replace(0, np.nan)
    out["total_inst_vs_acct_income"] = total_inst / acct_inc.replace(0, np.nan)
    out["rate_x_pti"] = app_["app_annual_interest_rate"] * total_inst / inc
    out["rate_x_score"] = app_["app_annual_interest_rate"] * (850 - bu["bu_score_mean"])
    stress = pd.DataFrame({
        "recent_dpd30": rpw["rp_w180_share_dpd30"].fillna(0) > 0,
        "low_balance": abi["ab_low_balance_days_mean"].fillna(0) > 5,
        "high_cancel": ec["ec_cancel_ratio"].fillna(0) > 0.15,
        "bureau_delinq": bu["bu_delinquent_accounts_last_12m_max"].fillna(0) > 0,
        "shopped_last_2d": ec["ec_days_since_last_order"].fillna(99) <= 2,
    })
    out["stress_count"] = stress.sum(axis=1)
    return finalize(out, "x_")


def build_x_rate_resid(apps):
    """E61: residual bunga. Regresi TANPA target: prediksi tier bunga dari informasi risiko yang
    terlihat (biro, cicilan, profil). Residual = penilaian risiko 'privat' lender."""
    names = ["prof_raw_v1", "bu_naive_v1", "rp_naive_v1", "pa_naive_v1", "ab_naive_v1",
             "ec_naive_v1"]
    X = pd.concat([_get(apps, n) for n in names], axis=1)
    y = _get(apps, "app_raw_v1")["app_annual_interest_rate"].values
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, 5, len(y))
    pred = np.zeros(len(y))
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63,
                  min_child_samples=100, feature_fraction=0.5, bagging_fraction=0.8,
                  bagging_freq=1, seed=SEED, deterministic=True, force_row_wise=True,
                  num_threads=N_THREADS, verbose=-1)
    for k in range(5):
        tr, va = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 600)
        pred[va] = m.predict(X[va])
    out = pd.DataFrame({"rate_pred": pred, "rate_resid": y - pred}, index=X.index)
    return finalize(out, "x_")
