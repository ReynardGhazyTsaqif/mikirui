"""E126: pinjaman lama sebagai data berlabel.

Pinjaman lama yang disetujui dan jendela 12 bulannya sudah lengkap sebelum cutoff aplikasi sekarang punya hasil yang
diketahui dari repayment_history (pernah DPD >= 90 dalam 12 bulan setelah pencairan). Untuk tiap pinjaman lama dan
tiap aplikasi sekarang dihitung fitur yang sama *as-of* tanggal keputusannya masing-masing (t), lalu:
  --build  : fitur as-of -> features/e126_asof_v1.parquet (baris pinjaman lama + aplikasi sekarang)
  --pd     : model PD pinjaman lama (LightGBM, CV dikelompokkan per customer) -> skor untuk aplikasi sekarang
             -> features/e126_prevpd_v1.parquet (kolom x_prevpd*), lalu screen di atas acuan
Label aplikasi sekarang tidak dipakai model PD pinjaman lama, jadi fiturnya tidak perlu nested.

Catatan as-of untuk cicilan: days_past_due / max_days_past_due di data adalah kondisi per cutoff aplikasi sekarang.
Untuk t lebih awal: cicilan yang jatuh tempo < t dan lunas sebelum t memakai max_days_past_due (sudah final);
yang belum lunas pada t diberi DPD = t - due_date. Data e-commerce/rekening hanya tersedia ±730 hari / 23 bulan
sebelum cutoff sekarang, jadi fitur memakai jendela waktu + penanda panjang riwayat yang tersedia.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR
from src.io_utils import cutoffs, load, load_apps

DAY = pd.Timedelta(days=1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_keys(apps):
    """Satu baris per titik waktu t: aplikasi sekarang (kind=0) + pinjaman lama dengan jendela lengkap (kind=1)."""
    cut = cutoffs(apps)[["customer_id", "cutoff"]]
    cur = apps[["application_id", "customer_id", "decision_datetime", "loan_amount", "term_months",
                "annual_interest_rate", "proposed_monthly_installment"]].rename(
        columns={"decision_datetime": "t", "loan_amount": "amount", "proposed_monthly_installment": "inst"})
    cur["requested"] = cur["amount"]
    cur["kind"] = 0
    pa = load("previous_applications").merge(cut, on="customer_id")
    rp = load("repayment_history", usecols=["loan_id", "due_date", "max_days_past_due"])
    ap = pa[pa.decision == "approved"].copy()
    lab = rp.merge(ap[["loan_id", "decision_datetime"]], on="loan_id")
    lab = lab[lab.due_date <= lab.decision_datetime + 275 * DAY]  # DPD 90 harus tercapai dalam 365 hari
    y = lab.groupby("loan_id")["max_days_past_due"].max().ge(90).astype(int)
    ap = ap[ap.decision_datetime + 365 * DAY <= ap.cutoff].copy()  # jendela 12 bulan lengkap sebelum cutoff sekarang
    ap["y"] = ap.loan_id.map(y)
    ap = ap[ap.y.notna()]
    prev = ap.rename(columns={"decision_datetime": "t", "approved_amount": "amount", "requested_amount": "requested",
                              "scheduled_monthly_installment": "inst"})[
        ["application_id", "customer_id", "t", "amount", "requested", "term_months", "annual_interest_rate", "inst", "loan_id", "y"]]
    prev["kind"] = 1
    K = pd.concat([cur, prev], ignore_index=True)
    K = K.merge(cut, on="customer_id")
    K["key"] = np.arange(len(K))
    log(f"kunci: {int((K.kind == 0).sum())} aplikasi sekarang + {int((K.kind == 1).sum())} pinjaman lama "
        f"(default {K.loc[K.kind == 1, 'y'].mean():.3f})")
    return K


def f_loan_profile(K):
    prof = load("customer_profiles")
    d = K[["key", "customer_id", "t", "cutoff", "amount", "requested", "term_months", "annual_interest_rate", "inst"]].merge(prof, on="customer_id")
    yrs = (d.cutoff - d.t).dt.days / 365.25
    out = pd.DataFrame({"key": d.key})
    out["amount"] = d.amount
    out["appr_ratio"] = d.amount / d.requested
    out["term"] = d.term_months
    out["rate"] = d.annual_interest_rate
    out["inst"] = d.inst
    out["age"] = d.age - yrs
    out["tenure"] = d.employment_tenure_months - yrs * 12
    out["income"] = d.declared_monthly_income
    out["dependents"] = d.dependents
    out["pti"] = d.inst / d.declared_monthly_income
    out["lti"] = d.amount / (12 * d.declared_monthly_income)
    for c in ["employment_type", "housing_status", "region"]:
        out[c] = d[c].astype("category").cat.codes
    out["month"] = d.t.dt.month
    out["hist_avail_d"] = 730 - (d.cutoff - d.t).dt.days  # panjang riwayat e-commerce yang tersedia sebelum t
    return out.set_index("key")


def f_bureau(K):
    bu = load("bureau_observations")
    d = K[["key", "application_id"]].merge(bu, on="application_id")
    g = d.groupby("key")
    out = pd.DataFrame({
        "bu_n": g.size(), "bu_score_mean": g.score.mean(), "bu_score_min": g.score.min(), "bu_score_max": g.score.max(),
        "bu_active": g.active_accounts.max(), "bu_out": g.outstanding_balance.max(), "bu_util": g.utilization.mean(),
        "bu_util_max": g.utilization.max(), "bu_inq": g.inquiries_last_6m.max(), "bu_delinq": g.delinquent_accounts_last_12m.max(),
        "bu_rev_limit": g.revolving_credit_limit.max(), "bu_lag_d": (g.available_datetime.max() - g.report_datetime.max()).dt.days,
    })
    return out


def f_prev_apps(K):
    pa = load("previous_applications")
    d = K[["key", "customer_id", "t"]].merge(pa, on="customer_id", suffixes=("", "_p"))
    d = d[d.decision_datetime < d.t]
    age = (d.t - d.decision_datetime).dt.days
    out = pd.DataFrame(index=K.key)
    g = d.groupby("key")
    out["pa_n"] = g.size()
    out["pa_days_last"] = age.groupby(d.key).min()
    for w in (90, 365):
        m = age <= w
        out[f"pa_n{w}"] = m.groupby(d.key).sum()
        for dec in ("approved", "rejected", "cancelled"):
            out[f"pa_{dec[:3]}{w}"] = (m & (d.decision == dec)).groupby(d.key).sum()
        out[f"pa_amt{w}"] = d.approved_amount.where(m, 0).groupby(d.key).sum()
        out[f"pa_inst{w}"] = d.scheduled_monthly_installment.where(m, 0).groupby(d.key).sum()
    out["pa_rate_max"] = g.annual_interest_rate.max()
    out["pa_rate_last"] = d.sort_values("decision_datetime").groupby("key").annual_interest_rate.last()
    return out


def f_repay(K):
    rp = load("repayment_history")
    d = K[["key", "customer_id", "t"]].merge(rp, on="customer_id")
    d = d[d.due_date < d.t]
    paid_before = (d.amount_paid >= d.amount_due) & (d.last_payment_datetime < d.t)
    dpd_t = np.where(paid_before, d.max_days_past_due, (d.t - d.due_date).dt.days.clip(lower=0))
    d = d.assign(dpd=dpd_t, open_t=(~paid_before).astype(int), age=(d.t - d.due_date).dt.days)
    d["open_dpd"] = np.where(d.open_t == 1, d.dpd, 0)
    out = pd.DataFrame(index=K.key)
    g = d.groupby("key")
    out["rp_n"] = g.size()
    out["rp_dpd_max"] = g.dpd.max()
    out["rp_dpd_mean"] = g.dpd.mean()
    out["rp_open_dpd_max"] = g.open_dpd.max()  # tunggakan pada t
    out["rp_open_n"] = g.open_t.sum()
    for thr in (1, 30, 90):
        out[f"rp_n_dpd{thr}"] = (d.dpd >= thr).groupby(d.key).sum()
        out[f"rp_share_dpd{thr}"] = out[f"rp_n_dpd{thr}"] / out["rp_n"]
    out["rp_loans_dpd90"] = d[d.dpd >= 90].groupby("key").loan_id.nunique()
    for w in (90, 180, 365):
        m = d.age <= w
        out[f"rp_dpd_max{w}"] = d.dpd.where(m).groupby(d.key).max()
        out[f"rp_n_dpd30_{w}"] = ((d.dpd >= 30) & m).groupby(d.key).sum()
    out["rp_days_since_dpd30"] = d[d.dpd >= 30].groupby("key").age.min()
    out["rp_days_last_due"] = g.age.min()
    return out


def f_account(K):
    ab = load("account_behavior")
    d = K[["key", "customer_id", "t"]].merge(ab, on="customer_id")
    mago = (d.t.dt.year * 12 + d.t.dt.month) - (d.month.dt.year * 12 + d.month.dt.month)
    d = d.assign(mago=mago)
    d = d[d.mago >= 1]
    out = pd.DataFrame(index=K.key)
    out["ab_n"] = d.groupby("key").size()
    for w in (3, 6):
        s = d[d.mago <= w]
        g = s.groupby("key")
        out[f"ab_close{w}"] = g.closing_balance.mean()
        out[f"ab_min{w}"] = g.minimum_balance.min()
        out[f"ab_in{w}"] = g.inflows.mean()
        out[f"ab_out{w}"] = g.outflows.mean()
        out[f"ab_inc{w}"] = g.income_like_credits.mean()
        out[f"ab_low{w}"] = g.low_balance_days.mean()
        out[f"ab_net{w}"] = (s.inflows - s.outflows).groupby(s.key).mean()
    last = d[d.mago == 1].set_index("key")
    out["ab_close_last"] = last.closing_balance
    out["ab_low_last"] = last.low_balance_days
    prev = d[(d.mago > 3) & (d.mago <= 6)].groupby("key").closing_balance.mean()
    out["ab_trend"] = out["ab_close3"] - prev
    return out


def f_ecom(K):
    ec = pd.read_csv("Dataset/ecommerce_transactions.csv",
                     usecols=["order_id", "customer_id", "transaction_datetime", "order_total", "payment_method", "order_status"],
                     parse_dates=["transaction_datetime"]).drop_duplicates("order_id")
    d = K[["key", "customer_id", "t"]].merge(ec, on="customer_id")
    d = d[d.transaction_datetime < d.t]
    age = (d.t - d.transaction_datetime).dt.total_seconds() / 86400
    d = d.assign(age=age, canc=(d.order_status == "cancelled").astype(int), cod=(d.payment_method == "COD").astype(int))
    out = pd.DataFrame(index=K.key)
    out["ec_days_last"] = d.groupby("key").age.min()
    s = d.sort_values(["key", "age"])
    s = s[s.groupby("key").cumcount() == 1]
    out["ec_days_2nd"] = s.set_index("key").age
    for w in (7, 30, 90, 180):
        m = d.age <= w
        out[f"ec_n{w}"] = m.groupby(d.key).sum()
        out[f"ec_spend{w}"] = d.order_total.where(m & (d.canc == 0), 0).groupby(d.key).sum()
    m90 = d.age <= 90
    out["ec_canc90"] = d.canc.where(m90).groupby(d.key).mean()
    out["ec_cod90"] = d.cod.where(m90).groupby(d.key).mean()
    out["ec_ratio_7_90"] = out["ec_n7"] / (out["ec_n90"] + 1)
    out["ec_ratio_30_180"] = out["ec_n30"] / (out["ec_n180"] + 1)
    return out


def build():
    apps = load_apps()
    K = make_keys(apps)
    parts = [f_loan_profile(K)]
    for f in (f_bureau, f_prev_apps, f_repay, f_account, f_ecom):
        log(f"{f.__name__} ...")
        parts.append(f(K))
    F = pd.concat(parts, axis=1).reindex(K.key)
    F = F.astype("float32")
    meta = K.set_index("key")[["application_id", "customer_id", "kind", "y", "t", "cutoff"]]
    out = pd.concat([meta, F], axis=1)
    out.to_parquet(FEATURES_DIR / "e126_asof_v1.parquet")
    log(f"selesai: {out.shape}, {len(F.columns)} fitur -> features/e126_asof_v1.parquet")


def pd_model(seeds=(42, 43, 44)):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    A = pd.read_parquet(FEATURES_DIR / "e126_asof_v1.parquet")
    feats = [c for c in A.columns if c not in ("application_id", "customer_id", "kind", "y", "t", "cutoff")]
    P = A[A.kind == 1].reset_index(drop=True)
    C = A[A.kind == 0].reset_index(drop=True)
    params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6)
    # CV pinjaman lama (dikelompokkan per customer) untuk menilai model dan memilih jumlah iterasi
    oof = np.zeros(len(P)); its = []
    for tr, va in GroupKFold(5).split(P, P.y, P.customer_id):
        m = lgb.train({**params, "seed": 42}, lgb.Dataset(P.loc[tr, feats], P.y[tr]), 5000,
                      valid_sets=[lgb.Dataset(P.loc[va, feats], P.y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(P.loc[va, feats], num_iteration=m.best_iteration); its.append(m.best_iteration)
    log(f"model PD pinjaman lama: AUC CV {roc_auc_score(P.y, oof):.4f}, iterasi {its}")
    n_it = int(np.mean(its) * 1.1)
    pred = np.zeros(len(C))
    for s in seeds:
        m = lgb.train({**params, "seed": s}, lgb.Dataset(P[feats], P.y), n_it)
        pred += m.predict(C[feats]) / len(seeds)
    imp = pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False)
    log("fitur terpenting: " + ", ".join(f"{k} {v:.0f}" for k, v in imp.head(10).items()))
    # skor untuk aplikasi sekarang; urutan = load_apps
    apps = load_apps()
    out = pd.DataFrame({"x_prevpd": pred}, index=C.application_id).reindex(apps.application_id)
    tr = apps.is_test.eq(0).values
    y = apps.target.values[tr]
    log(f"AUC x_prevpd terhadap target sekarang (train): {roc_auc_score(y, out.x_prevpd.values[tr]):.4f}")
    out.index.name = "application_id"
    out.astype("float32").to_parquet(FEATURES_DIR / "e126_prevpd_v1.parquet")
    log("-> features/e126_prevpd_v1.parquet")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--pd", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.pd:
        pd_model()
