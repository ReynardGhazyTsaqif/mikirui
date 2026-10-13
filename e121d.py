"""E121d: model level baris (nested, sama dengan e121.py) untuk tabel selain e-commerce.

Tabel: rekening (baris = bulan), cicilan (baris = cicilan), biro (baris = laporan), aplikasi lama.
Hanya fitur intrinsik baris + waktu relatif ke cutoff (tanpa konteks customer). Keluaran memakai format
npz yang sama dengan e121.py, jadi bisa digabung lewat `e121.py --cv --use a.npz+b.npz`.

Contoh:
  python e121d.py --table ab          # -> features/e121_ordermodel_v1_ab.npz
  python e121d.py --table rp --shuffle
"""
import argparse
import gc
import sys

import pandas as pd

import e121
from src.config import FEATURES_DIR
from src.io_utils import load, load_apps


def rows_ab(apps):
    ab = load("account_behavior").merge(apps[["customer_id", "decision_datetime"]], on="customer_id")
    prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
    ab = ab.merge(prof, on="customer_id")
    cut = ab["decision_datetime"]
    ab["recency_h"] = ((cut.dt.year * 12 + cut.dt.month) - (ab["month"].dt.year * 12 + ab["month"].dt.month)) * 720.0
    num = ["opening_balance", "closing_balance", "inflows", "outflows", "income_like_credits",
           "average_balance", "minimum_balance"]
    for c in num:
        ab[c + "_vi"] = ab[c] / ab["declared_monthly_income"]
    ab["net_vi"] = (ab["inflows"] - ab["outflows"]) / ab["declared_monthly_income"]
    ab["month_of_year"] = ab["month"].dt.month
    feats = ["recency_h", "low_balance_days", "net_vi", "month_of_year"] + num + [c + "_vi" for c in num]
    return ab, feats


def rows_rp(apps):
    rp = load("repayment_history").merge(apps[["customer_id", "decision_datetime"]], on="customer_id")
    rp["recency_h"] = (rp["decision_datetime"] - rp["due_date"]).dt.total_seconds() / 3600
    rp["paid_ratio"] = rp["amount_paid"] / rp["amount_due"].where(rp["amount_due"] > 0)
    rp["delay"] = (rp["last_payment_datetime"] - rp["due_date"]).dt.total_seconds() / 86400
    rp["unpaid"] = rp["last_payment_datetime"].isna().astype("int8")
    rp = rp.sort_values(["loan_id", "due_date"])
    rp["inst_no"] = rp.groupby("loan_id").cumcount() + 1
    pa = load("previous_applications", usecols=["loan_id", "approved_amount", "annual_interest_rate",
                                                 "term_months"]).dropna(subset=["loan_id"])
    rp = rp.merge(pa, on="loan_id", how="left")
    feats = ["recency_h", "amount_due", "paid_ratio", "delay", "unpaid", "days_past_due",
             "max_days_past_due", "inst_no", "approved_amount", "annual_interest_rate", "term_months"]
    return rp, feats


def rows_bu(apps):
    bu = load("bureau_observations").merge(apps[["customer_id", "decision_datetime"]], on="customer_id")
    bu["recency_h"] = (bu["decision_datetime"] - bu["available_datetime"]).dt.total_seconds() / 3600
    bu["lag"] = (bu["available_datetime"] - bu["report_datetime"]).dt.days
    bu["prov"] = bu["bureau_provider"].cat.codes
    bu["is_current"] = bu["application_id"].isin(set(apps["application_id"])).astype("int8")
    feats = ["recency_h", "lag", "prov", "is_current", "score", "active_accounts", "outstanding_balance",
             "revolving_credit_limit", "revolving_outstanding_balance", "utilization", "inquiries_last_6m",
             "delinquent_accounts_last_12m"]
    return bu, feats


def rows_pa(apps):
    pa = load("previous_applications").merge(apps[["customer_id", "decision_datetime"]], on="customer_id",
                                              suffixes=("", "_cur"))
    prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
    pa = pa.merge(prof, on="customer_id")
    pa["recency_h"] = (pa["decision_datetime_cur"] - pa["decision_datetime"]).dt.total_seconds() / 3600
    pa["proc_h"] = (pa["decision_datetime"] - pa["application_datetime"]).dt.total_seconds() / 3600
    pa["dec"] = pa["decision"].cat.codes
    pa["appr_ratio"] = pa["approved_amount"] / pa["requested_amount"]
    pa["req_vi"] = pa["requested_amount"] / pa["declared_monthly_income"]
    pa["inst_vi"] = pa["scheduled_monthly_installment"] / pa["declared_monthly_income"]
    feats = ["recency_h", "proc_h", "dec", "requested_amount", "term_months", "approved_amount",
             "annual_interest_rate", "scheduled_monthly_installment", "appr_ratio", "req_vi", "inst_vi"]
    return pa, feats


TABLES = {"ab": rows_ab, "rp": rows_rp, "bu": rows_bu, "pa": rows_pa}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=list(TABLES), required=True)
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()
    path = FEATURES_DIR / f"ec_rows_{a.table}_v1.parquet"
    if not path.exists():
        apps = load_apps()
        df, feats = TABLES[a.table](apps)
        df = df[["customer_id"] + feats].sort_values(["customer_id", "recency_h"], kind="stable")
        df[feats] = df[feats].astype("float32")
        df.reset_index(drop=True).to_parquet(path)
        del df, apps
        gc.collect()
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c != "customer_id"]
    e121.FEATS[:] = feats
    e121.ORDERS = path  # build_orders membaca cache ini
    e121.build(a.shuffle, f"_{a.table}")


if __name__ == "__main__":
    main()
