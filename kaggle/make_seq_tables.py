"""E122x: tabel urutan per baris untuk model urutan tabel non-e-commerce (cicilan, rekening, pinjaman lama).

Keluaran di kaggle/dataset/ (ikut diunggah ke Kaggle Dataset privat dsc-logika-exp):
  seq_rp_v1.parquet   1 baris = 1 cicilan, urut kronologis per customer (due_date)
  seq_ab_v1.parquet   1 baris = 1 bulan rekening, urut kronologis
  seq_pa_v1.parquet   1 baris = 1 aplikasi lama (+ ringkasan laporan biro milik aplikasi lama itu), urut kronologis

Semua fitur hanya milik baris itu sendiri (nilai, rasio, status) ditambah waktu relatif ke cutoff
(= decision_datetime aplikasi saat ini). Tidak ada target. Kolom `customer_id` untuk menyusun urutan; `t_order`
untuk urutan kronologis. Nominal rupiah diubah ke log bertanda; standarisasi dilakukan di notebook.

  python kaggle/make_seq_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.common import days_between  # noqa: E402
from src.io_utils import cutoffs, load, load_apps  # noqa: E402

OUT = Path(__file__).resolve().parent / "dataset"


def slog(x):
    """Log bertanda untuk nominal rupiah (bisa negatif)."""
    x = np.asarray(x, dtype="float64")
    return (np.sign(x) * np.log1p(np.abs(x))).astype("float32")


def build_rp(cut, pa):
    rp = load("repayment_history").merge(cut, on="customer_id", how="inner")
    rp = rp.merge(pa[["loan_id", "decision_datetime", "approved_amount", "annual_interest_rate",
                      "scheduled_monthly_installment", "term_months"]], on="loan_id", how="left")
    assert (rp["due_date"] < rp["cutoff"]).all()
    rp = rp.sort_values(["customer_id", "due_date", "loan_id"], kind="stable").reset_index(drop=True)
    age = days_between(rp["cutoff"], rp["due_date"])
    unpaid = rp["amount_due"] - rp["amount_paid"]
    delay = days_between(rp["last_payment_datetime"], rp["due_date"])
    delay = delay.where(unpaid <= 0)  # belum lunas: keterlambatan belum final
    k = rp.groupby("loan_id").cumcount()
    out = pd.DataFrame({
        "customer_id": rp["customer_id"].values,
        "t_order": -age.values,
        "age_d": np.log1p(age.clip(lower=0)).values,
        "gap_d": np.log1p(rp.groupby("customer_id")["due_date"].diff().dt.total_seconds().div(86400)
                          .fillna(0).clip(lower=0)).values,
        "amt_due": slog(rp["amount_due"]),
        "paid_ratio": (rp["amount_paid"] / rp["amount_due"]).clip(0, 1.5).values,
        "unpaid": (unpaid > 0).astype("float32").values,
        "unpaid_share": (unpaid.clip(lower=0) / rp["amount_due"]).clip(0, 1).values,
        "delay": slog(delay.fillna(0).clip(-60, 400)),
        "delay_na": delay.isna().astype("float32").values,
        "dpd_now": np.log1p(rp["days_past_due"]).values,
        "maxdpd": np.log1p(rp["max_days_past_due"]).values,
        "dpd30": (rp["max_days_past_due"] >= 30).astype("float32").values,
        "dpd90": (rp["max_days_past_due"] >= 90).astype("float32").values,
        "inst_k": (k / 12).values,
        "new_loan": (k == 0).astype("float32").values,
        "loan_amt": slog(rp["approved_amount"].fillna(0)),
        "loan_rate": rp["annual_interest_rate"].fillna(0).values,
        "loan_term": (rp["term_months"].fillna(0) / 12).values,
        "loan_age": (days_between(rp["due_date"], rp["decision_datetime"]).fillna(0) / 365).values,
    })
    return out


def build_ab(cut):
    ab = load("account_behavior").merge(cut, on="customer_id", how="inner")
    m_cut = ab["cutoff"].dt.year * 12 + ab["cutoff"].dt.month
    m_row = ab["month"].dt.year * 12 + ab["month"].dt.month
    ab["mago"] = m_cut - m_row
    assert (ab["mago"] >= 1).all()
    ab = ab.sort_values(["customer_id", "month"], kind="stable").reset_index(drop=True)
    flow = ab["inflows"] + ab["outflows"]
    out = pd.DataFrame({"customer_id": ab["customer_id"].values, "t_order": -ab["mago"].values,
                        "mago": np.log1p(ab["mago"]).values})
    for c in ["opening_balance", "closing_balance", "inflows", "outflows", "income_like_credits",
              "average_balance", "minimum_balance"]:
        out[c] = slog(ab[c])
    out["d_balance"] = slog(ab["closing_balance"] - ab["opening_balance"])
    out["net_ratio"] = ((ab["inflows"] - ab["outflows"]) / (flow + 1)).values
    out["inc_share"] = (ab["income_like_credits"] / (ab["inflows"] + 1)).clip(0, 1).values
    out["low_days"] = (ab["low_balance_days"] / 31).values
    out["neg_min"] = (ab["minimum_balance"] < 0).astype("float32").values
    return out


def build_pa(cut, pa):
    p = pa.merge(cut, on="customer_id", how="inner")
    assert (p["decision_datetime"] < p["cutoff"]).all()
    bu = load("bureau_observations")
    bu = bu[bu["application_id"].isin(p["application_id"])]
    b = bu.groupby("application_id").agg(
        bu_n=("score", "size"), bu_score=("score", "mean"), bu_score_na=("score", lambda s: s.isna().mean()),
        bu_active=("active_accounts", "mean"), bu_out=("outstanding_balance", "mean"),
        bu_util=("utilization", "mean"), bu_inq=("inquiries_last_6m", "mean"),
        bu_delinq=("delinquent_accounts_last_12m", "mean"))
    p = p.merge(b, left_on="application_id", right_index=True, how="left")
    p = p.sort_values(["customer_id", "decision_datetime", "application_id"], kind="stable").reset_index(drop=True)
    age = days_between(p["cutoff"], p["decision_datetime"])
    out = pd.DataFrame({
        "customer_id": p["customer_id"].values,
        "t_order": -age.values,
        "age_d": np.log1p(age.clip(lower=0)).values,
        "gap_d": np.log1p(p.groupby("customer_id")["decision_datetime"].diff().dt.total_seconds().div(86400)
                          .fillna(0).clip(lower=0)).values,
        "req_amt": slog(p["requested_amount"]),
        "term": (p["term_months"] / 12).values,
        "dec_wait": np.log1p(days_between(p["decision_datetime"], p["application_datetime"]).clip(lower=0)).values,
        "approved": (p["decision"] == "approved").astype("float32").values,
        "rejected": (p["decision"] == "rejected").astype("float32").values,
        "cancelled": (p["decision"] == "cancelled").astype("float32").values,
        "appr_amt": slog(p["approved_amount"].fillna(0)),
        "appr_ratio": (p["approved_amount"] / p["requested_amount"]).fillna(0).clip(0, 2).values,
        "rate": p["annual_interest_rate"].fillna(0).values,
        "inst": slog(p["scheduled_monthly_installment"].fillna(0)),
        "bu_has": p["bu_n"].notna().astype("float32").values,
        "bu_n": p["bu_n"].fillna(0).values / 3,
        "bu_score": ((p["bu_score"] - 669) / 53).fillna(0).values,
        "bu_score_na": p["bu_score_na"].fillna(1).values,
        "bu_active": np.log1p(p["bu_active"].fillna(0)).values,
        "bu_out": slog(p["bu_out"].fillna(0)),
        "bu_util": p["bu_util"].fillna(-1).clip(-1, 2).values,
        "bu_inq": np.log1p(p["bu_inq"].fillna(0)).values,
        "bu_delinq": np.log1p(p["bu_delinq"].fillna(0)).values,
    })
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    apps = load_apps()
    cut = cutoffs(apps)[["customer_id", "cutoff"]]
    pa = load("previous_applications")
    for name, df in [("rp", build_rp(cut, pa)), ("ab", build_ab(cut)), ("pa", build_pa(cut, pa))]:
        num = df.columns.drop(["customer_id", "t_order"])
        df[num] = df[num].astype("float32")
        assert np.isfinite(df[num].values).all(), name
        n = df.groupby("customer_id").size()
        df.to_parquet(OUT / f"seq_{name}_v1.parquet", index=False)
        print(f"seq_{name}_v1: {len(df):,} baris, {len(num)} fitur, {len(n):,} customer, "
              f"panjang median {int(n.median())} / p99 {int(n.quantile(.99))} / maks {n.max()}, "
              f"{(OUT / f'seq_{name}_v1.parquet').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
