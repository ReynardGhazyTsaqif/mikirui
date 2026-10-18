"""Fitur rekening tambahan (ab_nonin_), H21 eksplorasi 17 Okt (E204, kandidat).

Inflow non-gaji per bulan rekening = inflows - income_like_credits (bulan rekening selalu < bulan keputusan). Rasio
terhadap income deklarasi profil dan terhadap total inflow: rata-rata semua bulan, sd, dan rata-rata 3 bulan terakhir
(mago <= 3; mago = bulan kalender keputusan - bulan rekening). Tidak memakai target. Hasil identik dengan
features/cand_h21_v1.parquet yang dipakai E204.
"""
import numpy as np

from ..io_utils import cutoffs, load
from .common import by_customer


def build_ab_nonin(apps):
    ab = load("account_behavior", usecols=["customer_id", "month", "inflows", "income_like_credits"])
    ab = ab.merge(cutoffs(apps)[["customer_id", "cutoff"]], on="customer_id")
    inc = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"]).set_index("customer_id")
    ab["mago"] = (ab.cutoff.dt.year * 12 + ab.cutoff.dt.month) - (ab.month.dt.year * 12 + ab.month.dt.month)
    nonin = ab.inflows - ab.income_like_credits
    ab["nonin"] = nonin / ab.customer_id.map(inc.declared_monthly_income)
    ab["nonin_share"] = nonin / ab.inflows.replace(0, np.nan)
    g, g3 = ab.groupby("customer_id"), ab[ab.mago <= 3].groupby("customer_id")
    agg = g.nonin.mean().rename("nonin_inc_mean").to_frame()
    agg["nonin_inc_std"] = g.nonin.std()
    agg["nonin_inc_last3"] = g3.nonin.mean()
    agg["nonin_share_mean"] = g.nonin_share.mean()
    agg["nonin_share_last3"] = g3.nonin_share.mean()
    return by_customer(agg, apps, "ab_")
