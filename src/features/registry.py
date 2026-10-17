"""Daftar semua grup fitur: nama grup (= nama file cache features/<nama>.parquet) -> builder.

Naikkan versi (_v2) setiap kali logika builder berubah agar cache lama tidak terpakai.
"""
from . import account, app, bureau, cross, ecommerce, events, naive, previous, repayment

GROUPS = {
    # Fase 1
    "app_raw_v1": app.build_app_raw,
    "prof_raw_v1": app.build_prof_raw,
    "app_ratios_v1": app.build_app_ratios,
    "pa_naive_v1": naive.build_pa_naive,
    "rp_naive_v1": naive.build_rp_naive,
    "bu_naive_v1": naive.build_bu_naive,
    "bp_naive_v1": naive.build_bp_naive,
    "ab_naive_v1": naive.build_ab_naive,
    "ec_naive_v1": naive.build_ec_naive,
    # Fase 2
    "rp_core_v1": repayment.build_rp_core,
    "rp_window_v1": repayment.build_rp_window,
    "rp_loan_v1": repayment.build_rp_loan,
    "rp_trend_v1": repayment.build_rp_trend,
    "bu_extra_v1": bureau.build_bu_extra,
    "bp_trend_v1": bureau.build_bp_trend,
    "pa_window_v1": previous.build_pa_window,
    "pa_burden_v1": previous.build_pa_burden,
    "ab_window_v1": account.build_ab_window,
    "ab_income_v1": account.build_ab_income,
    "ec_window_v1": ecommerce.build_ec_window,
    "ec_mix_v1": ecommerce.build_ec_mix,
    "x_cross_v1": cross.build_x_cross,
    "x_rate_resid_v1": cross.build_x_rate_resid,
}

# Grup fitur eksperimen 17 Okt+ (E200+). Sengaja TIDAK masuk GROUPS supaya set acuan "all" (528 kolom,
# `e121.py --basefs all`, kaggle X_all.parquet) tidak berubah; dipakai lewat `e121.py --extra <nama>.parquet`.
# Bangun cache: python build_extra.py <nama>
EXTRA_GROUPS = {
    "rp_events_v1": events.build_rp_events,  # H01: rantai bayar bulanan level event
}
