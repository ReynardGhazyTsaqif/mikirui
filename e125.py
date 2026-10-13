"""E125: fitur hasil prediksi ("predicted features") dengan label bantu, BUKAN target.

Untuk customer tanpa riwayat (30%) informasi riwayat/rekening kosong. Model bantu belajar memetakan
informasi yang dimiliki SEMUA customer (profil, aplikasi, biro saat ini, e-commerce) ke ringkasan riwayat
(DPD terburuk, rasio bayar, porsi ditolak, income rekening, hari saldo rendah). Prediksinya dipakai sebagai
fitur untuk semua customer, residual (aktual - prediksi) untuk yang punya data. Pola sama dengan
x_rate_pred (src/features/cross.py): cross-fit 5 fold acak atas 100k baris, tanpa target.

  python e125.py            # -> features/x_pred_v1.parquet (index application_id, prefix x_pred_)
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import FEATURES_DIR, N_THREADS, SEED
from src.features.common import cached
from src.features.registry import GROUPS
from src.io_utils import load_apps

OUT = FEATURES_DIR / "x_pred_v1.parquet"
INPUTS = ["prof_raw_v1", "app_raw_v1", "app_ratios_v1", "bu_naive_v1", "bu_extra_v1", "ec_naive_v1",
          "ec_window_v1", "ec_mix_v1"]
P = dict(learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, seed=SEED, deterministic=True, force_row_wise=True,
         num_threads=N_THREADS, verbose=-1)


def labels(apps):
    rp, pa, ab = (cached(n, GROUPS[n], apps) for n in ("rp_naive_v1", "pa_naive_v1", "ab_naive_v1"))
    has_rp = rp["rp_n_installments"].fillna(0) > 0
    L = pd.DataFrame(index=rp.index)
    L["rp_bad90"] = (rp["rp_max_days_past_due_max"] >= 90).astype(float).where(has_rp)
    L["rp_maxdpd"] = np.log1p(rp["rp_max_days_past_due_max"]).where(has_rp)
    L["rp_paidratio"] = rp["rp_paid_ratio_total"].where(has_rp)
    L["pa_rejshare"] = pa["pa_share_rejected"].where(pa["pa_count"].fillna(0) > 0)
    L["ab_inc"] = np.log1p(ab["ab_income_like_credits_mean"].clip(lower=0))
    L["ab_lowbal"] = ab["ab_low_balance_days_mean"]
    return L


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    apps = load_apps()
    X = pd.concat([cached(n, GROUPS[n], apps) for n in INPUTS], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    L = labels(apps).reindex(X.index)
    fold = np.random.RandomState(SEED).randint(0, 5, len(X))
    out = pd.DataFrame(index=X.index)
    for c in L.columns:
        y = L[c].values
        has = ~np.isnan(y)
        binary = set(np.unique(y[has])) <= {0.0, 1.0}
        params = {**P, "objective": "binary" if binary else "regression"}
        pred = np.zeros(len(X))
        for k in range(5):
            tr = (fold != k) & has
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 600)
            pred[fold == k] = m.predict(X[fold == k])
        r = np.corrcoef(pred[has], y[has])[0, 1]
        print(f"{c}: n label {has.sum()}, korelasi prediksi-aktual (cross-fit) {r:.3f}", flush=True)
        out[f"x_pred_{c}"] = pred
        out[f"x_pred_{c}_resid"] = np.where(has, y - pred, np.nan)
    out.astype("float32").to_parquet(OUT)
    print(f"disimpan {OUT.name} {out.shape}")


if __name__ == "__main__":
    main()
