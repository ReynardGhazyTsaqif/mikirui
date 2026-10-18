"""Putaran 2, H14: bagaimana lender menetapkan tier bunga sekarang, dan apakah "kejutan tier" membawa sinyal di luar model?

Tier sekarang jauh lebih prediktif untuk target sekarang (koef. 0,77) daripada tier lama untuk default pinjaman lama
(0,15) -> mungkin tier = ember(skor lender) dan skor lender memuat informasi privat. Jika tier = ember(g(X) + e), maka
P(y | tier, X) bergantung pada g(X) relatif ambang tier; pohon sulit merakit g(X) dari ratusan kolom.
Cara: LightGBM (tanpa target) memprediksi tier dari semua kolom set E200 kecuali turunan bunga/cicilan pinjaman sekarang,
cross-fit 5 fold atas 100 rb aplikasi (train + test). Lalu z residual OOF E202 untuk prediksi, residual, dan kejutan.
"""
import gc
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from resid import reference, table  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load_apps  # noqa: E402

apps = load_apps()
X = run.build_matrix(list(GROUPS), apps)
E = pd.read_parquet(FEATURES_DIR / "rp_events_v1.parquet").reindex(apps.application_id)
X = pd.concat([X, E], axis=1).reset_index(drop=True)
num = X.select_dtypes(exclude="category").columns  # kolom kategori tetap category (LightGBM)
X[num] = X[num].astype("float32")
rate = X["app_annual_interest_rate"].values
tiers = np.sort(np.unique(rate))
lab = np.searchsorted(tiers, rate)
drop = [c for c in X.columns if re.search("rate|inst|pti|interest|residual_income", c)
        and not c.startswith(("pa_", "rp_", "bp_"))]
print("kolom dibuang (turunan bunga/cicilan sekarang):", drop)
X = X.drop(columns=drop)
cid = apps.customer_id.values
del E
gc.collect()
pr = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.5,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, deterministic=True, force_row_wise=True,
          num_threads=4, verbose=-1)
pm = {**pr, "objective": "multiclass", "num_class": len(tiers), "metric": "multi_logloss"}
pred, prob = np.zeros(len(X)), np.zeros((len(X), len(tiers)))
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
    tr2, es = tr[: int(len(tr) * 0.9)], tr[int(len(tr) * 0.9):]
    d1, d2 = lgb.Dataset(X.iloc[tr2], lab[tr2]), lgb.Dataset(X.iloc[es], lab[es])
    mr = lgb.train(pr, d1, 3000, valid_sets=[d2], callbacks=[lgb.early_stopping(100, verbose=False)])
    pred[va] = mr.predict(X.iloc[va], num_iteration=mr.best_iteration)
    mc = lgb.train(pm, d1, 3000, valid_sets=[d2], callbacks=[lgb.early_stopping(100, verbose=False)])
    prob[va] = mc.predict(X.iloc[va], num_iteration=mc.best_iteration)
    print(f"fold {k}: iter reg {mr.best_iteration}, multiclass {mc.best_iteration}", flush=True)
acc = (prob.argmax(1) == lab).mean()
r2 = 1 - ((lab - pred) ** 2).mean() / lab.var()
print(f"prediksi tier dari fitur lain: akurasi multiclass {acc:.3f}, R2 ordinal {r2:.3f}")
print("matriks tier asli (baris) x prediksi (kolom), porsi baris:")
print(pd.crosstab(lab, prob.argmax(1), normalize="index").round(3))
imp = pd.Series(mr.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("10 kolom terpenting untuk tier:", (imp.head(10) / imp.sum()).round(3).to_dict())
F = pd.DataFrame({"customer_id": cid, "t_pred": pred, "t_resid": lab - pred,
                  "t_surprise": -np.log(np.clip(prob[np.arange(len(lab)), lab], 1e-9, 1)),
                  "t_exp": prob @ np.arange(len(tiers))})
F["t_resid_exp"] = lab - F.t_exp
F["t_rank_in_tier"] = F.groupby(lab).t_pred.rank(pct=True)
F.to_parquet(HERE / "s32_tier.parquet")
ref = reference("E202")
Fi = F.set_index("customer_id")
print(table(ref, {c: Fi[c] for c in Fi.columns}).to_string(index=False))
