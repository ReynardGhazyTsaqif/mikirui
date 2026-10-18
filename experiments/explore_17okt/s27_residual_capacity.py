"""Putaran 2: masih adakah struktur interaksi yang belum dipakai model acuan (E200: pohon depth 4, ff 0,13)?

Residual boosting di atas OOF E200 (init_score = logit OOF) dengan SEMUA kolom set E200 (528 + rp_ev + fitur nested
versi OOF) dan pohon lebih dalam. Dibandingkan beberapa kapasitas. Kenaikan nyata -> ada kapasitas/interaksi tersisa
(arah H07 sub-model / H12 tuning); ~0 -> model jenuh terhadap fitur yang ada.
Catatan: seperti screen.py, ini tidak nested (sedikit optimis).
"""
import gc
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402
from common import bag_oof  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load_apps  # noqa: E402
from src.train import make_folds  # noqa: E402

USE = ["e121_ordermodel_v1.npz", "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",
       "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz"]
apps = load_apps()
m = apps.is_test.eq(0).values
X = run.build_matrix(list(GROUPS), apps)
E = pd.read_parquet(FEATURES_DIR / "rp_events_v1.parquet").reindex(apps.application_id)
X = pd.concat([X, E], axis=1)[m].reset_index(drop=True)
del E
y = apps.loc[m, "target"].astype(int).values
fold = make_folds(apps[m])["fold"].values
for i, f in enumerate(USE):  # fitur nested versi OOF: baris fold k = va{k}
    z = np.load(FEATURES_DIR / f, allow_pickle=True)
    arr = np.full((len(y), len(z["cols"])), np.nan, dtype="float32")
    for k in range(5):
        arr[fold == k] = z[f"va{k}"]
    for j, c in enumerate(z["cols"]):
        X[f"{c}{i}"] = arr[:, j]
gc.collect()
base, n = bag_oof("E200")
init = np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
configs = {"depth3_l7": dict(num_leaves=7, max_depth=3), "depth6_l63": dict(num_leaves=63, max_depth=6),
           "depth8_l255": dict(num_leaves=255, max_depth=8)}
for name, cfg in configs.items():
    p = dict(objective="binary", metric="auc", learning_rate=0.03, min_child_samples=200, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, deterministic=True, force_row_wise=True,
             num_threads=4, verbose=-1, **cfg)
    d = []
    for k in range(5):
        tr, va = fold != k, fold == k
        dtr = lgb.Dataset(X[tr], y[tr], init_score=init[tr])
        dva = lgb.Dataset(X[va], y[va], init_score=init[va], reference=dtr)
        mdl = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        new = init[va] + mdl.predict(X[va], raw_score=True, num_iteration=mdl.best_iteration)
        d.append(roc_auc_score(y[va], new) - roc_auc_score(y[va], base[va]))
        print(f"  {name} fold {k}: delta {d[-1]:+.5f} iter {mdl.best_iteration}", flush=True)
    print(f"{name}: delta mean-fold {np.mean(d):+.5f}, naik {sum(x > 0 for x in d)}/5", flush=True)
