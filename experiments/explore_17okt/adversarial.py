"""Adversarial validation train vs test untuk parquet fitur (audit-leakage Lapis 2), opsional + 528 kolom acuan.

  python experiments/explore_17okt/adversarial.py features/rp_events_v1.parquet [--with-base]
AUC ~ 0,50 = aman; > 0,55 = ada fitur yang membedakan train/test.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.io_utils import load_apps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("features", nargs="+")
ap.add_argument("--with-base", action="store_true", help="tambahkan 528 kolom set acuan")
a = ap.parse_args()
apps = load_apps()
F = pd.concat([pd.read_parquet(ROOT / f).reindex(apps.application_id) for f in a.features], axis=1)
if a.with_base:
    import run
    from src.features.registry import GROUPS
    F = pd.concat([run.build_matrix(list(GROUPS), apps).set_index(apps.application_id), F], axis=1)
y = apps.is_test.values
fold = np.random.RandomState(42).randint(0, 5, len(y))
params = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, seed=42, deterministic=True, force_row_wise=True, num_threads=4, verbose=-1)
oof = np.zeros(len(y))
imp = pd.Series(0.0, index=F.columns)
for k in range(5):
    tr, va = fold != k, fold == k
    m = lgb.train(params, lgb.Dataset(F[tr], y[tr]), 300)
    oof[va] = m.predict(F[va])
    imp += pd.Series(m.feature_importance("gain"), index=F.columns)
auc = roc_auc_score(y, oof)
print(f"adversarial AUC {auc:.4f} ({F.shape[1]} kolom) -> {'aman' if auc < 0.55 else 'PERIKSA'}")
print("kolom paling membedakan:", imp.sort_values(ascending=False).head(5).round(0).to_dict())
