"""Putaran 2, H07 pendahuluan: sub-model per sumber bukti + stacking, sebelum membangun versi nested yang mahal.

Untuk tiap sumber (kolom berawalan rp_/bu_+bp_/pa_/ab_/ec_/app_+prof_ dari set E200, tanpa fitur nested), latih LightGBM
5-fold (fold tim) -> skor OOF per sumber. Lalu:
  1. AUC tiap sumber dan z residual terhadap OOF E202 (keduanya OOF, tidak bocor);
  2. stacking logistik lintas fold: logit E202 + logit 6 sumber, koefisien dilatih di 4 fold, dinilai di fold ke-5.
Early stopping memakai fold validasi (sedikit optimis) -> kalau stacking pun < +0,0003, H07 ditolak tanpa versi nested.
Keluaran: experiments/explore_17okt/s31_source_oof.parquet + cetak tabel.
"""
import gc
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from resid import reference, table  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load_apps  # noqa: E402
from src.train import make_folds  # noqa: E402

SOURCES = {"rp": ("rp_",), "bu": ("bu_", "bp_"), "pa": ("pa_",), "ab": ("ab_",), "ec": ("ec_",),
           "ap": ("app_", "prof_")}
apps = load_apps()
m = apps.is_test.eq(0).values
X = run.build_matrix(list(GROUPS), apps)
E = pd.read_parquet(FEATURES_DIR / "rp_events_v1.parquet").reindex(apps.application_id)
X = pd.concat([X, E], axis=1)[m].reset_index(drop=True)
num = X.select_dtypes(exclude="category").columns  # kolom kategori tetap category (LightGBM)
X[num] = X[num].astype("float32")
y = apps.loc[m, "target"].astype(int).values
ids = apps.loc[m, "application_id"].values
fold = make_folds(apps[m])["fold"].values
del E, apps
gc.collect()
p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31, max_depth=5, min_child_samples=100,
         feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, deterministic=True,
         force_row_wise=True, num_threads=4, verbose=-1)
S = pd.DataFrame(index=range(len(y)))
for s, pre in SOURCES.items():
    cols = [c for c in X.columns if c.startswith(pre)]
    oof = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        dtr = lgb.Dataset(X.loc[tr, cols], y[tr])
        dva = lgb.Dataset(X.loc[va, cols], y[va], reference=dtr)
        mdl = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = mdl.predict(X.loc[va, cols], raw_score=True, num_iteration=mdl.best_iteration)
    S[s] = oof
    aucs = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(5)]
    print(f"sumber {s}: {len(cols)} kolom, mean-fold AUC {np.mean(aucs):.4f}", flush=True)
S["application_id"] = ids
S.to_parquet(HERE / "s31_source_oof.parquet")
ref = reference("E202")
assert (ref.application_id.values == ids).all()
cid = ref.customer_id.values
print(table(ref, {f"src_{s}": pd.Series(S[s].values, index=cid) for s in SOURCES}).to_string(index=False))
Z = np.column_stack([ref.lg.values] + [S[s].values for s in SOURCES])
base, stack = [], []
for k in range(5):
    tr, va = fold != k, fold == k
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], y[tr])
    base.append(roc_auc_score(y[va], Z[va, 0]))
    stack.append(roc_auc_score(y[va], lr.decision_function(Z[va])))
d = np.array(stack) - np.array(base)
print(f"stacking E202 + 6 sumber: mean-fold {np.mean(base):.5f} -> {np.mean(stack):.5f}, delta {d.mean():+.5f}, "
      f"naik {(d > 0).sum()}/5, per fold {np.round(d, 5).tolist()}")
print("koefisien (fold terakhir):", dict(zip(["E202"] + list(SOURCES), np.round(lr.coef_[0], 3))))
