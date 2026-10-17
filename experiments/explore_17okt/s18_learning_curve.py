"""Langkah 2q: kurva belajar. Berapa AUC naik jika data latih bertambah?

Set fitur E130a (528 kolom + 7 ecm_* nested dari features/e121_ordermodel_v1.npz), parameter LGB tuned (tune_lgb)
dengan lr 0,05 (cepat), early stopping 200 pada fold validasi. Untuk tiap fold k (0..4), data latih fold != k
diambil subsampel 25/50/75/100% (customer acak, seed 42); fitur ecm baris latih = tr{k} (nested asli).
Keluaran: experiments/explore_17okt/learning_curve.csv
"""
import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import MAIN, apps

sys.path.insert(0, str(MAIN))
A = apps()
m = (A.is_test == 0).values
fold = A.fold.values[m]
y = A.target.values[m].astype(int)
import glob
parts = []
for f in sorted(glob.glob(str(MAIN / "features" / "*_v1.parquet"))):
    n = f.split("\\")[-1].split("/")[-1][:-8]
    if n.startswith(("ec_orders", "ec_items", "ec_rows", "e126", "x_pred")):
        continue
    parts.append(pd.read_parquet(f))
X = pd.concat(parts, axis=1)
X = X[sorted(X.columns)]
assert X.shape[1] == 528, X.shape
X = X.reindex(A.application_id)
Xtr = X[m].reset_index(drop=True)
del X, parts
gc.collect()
z = np.load(MAIN / "features" / "e121_ordermodel_v1.npz", allow_pickle=True)
cols = [f"{c}0" for c in z["cols"]]
params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 35, "max_depth": 4,
          "min_child_samples": 34, "feature_fraction": 0.12870618888259533, "bagging_fraction": 0.9264683082866251,
          "bagging_freq": 1, "lambda_l2": 7.664965525069028, "lambda_l1": 0.011885886537829151,
          "min_gain_to_split": 0.9382501776308856, "extra_trees": True, "max_bin": 255, "seed": 42,
          "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1}
rows = []
for k in range(5):
    tr, va = fold != k, fold == k
    Atr = pd.concat([Xtr[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
    Bva = pd.concat([Xtr[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
    ytr, yva = y[tr], y[va]
    rng = np.random.default_rng(42 + k)
    perm = rng.permutation(len(ytr))
    for frac in (0.25, 0.5, 0.75, 1.0):
        idx = np.sort(perm[: int(len(ytr) * frac)])
        dtr = lgb.Dataset(Atr.iloc[idx], ytr[idx], params={"max_bin": 255})
        dva = lgb.Dataset(Bva, yva, reference=dtr)
        mdl = lgb.train(params, dtr, 8000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        auc = roc_auc_score(yva, mdl.predict(Bva, num_iteration=mdl.best_iteration))
        rows.append(dict(fold=k, frac=frac, n_train=len(idx), auc=auc, iters=mdl.best_iteration))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("learning_curve.csv", index=False)
    del Atr, Bva
    gc.collect()
r = pd.DataFrame(rows)
print(r.groupby("frac").agg(n=("n_train", "mean"), auc=("auc", "mean"), iters=("iters", "mean")).round(5))
