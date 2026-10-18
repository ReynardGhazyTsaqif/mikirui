"""Putaran 2: kurva belajar ulang di set E200 (528 + rp_ev + 7 npz nested), sama seperti s18 (set E130a).

Pertanyaan: setelah representasi rantai bayar masuk, berapa asimtot representasi sekarang dan berapa sisa kenaikan
dari "lebih banyak data" (10-fold, refit 100%)? LGB tuned lr 0,05, early stopping 200 di fold validasi, 1 seed.
Data latih fold != k disubsampel 25/50/75/100% (seed 42 + k); fitur nested baris latih = tr{k}.
Keluaran: experiments/explore_17okt/learning_curve_e200.csv + fit kurva pangkat AUC = a - b * n^-c.
"""
import gc
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run  # noqa: E402
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
X = pd.concat([X.set_index(apps.application_id), E], axis=1)[m].reset_index(drop=True)
y = apps.loc[m, "target"].astype(int).values
fold = make_folds(apps[m])["fold"].values
del E, apps
gc.collect()
zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in USE]
cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 35, "max_depth": 4,
          "min_child_samples": 34, "feature_fraction": 0.12870618888259533, "bagging_fraction": 0.9264683082866251,
          "bagging_freq": 1, "lambda_l2": 7.664965525069028, "lambda_l1": 0.011885886537829151,
          "min_gain_to_split": 0.9382501776308856, "extra_trees": True, "max_bin": 255, "seed": 42,
          "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1}
rows = []
for k in range(5):
    tr, va = fold != k, fold == k
    Atr = pd.concat([X[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
    Bva = pd.concat([X[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
    ytr, yva = y[tr], y[va]
    perm = np.random.default_rng(42 + k).permutation(len(ytr))
    for frac in (0.25, 0.5, 0.75, 1.0):
        idx = np.sort(perm[: int(len(ytr) * frac)])
        dtr = lgb.Dataset(Atr.iloc[idx], ytr[idx], params={"max_bin": 255})
        dva = lgb.Dataset(Bva, yva, reference=dtr)
        mdl = lgb.train(params, dtr, 8000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        auc = roc_auc_score(yva, mdl.predict(Bva, num_iteration=mdl.best_iteration))
        rows.append(dict(fold=k, frac=frac, n_train=len(idx), auc=auc, iters=mdl.best_iteration))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(HERE / "learning_curve_e200.csv", index=False)
        del dtr, dva, mdl
    del Atr, Bva
    gc.collect()
r = pd.DataFrame(rows)
g = r.groupby("frac").agg(n=("n_train", "mean"), auc=("auc", "mean"), iters=("iters", "mean"))
print(g.round(5))
f = lambda n, a, b, c: a - b * n ** (-c)  # noqa: E731
(a, b, c), _ = curve_fit(f, g.n.values, g.auc.values, p0=(0.85, 3.0, 0.6), maxfev=20000)
print(f"kurva pangkat: AUC = {a:.4f} - {b:.3f} * n^-{c:.3f}")
for n in (63000, 70000, 112000, 140000):
    print(f"  n={n}: {f(n, a, b, c):.4f}")
