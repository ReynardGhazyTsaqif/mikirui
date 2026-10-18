"""E203 (H12 pendahuluan): sensitivitas parameter LightGBM di set E200, sebelum memutuskan Optuna 5 fold.

Parameter tuned (study tune_lgb) dipilih di set B3 (176 kolom, fold 0-2, lr 0,05) lalu dipakai di 559 kolom. Script ini
membandingkan beberapa variasi (lr 0,05, early stopping 200, 1 seed, 5 fold tetap, fitur sama dengan E200) untuk melihat
apakah ada arah perbaikan yang jelas. Bukan keputusan akhir: variasi yang menjanjikan diulang dengan lr final 3 seed.

  python e203_sens.py --use <npz+...> --extra rp_events_v1.parquet
Keluaran: experiments/e203_sens.csv
"""
import argparse
import gc
import sys
import time
from argparse import Namespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run
from src.config import FEATURES_DIR, N_FOLDS, ROOT
from src.features.registry import GROUPS
from src.io_utils import load_apps
from src.train import make_folds
from src.tuning import full_params


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--use", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from tune import Ctx, best_tuned
    ctx = Ctx.__new__(Ctx)
    ctx.storage = f"sqlite:///{(ROOT / 'experiments' / 'p5' / 'optuna.db').as_posix()}"
    ctx.smoke = False
    tuned, _ = best_tuned(ctx, "lgb", 15)
    base = full_params("lgb", tuned, 0.05)
    apps = load_apps()
    m = apps["is_test"].eq(0).values
    X = run.build_matrix(list(GROUPS), apps)
    if a.extra:
        E = pd.concat([pd.read_parquet(FEATURES_DIR / f).reindex(apps["application_id"]) for f in a.extra.split("+")], axis=1)
        X = pd.concat([X.set_index(apps["application_id"]), E], axis=1).reset_index(drop=True)
    X_tr = X[m].reset_index(drop=True)
    y = apps.loc[m, "target"].astype(int).values
    fold = make_folds(apps[m])["fold"].values
    del X, apps
    gc.collect()
    zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in a.use.split("+")]
    cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
    z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
    configs = {
        "tuned": {},
        "ff0.08": {"feature_fraction": 0.08},
        "ff0.20": {"feature_fraction": 0.20},
        "ff0.30": {"feature_fraction": 0.30},
        "depth5": {"max_depth": 5},
        "depth6_l63": {"max_depth": 6, "num_leaves": 63},
        "no_extra_trees": {"extra_trees": False},
        "minchild100": {"min_child_samples": 100},
    }
    rows = []
    for name, over in configs.items():
        p = {**base, **over, "seed": a.seed}
        t0 = time.time()
        aucs, its = [], []
        for k in range(N_FOLDS):
            tr, va = fold != k, fold == k
            A_ = pd.concat([X_tr[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
            B_ = pd.concat([X_tr[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
            dtr = lgb.Dataset(A_, y[tr], params={"max_bin": p.get("max_bin", 255)})
            dva = lgb.Dataset(B_, y[va], reference=dtr)
            mdl = lgb.train(p, dtr, 15000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            aucs.append(roc_auc_score(y[va], mdl.predict(B_, num_iteration=mdl.best_iteration)))
            its.append(mdl.best_iteration)
            del A_, B_, dtr, dva, mdl
            gc.collect()
        rows.append(dict(config=name, mean_fold_auc=np.mean(aucs), **{f"f{k}": aucs[k] for k in range(N_FOLDS)},
                         iters=int(np.mean(its)), menit=round((time.time() - t0) / 60, 1)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "experiments" / "e203_sens.csv", index=False)
    print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
