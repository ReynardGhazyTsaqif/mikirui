"""PROMPT 3 Langkah 8 (E145): validasi aturan iterasi refit di CV.

Di tiap fold, model LGB tuned dengan iterasi TETAP (rata-rata best_iter early stopping x faktor, tanpa early stopping)
dibandingkan dengan model early stopping biasa. Syarat refit 100% data: selisih mean-fold AUC <= 0,0002 untuk faktor
1,0 (ukuran data sama). Faktor 1,15 dilaporkan sebagai sensitivitas (refit memakai 25% data lebih banyak).

  python experiments/prompt3/p3_refit_check.py --use <npz+...> --extra rp_events_v1.parquet+ab_nonin_v1.parquet --seed 42
"""
import argparse
import gc
import sys
from argparse import Namespace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import FEATURES_DIR  # noqa: E402
from src.io_utils import load_apps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from tune import Ctx, best_tuned
    from src.tuning import FINAL_LR, full_params
    import run
    from src.features.registry import GROUPS
    ctx = Ctx(Namespace(smoke=False, stop_at=None))
    del ctx.Xb2_tr, ctx.Xb2_te
    apps = load_apps()
    m = apps["is_test"].eq(0).values
    X = run.build_matrix(list(GROUPS), apps)
    if a.extra:
        E = pd.concat([pd.read_parquet(FEATURES_DIR / f).reindex(apps["application_id"]) for f in a.extra.split("+")], axis=1)
        X = pd.concat([X, E], axis=1)
    X_tr = X[m].reset_index(drop=True)
    del X, apps
    gc.collect()
    tuned, _ = best_tuned(ctx, "lgb", 15)
    params = {**full_params("lgb", tuned, FINAL_LR["lgb"]), "seed": a.seed}
    zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in a.use.split("+")]
    cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
    z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
    y, fold = ctx.y, ctx.fold
    es_auc, es_it, data = [], [], []
    for k in range(5):
        tr, va = fold != k, fold == k
        A = pd.concat([X_tr[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
        B = pd.concat([X_tr[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
        dtr = lgb.Dataset(A, y[tr], params={"max_bin": params.get("max_bin", 255)}, free_raw_data=False)
        mdl = lgb.train(params, dtr, 15000, valid_sets=[lgb.Dataset(B, y[va], reference=dtr)],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        es_it.append(mdl.best_iteration)
        es_auc.append(roc_auc_score(y[va], mdl.predict(B, num_iteration=mdl.best_iteration)))
        data.append((A, B, tr, va))
        print(f"fold {k}: early stopping iter {mdl.best_iteration}, AUC {es_auc[-1]:.5f}", flush=True)
    for factor in (1.0, 1.15):
        n_it = int(round(np.mean(es_it) * factor))
        fx = []
        for k, (A, B, tr, va) in enumerate(data):
            mdl = lgb.train(params, lgb.Dataset(A, y[tr], params={"max_bin": params.get("max_bin", 255)}), n_it)
            fx.append(roc_auc_score(y[va], mdl.predict(B)))
        d = np.array(fx) - np.array(es_auc)
        print(f"faktor {factor}: iterasi tetap {n_it} -> mean-fold {np.mean(fx):.5f} vs early stopping {np.mean(es_auc):.5f}, "
              f"selisih {d.mean():+.5f} (per fold {np.round(d, 5).tolist()}); syarat |selisih| <= 0,0002 untuk faktor 1,0",
              flush=True)


if __name__ == "__main__":
    main()
