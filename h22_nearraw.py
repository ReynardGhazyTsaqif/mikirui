"""H22 (mentor ide 14), bagian lokal: model berbeda dengan preprocessing minimal, fold tim yang sama.

Fitur "hampir mentah" = grup agregat naif/mentah saja (app_raw, prof_raw, pa/rp/bu/bp/ab/ec_naive), tanpa fitur
rekayasa lanjutan dan tanpa fitur nested. Model: LGB, XGB, CatBoost dengan parameter tuned (study tune_<model>),
5 fold, 1 seed. Tujuan: (1) cek apakah rekayasa fitur kita membuang informasi (stacking OOF mentah di atas E202),
(2) batas kasar, (3) varian berbeda untuk portofolio.

  python h22_nearraw.py --models lgb xgb cat
Keluaran: oof/H22_<model>_s42.npy, preds/H22_<model>_s42.npy, log experiments/h22.log
"""
import argparse
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

import run
from src.config import N_FOLDS, ROOT
from src.io_utils import load_apps
from src.train import ks_score, make_folds
from src.tuning import FINAL_LR, fit_fold, full_params

NAIVE = ["app_raw_v1", "prof_raw_v1", "pa_naive_v1", "rp_naive_v1", "bu_naive_v1", "bp_naive_v1", "ab_naive_v1",
         "ec_naive_v1"]
LOG = ROOT / "experiments" / "h22.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lgb", "xgb", "cat"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from tune import Ctx, best_tuned
    ctx = Ctx.__new__(Ctx)
    ctx.storage = f"sqlite:///{(ROOT / 'experiments' / 'p5' / 'optuna.db').as_posix()}"
    ctx.smoke = False
    apps = load_apps()
    m = apps["is_test"].eq(0).values
    X = run.build_matrix(NAIVE, apps)
    X_tr, X_te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
    y = apps.loc[m, "target"].astype(int).values
    fold = make_folds(apps[m])["fold"].values
    log(f"=== H22 fitur hampir mentah: {X_tr.shape[1]} kolom dari {NAIVE}")
    for model in a.models:
        tuned, _ = best_tuned(ctx, model, 15)
        params = full_params(model, tuned, FINAL_LR[model])
        oof, pred, aucs, its = np.zeros(len(y)), np.zeros(len(X_te)), [], []
        t0 = time.time()
        for k in range(N_FOLDS):
            tr, va = fold != k, fold == k
            p_va, p_te, it = fit_fold(model, params, X_tr[tr], y[tr], X_tr[va], y[va], X_te, a.seed, rounds=15000)
            oof[va], pred = p_va, pred + p_te / N_FOLDS
            aucs.append(roc_auc_score(y[va], p_va))
            its.append(it)
        np.save(ROOT / "oof" / f"H22_{model}_s{a.seed}.npy", oof)
        np.save(ROOT / "preds" / f"H22_{model}_s{a.seed}.npy", pred)
        log(f"H22_{model} seed {a.seed}: AUC mean-fold {np.mean(aucs):.5f} {np.round(aucs, 4).tolist()} KS {ks_score(y, oof):.4f} "
            f"iter {its} ({(time.time() - t0) / 60:.1f} mnt)")


if __name__ == "__main__":
    main()
