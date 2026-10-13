"""E146: tuning ulang Optuna LightGBM pada set fitur E130a (528 kolom semua grup + 7 fitur ecm_* nested).

Ruang pencarian sama dengan Fase 5 (src/tuning.suggest), lr pencarian 0.05, fold 0-2, MedianPruner.
Fitur ecm_* berbeda per fold (nested), jadi objective dirakit per fold dari features/e121_ordermodel_v1.npz.
Study: tune_lgb_e146 di experiments/p5/optuna.db (bisa dilanjutkan). Trial terbaik Fase 5 dimasukkan
sebagai trial pertama (enqueue) supaya pencarian mulai dari titik yang sudah bagus.

  python tune_e146.py --trials 60 --budget 60
Final: python e121.py --cv --use e121_ordermodel_v1.npz --name E146 --basefs all --study lgb_e146
"""
import argparse
import gc
import sys
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

import run
from src.config import FEATURES_DIR, ROOT, SEED
from src.features.registry import GROUPS
from src.io_utils import load_apps
from src.train import make_folds
from src.tuning import SEARCH_LR, full_params, suggest

STORAGE = f"sqlite:///{(ROOT / 'experiments' / 'p5' / 'optuna.db').as_posix()}"
LOG = ROOT / "experiments" / "tune_e146.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--budget", type=float, default=60, help="menit")
    a = ap.parse_args()
    apps = load_apps()
    m = apps["is_test"].eq(0).values
    X = run.build_matrix(list(GROUPS), apps)[m].reset_index(drop=True)
    y = apps.loc[m, "target"].astype(int).values
    fold = make_folds(apps[m])["fold"].values
    del apps
    gc.collect()
    z = np.load(FEATURES_DIR / "e121_ordermodel_v1.npz", allow_pickle=True)
    cols = [f"{c}0" for c in z["cols"]]  # nama sama dengan e121.cv
    data = {}
    for k in (0, 1, 2):
        tr, va = fold != k, fold == k
        data[k] = (pd.concat([X[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1),
                   y[tr],
                   pd.concat([X[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1),
                   y[va])
    del X
    gc.collect()

    def objective(trial):
        params = full_params("lgb", suggest("lgb", trial), SEARCH_LR["lgb"])
        aucs = []
        for step, k in enumerate((0, 1, 2)):
            X_tr, y_tr, X_va, y_va = data[k]
            dtr = lgb.Dataset(X_tr, y_tr, params={"max_bin": params.get("max_bin", 255)})
            dva = lgb.Dataset(X_va, y_va, reference=dtr)
            mdl = lgb.train({**params, "seed": SEED}, dtr, 5000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
            aucs.append(roc_auc_score(y_va, mdl.predict(X_va, num_iteration=mdl.best_iteration)))
            trial.report(float(np.mean(aucs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="tune_lgb_e146", storage=STORAGE, load_if_exists=True,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    if len(study.trials) == 0:
        study.enqueue_trial(optuna.load_study(study_name="tune_lgb", storage=STORAGE).best_trial.params)
    start = time.time()
    log(f"=== MULAI E146, sudah {len(study.trials)} trial, target {a.trials}, anggaran {a.budget} mnt")
    while len(study.trials) < a.trials and (time.time() - start) / 60 < a.budget:
        t0 = time.time()
        study.optimize(objective, n_trials=1, catch=(Exception,))
        tr = study.trials[-1]
        best = study.best_value if any(t.state.name == "COMPLETE" for t in study.trials) else float("nan")
        log(f"trial {tr.number} {tr.state.name} "
            f"{'' if tr.value is None else f'AUC {tr.value:.5f} '}({(time.time() - t0) / 60:.1f} mnt) | terbaik {best:.5f}")
    log(f"=== SELESAI, terbaik {study.best_value:.5f} params {study.best_trial.params}")


if __name__ == "__main__":
    main()
