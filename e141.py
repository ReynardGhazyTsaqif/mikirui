"""E141: LightGBM DART pada set E130a (528 kolom + 7 ecm_* nested), parameter pohon dari study tune_lgb.

DART tidak mendukung early stopping, jadi jumlah ronde tetap (--rounds). Kurva AUC validasi dicatat tiap
500 ronde (rata-rata 5 fold) untuk memilih ronde di run berikutnya; prediksi memakai semua ronde.

  python e141.py --rounds 3000 --lr 0.05 --seeds 42 --name E141_scr
Keluaran: oof/<name>_s<seed>.npy, preds/<name>_s<seed>.npy, log experiments/e141.log
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
from src.config import FEATURES_DIR, N_FOLDS, ROOT
from src.features.registry import GROUPS
from src.io_utils import load_apps
from src.train import Timer, ks_score, log_experiment, make_folds, write_submission
from src.tuning import full_params

STORAGE = f"sqlite:///{(ROOT / 'experiments' / 'p5' / 'optuna.db').as_posix()}"
LOG = ROOT / "experiments" / "e141.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--skip", type=float, default=0.5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--name", default="E141")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--use", default="e121_ordermodel_v1.npz", help="fitur nested format e121, digabung '+'")
    a = ap.parse_args()

    apps = load_apps()
    m = apps["is_test"].eq(0).values
    X = run.build_matrix(list(GROUPS), apps)
    X_tr, X_te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
    y = apps.loc[m, "target"].astype(int).values
    fold = make_folds(apps[m])["fold"].values
    test_ids = apps.loc[~m, "application_id"].values
    del X, apps
    gc.collect()
    zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in a.use.split("+")]
    cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]  # nama kolom sama dengan e121.py
    z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
    tuned = optuna.load_study(study_name="tune_lgb", storage=STORAGE).best_trial.params
    params = {**full_params("lgb", tuned, a.lr), "boosting": "dart", "drop_rate": a.drop,
              "skip_drop": a.skip, "max_drop": 50, "metric": "auc"}
    log(f"=== {a.name} {vars(a)} params {params}")
    for s in a.seeds:
        path = ROOT / "oof" / f"{a.name}_s{s}.npy"
        if path.exists():
            log(f"{a.name} seed {s}: sudah ada, dilewati")
            continue
        oof, pred, curves = np.zeros(len(y)), np.zeros(len(X_te)), []
        t0 = time.time()
        with Timer() as t:
            for k in range(N_FOLDS):
                tr, va = fold != k, fold == k
                A = pd.concat([X_tr[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
                B = pd.concat([X_tr[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
                C = pd.concat([X_te, pd.DataFrame(z[f"te{k}"], columns=cols)], axis=1)
                dtr = lgb.Dataset(A, y[tr], params={"max_bin": params.get("max_bin", 255)})
                dva = lgb.Dataset(B, y[va], reference=dtr)
                ev = {}
                mdl = lgb.train({**params, "seed": s, "drop_seed": s}, dtr, a.rounds, valid_sets=[dva],
                                callbacks=[lgb.record_evaluation(ev)])
                oof[va] = mdl.predict(B)
                pred += mdl.predict(C) / N_FOLDS
                curves.append(ev["valid_0"]["auc"])
                log(f"  seed {s} fold {k}: AUC {roc_auc_score(y[va], oof[va]):.5f} ({(time.time() - t0) / 60:.1f} mnt)")
                del A, B, C, dtr, dva, mdl
                gc.collect()
        np.save(path, oof)
        np.save(ROOT / "preds" / f"{a.name}_s{s}.npy", pred)
        cm = np.mean(curves, axis=0)
        pts = " ".join(f"{r}:{cm[r - 1]:.5f}" for r in range(500, a.rounds + 1, 500))
        aucs = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)]
        log(f"{a.name} seed {s}: AUC mean-fold {np.mean(aucs):.5f} KS {ks_score(y, oof):.4f} "
            f"| kurva {pts} | puncak ronde {int(np.argmax(cm)) + 1} ({t.minutes:.1f} mnt)")
    seeds = sorted(int(p.stem[len(a.name) + 2:]) for p in (ROOT / "oof").glob(f"{a.name}_s*.npy")
                   if p.stem[len(a.name) + 2:].isdigit())  # E141_s* jangan ikut E141_scr_s*
    oof = np.mean([np.load(ROOT / "oof" / f"{a.name}_s{s}.npy") for s in seeds], axis=0)
    pred = np.mean([np.load(ROOT / "preds" / f"{a.name}_s{s}.npy") for s in seeds], axis=0)
    aucs = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)]
    kss = [ks_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)]
    log(f"{a.name} BAG {seeds}: AUC mean-fold {np.mean(aucs):.5f} KS {ks_score(y, oof):.4f}")
    sub, md5 = "", ""
    if a.submit:
        p, md5 = write_submission(test_ids, pred, f"{a.name}_bag{len(seeds)}")
        sub = p.name
        log(f"submission {sub} md5 {md5}")
    log_experiment(a.name, f"LGB DART lr {a.lr} ronde {a.rounds} drop {a.drop} skip {a.skip}, set E130a, bag {seeds}",
                   "all+ecm", X_tr.shape[1] + len(cols), "lgb_dart_bag", params, aucs,
                   roc_auc_score(y, oof), 0, "kandidat", ks_score(y, oof), kss, submission_file=sub, md5_hash=md5)


if __name__ == "__main__":
    main()
