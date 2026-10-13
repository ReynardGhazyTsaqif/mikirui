"""CV, logging eksperimen, dan pembuatan submission."""
import hashlib
import json
import subprocess
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from .config import (FOLDS_PATH, LOG_PATH, N_FOLDS, OOF_DIR, PREDS_DIR, SEED,
                     SUBMISSIONS_DIR, DATA_DIR)

LOG_COLS = ["exp_id", "tanggal", "pembuat", "deskripsi", "feature_set", "n_fitur", "model",
            "params_hash", *[f"auc_f{i}" for i in range(1, N_FOLDS + 1)], "oof_auc", "std_fold",
            "delta_vs_best", "oof_ks", "mean_fold_ks", "diterima", "runtime_menit", "git_commit", "submission_file", "md5",
            "public_lb"]


def ks_score(y, p):
    """Metrik lomba: KS dua sampel = max(TPR - FPR). Identik dengan scipy ks_2samp milik panitia."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def fold_ks(y, oof, fold):
    return [ks_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)]


def make_folds(train):
    """Buat folds.csv sekali; panggilan berikutnya membaca file dan memastikan isinya identik."""
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(skf.split(train, train["target"])):
        fold[va] = k
    folds = pd.DataFrame({"application_id": train["application_id"].values, "fold": fold})
    if FOLDS_PATH.exists():
        saved = pd.read_csv(FOLDS_PATH)
        assert saved.equals(folds), "folds.csv berbeda dari fold yang dibuat ulang"
    else:
        FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(FOLDS_PATH, index=False)
    return folds


def run_lgb_cv(params, X, y, fold, X_test, num_boost_round=5000, early_stopping=200, prep=None):
    """LightGBM 5-fold. Return oof, rata-rata prediksi test, AUC per fold, iterasi terbaik, importance.

    prep(X_tr, y_tr, X_va, X_te) -> (X_tr, X_va, X_te): transformasi yang di-fit di dalam fold
    (mis. target encoding), agar tidak ada kebocoran dari fold validasi.
    """
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    aucs, iters = [], []
    imp = pd.Series(0.0, index=X.columns)
    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        X_tr, X_va, X_te = X[tr], X[va], X_test
        if prep is not None:
            X_tr, X_va, X_te = prep(X_tr, y[tr], X_va, X_te)
        dtr = lgb.Dataset(X_tr, y[tr])
        dva = lgb.Dataset(X_va, y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        oof[va] = m.predict(X_va, num_iteration=m.best_iteration)
        test_pred += m.predict(X_te, num_iteration=m.best_iteration) / N_FOLDS
        aucs.append(roc_auc_score(y[va], oof[va]))
        iters.append(m.best_iteration)
        imp += pd.Series(m.feature_importance("gain"), index=X.columns) / N_FOLDS
    return oof, test_pred, aucs, iters, imp.sort_values(ascending=False)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def best_logged_auc():
    if not LOG_PATH.exists():
        return None
    log = pd.read_csv(LOG_PATH)
    log = log[log["diterima"].astype(str) == "True"]  # kolom bisa berisi 'kandidat' (teks)
    return log["oof_auc"].max() if len(log) else None


def log_experiment(exp_id, deskripsi, feature_set, n_fitur, model, params, aucs, oof_auc,
                   runtime_min, diterima, oof_ks, ks_folds, pembuat="", submission_file="",
                   md5_hash=""):
    best = best_logged_auc()
    row = dict(
        exp_id=exp_id, tanggal=datetime.now().strftime("%Y-%m-%d %H:%M"), pembuat=pembuat,
        deskripsi=deskripsi, feature_set=feature_set, n_fitur=n_fitur, model=model,
        params_hash=hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8],
        **{f"auc_f{i + 1}": round(a, 6) for i, a in enumerate(aucs)},
        oof_auc=round(oof_auc, 6), std_fold=round(float(np.std(aucs)), 6),
        delta_vs_best="" if best is None else round(oof_auc - best, 6),
        oof_ks=round(oof_ks, 6), mean_fold_ks=round(float(np.mean(ks_folds)), 6), diterima=diterima,
        runtime_menit=round(runtime_min, 2), git_commit=_git_commit(),
        submission_file=submission_file, md5=md5_hash, public_lb="",
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=LOG_COLS)
    df.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
    return row


def save_arrays(exp_id, oof, test_pred):
    OOF_DIR.mkdir(exist_ok=True)
    PREDS_DIR.mkdir(exist_ok=True)
    np.save(OOF_DIR / f"{exp_id}.npy", oof)
    np.save(PREDS_DIR / f"{exp_id}.npy", test_pred)


def write_submission(test_ids, test_pred, name):
    """Merge ke sample_submission via application_id, validasi, simpan, return (path, md5)."""
    ss = pd.read_csv(DATA_DIR / "sample_submission.csv")[["application_id"]]
    pred = pd.DataFrame({"application_id": np.asarray(test_ids),
                         "default_probability": np.round(np.asarray(test_pred, dtype=float), 6)})
    sub = ss.merge(pred, on="application_id", how="left")
    assert len(sub) == 30000 and sub["application_id"].is_unique
    assert sub["default_probability"].notna().all()
    assert sub["default_probability"].between(0, 1).all()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / f"{name}.csv"
    sub.to_csv(path, index=False, lineterminator="\n")  # LF: MD5 sama di Windows dan Kaggle
    return path, md5(path)


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.minutes = (time.time() - self.t) / 60
