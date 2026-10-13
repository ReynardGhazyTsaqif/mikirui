"""E128: model khusus segmen (tanpa riwayat / punya riwayat) digabung dengan model global E130a.

Segmen "tanpa riwayat" = customer tanpa baris previous_applications (30%). Model segmen = LightGBM
parameter tuned Fase 5 pada set fitur E130a (528 kolom + 7 ecm_* nested), dilatih HANYA pada baris
segmen dari fold latih; kolom yang seluruhnya kosong di segmen dibuang. Fold sama dengan CV utama.

Gabungan: di dalam segmen, logit akhir = (1 - w) * logit(global E130a bag) + w * logit(model segmen);
di luar segmen tetap global. Evaluasi mean-fold AUC di seluruh data untuk beberapa w.

  python e128.py --seg nohist --lr 0.05 --seeds 42          # screen
  python e128.py --seg nohist hist --lr 0.02 --seeds 42 43 44
Keluaran: oof/E128_<seg><tag>_s<seed>.npy (NaN di luar segmen), preds/... (test), log experiments/e128.log
"""
import argparse
import gc
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

import run
from src.config import FEATURES_DIR, N_FOLDS, ROOT
from src.features.registry import GROUPS
from src.io_utils import load, load_apps
from src.train import Timer, ks_score, make_folds
from src.tuning import full_params

STORAGE = f"sqlite:///{(ROOT / 'experiments' / 'p5' / 'optuna.db').as_posix()}"
LOG = ROOT / "experiments" / "e128.log"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def bag(name, kind):
    paths = [p for p in sorted((ROOT / kind).glob(f"{name}_s*.npy")) if p.stem[len(name) + 2:].isdigit()]
    return np.mean([np.load(p) for p in paths], axis=0)


def combine(g, s, seg, w):
    out = g.copy()
    out[seg] = expit((1 - w) * logit(np.clip(g[seg], 1e-7, 1 - 1e-7)) + w * logit(np.clip(s[seg], 1e-7, 1 - 1e-7)))
    return out


def report(y, fold, g, segs):
    """segs: list (mask, oof segmen). Cetak AUC/KS gabungan untuk tiap w (sama untuk semua segmen)."""
    base = [roc_auc_score(y[fold == k], g[fold == k]) for k in range(N_FOLDS)]
    log(f"global E130a: mean-fold {np.mean(base):.5f} KS {ks_score(y, g):.4f}")
    for mask, s in segs:
        log(f"  segmen n={mask.sum()}: AUC global {roc_auc_score(y[mask], g[mask]):.5f} "
            f"segmen {roc_auc_score(y[mask], s[mask]):.5f}")
    for w in WEIGHTS[1:]:
        p = g
        for mask, s in segs:
            p = combine(p, s, mask, w)
        au = [roc_auc_score(y[fold == k], p[fold == k]) for k in range(N_FOLDS)]
        d = np.array(au) - np.array(base)
        log(f"  w={w:.2f}: mean-fold {np.mean(au):.5f} (delta {np.mean(d):+.5f}, naik {int((d > 0).sum())}/5) "
            f"KS {ks_score(y, p):.4f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="+", default=["nohist"], choices=["nohist", "hist"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=15000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--tag", default="", help="akhiran nama, mis. _scr")
    a = ap.parse_args()

    apps = load_apps()
    m = apps["is_test"].eq(0).values
    pa_cust = set(load("previous_applications", usecols=["customer_id"])["customer_id"])
    hist_all = apps["customer_id"].isin(pa_cust).values
    X = run.build_matrix(list(GROUPS), apps)
    X_tr, X_te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
    y = apps.loc[m, "target"].astype(int).values
    fold = make_folds(apps[m])["fold"].values
    hist_tr, hist_te = hist_all[m], hist_all[~m]
    del X, apps
    gc.collect()
    z = np.load(FEATURES_DIR / "e121_ordermodel_v1.npz", allow_pickle=True)
    cols = [f"{c}0" for c in z["cols"]]
    tuned = optuna.load_study(study_name="tune_lgb", storage=STORAGE).best_trial.params
    params = full_params("lgb", tuned, a.lr)
    log(f"=== E128 {vars(a)} | punya riwayat train {hist_tr.mean():.3f} test {hist_te.mean():.3f}")

    seg_masks = {"nohist": (~hist_tr, ~hist_te), "hist": (hist_tr, hist_te)}
    for seg in a.seg:
        sm_tr, sm_te = seg_masks[seg]
        name = f"E128_{seg}{a.tag}"
        for s in a.seeds:
            path = ROOT / "oof" / f"{name}_s{s}.npy"
            if path.exists():
                log(f"{name} seed {s}: sudah ada, dilewati")
                continue
            oof, pred, iters = np.full(len(y), np.nan), np.zeros(len(X_te)), []
            with Timer() as t:
                for k in range(N_FOLDS):
                    tr, va = (fold != k) & sm_tr, (fold == k) & sm_tr
                    f_tr = pd.DataFrame(z[f"tr{k}"], columns=cols)[sm_tr[fold != k]].reset_index(drop=True)
                    f_va = pd.DataFrame(z[f"va{k}"], columns=cols)[sm_tr[fold == k]].reset_index(drop=True)
                    f_te = pd.DataFrame(z[f"te{k}"], columns=cols)[sm_te].reset_index(drop=True)
                    A = pd.concat([X_tr[tr].reset_index(drop=True), f_tr], axis=1)
                    keep = [c for c in A.columns if A[c].notna().any() and A[c].nunique() > 1]
                    A = A[keep]
                    B = pd.concat([X_tr[va].reset_index(drop=True), f_va], axis=1)[keep]
                    C = pd.concat([X_te[sm_te].reset_index(drop=True), f_te], axis=1)[keep]
                    dtr = lgb.Dataset(A, y[tr], params={"max_bin": params.get("max_bin", 255)})
                    dva = lgb.Dataset(B, y[va], reference=dtr)
                    mdl = lgb.train({**params, "seed": s}, dtr, a.rounds, valid_sets=[dva],
                                    callbacks=[lgb.early_stopping(200, verbose=False)])
                    oof[va] = mdl.predict(B, num_iteration=mdl.best_iteration)
                    pred[sm_te] += mdl.predict(C, num_iteration=mdl.best_iteration) / N_FOLDS
                    iters.append(mdl.best_iteration)
                    del A, B, C, dtr, dva, mdl
                    gc.collect()
            pred[~sm_te] = np.nan
            np.save(path, oof)
            np.save(ROOT / "preds" / f"{name}_s{s}.npy", pred)
            au = [roc_auc_score(y[(fold == k) & sm_tr], oof[(fold == k) & sm_tr]) for k in range(N_FOLDS)]
            log(f"{name} seed {s}: AUC dalam segmen mean-fold {np.mean(au):.5f} ({len(keep)} kolom) "
                f"iter {iters} ({t.minutes:.1f} mnt)")

    g = bag("E130a", "oof")
    segs = []
    for seg in a.seg:
        o = bag(f"E128_{seg}{a.tag}", "oof")
        segs.append((seg_masks[seg][0], o))
        log(f"--- gabungan segmen {seg}")
        report(y, fold, g, [segs[-1]])
    if len(segs) > 1:
        log("--- gabungan semua segmen")
        report(y, fold, g, segs)


if __name__ == "__main__":
    main()
