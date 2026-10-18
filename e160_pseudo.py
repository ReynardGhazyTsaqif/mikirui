"""E160 (PROMPT 3 Langkah 8): pseudo-label test, nested per fold.

Untuk fold luar k: (1) model dasar LGB tuned dilatih di fold != k (fitur nested tr{k}), early stopping di fold k,
memprediksi test dengan fitur te{k} -> label lunak p (HANYA dari model fold k, yang tidak melihat label fold k);
(2) model kedua dilatih di fold != k + baris test: mode "soft" = tiap baris test diduplikasi sebagai label 1 berbobot p
dan label 0 berbobot 1-p; mode "conf" = hanya baris yakin (p < 0,03 -> 0, p > 0,6 -> 1), bobot 1; (3) model kedua
dinilai di fold k. Prediksi test = rata-rata model kedua atas 5 fold. Bandingkan dengan acuan set sama (E204) lewat
compare_oof.

  python e160_pseudo.py --use <npz+...> --extra rp_events_v1.parquet+ab_nonin_v1.parquet --mode soft --name E216
"""
import argparse
import gc
import sys
from argparse import Namespace
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FEATURES_DIR, N_FOLDS, ROOT
from src.io_utils import load_apps
from src.train import ks_score

LOG = ROOT / "experiments" / "e160.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--use", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--mode", choices=["soft", "conf"], default="soft")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--name", required=True)
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
    X_tr, X_te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
    del X, apps
    gc.collect()
    tuned, _ = best_tuned(ctx, "lgb", 15)
    params = full_params("lgb", tuned, FINAL_LR["lgb"])
    zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in a.use.split("+")]
    cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
    z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
    y, fold = ctx.y, ctx.fold
    mb = {"max_bin": params.get("max_bin", 255)}
    log(f"=== {a.name} pseudo-label mode {a.mode}, seed {a.seeds}, {X_tr.shape[1] + len(cols)} kolom")
    for s in a.seeds:
        path = ROOT / "oof" / f"{a.name}_s{s}.npy"
        if path.exists():
            continue
        p_s = {**params, "seed": s}
        oof, pred, base_auc = np.zeros(len(y)), np.zeros(len(X_te)), []
        for k in range(N_FOLDS):
            tr, va = fold != k, fold == k
            A = pd.concat([X_tr[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
            B = pd.concat([X_tr[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
            C = pd.concat([X_te, pd.DataFrame(z[f"te{k}"], columns=cols)], axis=1)
            dtr = lgb.Dataset(A, y[tr], params=mb)
            m1 = lgb.train(p_s, dtr, 15000, valid_sets=[lgb.Dataset(B, y[va], reference=dtr)],
                           callbacks=[lgb.early_stopping(200, verbose=False)])
            base_auc.append(roc_auc_score(y[va], m1.predict(B, num_iteration=m1.best_iteration)))
            pt = m1.predict(C, num_iteration=m1.best_iteration)
            if a.mode == "soft":
                A2 = pd.concat([A, C, C], ignore_index=True)
                y2 = np.concatenate([y[tr], np.ones(len(C)), np.zeros(len(C))])
                w2 = np.concatenate([np.ones(tr.sum()), pt, 1 - pt])
            else:
                sel = (pt < 0.03) | (pt > 0.6)
                A2 = pd.concat([A, C[sel]], ignore_index=True)
                y2 = np.concatenate([y[tr], (pt[sel] > 0.5).astype(float)])
                w2 = np.ones(len(y2))
            d2 = lgb.Dataset(A2, y2, weight=w2, params=mb)
            m2 = lgb.train(p_s, d2, 15000, valid_sets=[lgb.Dataset(B, y[va], reference=d2)],
                           callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[va] = m2.predict(B, num_iteration=m2.best_iteration)
            pred += m2.predict(C, num_iteration=m2.best_iteration) / N_FOLDS
            log(f"  seed {s} fold {k}: dasar {base_auc[-1]:.5f} -> pseudo {roc_auc_score(y[va], oof[va]):.5f} "
                f"(iter {m1.best_iteration} -> {m2.best_iteration})")
            del A, B, C, A2, d2, m1, m2
            gc.collect()
        np.save(path, oof)
        np.save(ROOT / "preds" / f"{a.name}_s{s}.npy", pred)
        aucs = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)]
        log(f"{a.name} seed {s}: AUC mean-fold {np.mean(aucs):.5f} (dasar {np.mean(base_auc):.5f}) KS {ks_score(y, oof):.4f}")


if __name__ == "__main__":
    main()
