"""PROMPT 3 Langkah 6: cek ulang keputusan level model ("DART vs GBDT", bobot blend) dengan fold BARU
StratifiedKFold(5, shuffle=True, random_state=2026).

Hanya fitur non-nested (528 kolom + rp_ev + ab_nonin): fitur nested (ecm, seq, seqx) dibangun per fold tim; memakainya
dengan fold baru akan bocor (baris validasi fold baru pernah menjadi data latih model nested yang menghasilkan
fiturnya), jadi tidak dipakai tanpa dibangun ulang.

  python experiments/prompt3/p3_newfold.py --model gbdt --seeds 42 43 44
  python experiments/prompt3/p3_newfold.py --model dart --seeds 42 43 44   # 5000 ronde, setting E141
Keluaran: oof/NF_<model>_s<seed>.npy (urutan train), experiments/prompt3/newfold_folds.npy
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import FEATURES_DIR  # noqa: E402
from src.io_utils import load_apps  # noqa: E402
from src.train import ks_score  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gbdt", "dart"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--extra", default="rp_events_v1.parquet+ab_nonin_v1.parquet")
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
    E = pd.concat([pd.read_parquet(FEATURES_DIR / f).reindex(apps["application_id"]) for f in a.extra.split("+")], axis=1)
    X_tr = pd.concat([X, E], axis=1)[m].reset_index(drop=True)
    del X, E, apps
    gc.collect()
    y = ctx.y
    nf = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(len(y)), y)):
        nf[va] = k
    np.save(ROOT / "experiments" / "prompt3" / "newfold_folds.npy", nf)
    print(f"kesamaan fold baru vs fold tim (porsi baris dengan nomor fold sama): {(nf == ctx.fold).mean():.3f}", flush=True)
    tuned, _ = best_tuned(ctx, "lgb", 15)
    if a.model == "gbdt":
        params = full_params("lgb", tuned, FINAL_LR["lgb"])
    else:
        params = {**full_params("lgb", tuned, 0.05), "boosting": "dart", "drop_rate": 0.1, "skip_drop": 0.5,
                  "max_drop": 50, "metric": "auc"}
    mb = {"max_bin": params.get("max_bin", 255)}
    for s in a.seeds:
        path = ROOT / "oof" / f"NF_{a.model}_s{s}.npy"
        if path.exists():
            continue
        oof, aucs = np.zeros(len(y)), []
        for k in range(5):
            tr, va = nf != k, nf == k
            dtr = lgb.Dataset(X_tr[tr], y[tr], params=mb)
            if a.model == "gbdt":
                mdl = lgb.train({**params, "seed": s}, dtr, 15000,
                                valid_sets=[lgb.Dataset(X_tr[va], y[va], reference=dtr)],
                                callbacks=[lgb.early_stopping(200, verbose=False)])
                oof[va] = mdl.predict(X_tr[va], num_iteration=mdl.best_iteration)
            else:
                mdl = lgb.train({**params, "seed": s, "drop_seed": s}, dtr, 5000)
                oof[va] = mdl.predict(X_tr[va])
            aucs.append(roc_auc_score(y[va], oof[va]))
            print(f"  {a.model} seed {s} fold {k}: AUC {aucs[-1]:.5f}", flush=True)
            del dtr, mdl
            gc.collect()
        np.save(path, oof)
        print(f"NF_{a.model} seed {s}: mean-fold {np.mean(aucs):.5f} KS {ks_score(y, oof):.4f}", flush=True)


if __name__ == "__main__":
    main()
