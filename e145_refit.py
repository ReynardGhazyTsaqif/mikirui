"""E145: refit LGB tuned pada 100% data train dengan jumlah iterasi tetap (tanpa fold validasi).

Prediksi test di CV = rata-rata 5 model yang masing-masing hanya melihat 80% data. Refit memakai semua 70.000 baris,
jumlah iterasi = rata-rata best_iter CV × FACTOR (data 25% lebih banyak → butuh sedikit lebih banyak iterasi).
Fitur nested per fold (format e121): baris train fold k memakai va{k} (model luar fold k, dilatih 80% data),
test memakai rata-rata te{k} atas 5 fold — keduanya berasal dari model yang dilatih 80% data, jadi skalanya sama.
Tidak ada skor CV untuk refit (semua data dipakai); sanity check = korelasi dengan prediksi test versi CV.

  python e145_refit.py --use <npz+npz> --basefs all --iters 2330 --seeds 42 43 44 --name E145_E122x1
"""
import argparse
import gc
import sys
from argparse import Namespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.config import FEATURES_DIR, ROOT
from src.io_utils import load_apps
from src.train import write_submission


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--use", required=True)
    ap.add_argument("--basefs", default="all")
    ap.add_argument("--iters", type=float, required=True, help="rata-rata best_iter di CV")
    ap.add_argument("--factor", type=float, default=1.15)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--name", required=True)
    ap.add_argument("--cv_name", default=None, help="eksperimen CV pembanding (preds/<cv_name>_s*.npy)")
    ap.add_argument("--extra", default="", help="parquet tambahan di features/ (seperti e121.py), digabung '+'")
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
    assert a.basefs == "all"
    if a.extra:  # urutan kolom sama dengan e121.py: basis, parquet tambahan, lalu fitur nested
        E = pd.concat([pd.read_parquet(FEATURES_DIR / f).reindex(apps["application_id"])
                       for f in a.extra.split("+")], axis=1)
        X = pd.concat([X, E], axis=1)
        del E
    ctx.X_tr, ctx.X_te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
    del X, apps
    gc.collect()
    tuned, _ = best_tuned(ctx, "lgb", 15)
    params = full_params("lgb", tuned, FINAL_LR["lgb"])
    zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in a.use.split("+")]
    cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
    z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
    nested_tr = np.zeros((len(ctx.y), len(cols)), dtype="float32")
    for k in range(5):
        nested_tr[ctx.fold == k] = z[f"va{k}"]
    nested_te = np.mean([z[f"te{k}"] for k in range(5)], axis=0)
    X_tr = pd.concat([ctx.X_tr, pd.DataFrame(nested_tr, columns=cols)], axis=1)
    X_te = pd.concat([ctx.X_te, pd.DataFrame(nested_te, columns=cols)], axis=1)
    n_it = int(round(a.iters * a.factor))
    print(f"refit {a.name}: {X_tr.shape[1]} kolom, {len(X_tr)} baris, {n_it} iterasi, seed {a.seeds}", flush=True)
    pred = np.zeros(len(X_te))
    for s in a.seeds:
        mdl = lgb.train({**params, "seed": s}, lgb.Dataset(X_tr, ctx.y, params={"max_bin": params.get("max_bin", 255)}), n_it)
        p = mdl.predict(X_te)
        np.save(ROOT / "preds" / f"{a.name}_s{s}.npy", p)
        pred += rankdata(p) / len(p) / len(a.seeds)
        print(f"  seed {s} selesai", flush=True)
    if a.cv_name:
        import glob
        cvp = np.mean([rankdata(np.load(f)) for f in sorted(glob.glob(str(ROOT / "preds" / f"{a.cv_name}_s*.npy")))], 0)
        print(f"Spearman test refit vs CV ({a.cv_name}): {spearmanr(pred, cvp)[0]:.4f}")
    path, md5 = write_submission(ctx.test_ids, pred, f"{a.name}_refit{len(a.seeds)}")
    print(f"submission {path.name} md5 {md5}")


if __name__ == "__main__":
    main()
