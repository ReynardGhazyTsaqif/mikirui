"""Verifikasi bertahap notebook final terhadap artefak pipeline eksperimen (TIDAK dipakai notebook itu sendiri).

Menjalankan sel kode notebook berurutan dalam satu namespace sampai fitur ecm selesai, lalu membandingkan PERSIS:
  1. tiap grup fitur di cache memori notebook vs features/<grup>.parquet (nilai, dtype, index, urutan kolom);
  2. matriks fitur 546 kolom vs rakitan dari parquet;
  3. fitur ecm nested vs features/e121_ordermodel_v1.npz;
  4. (--train-check) GBDT seed 42 fold 0 vs oof/E205_s42.npy di baris fold 0.

  python final/verify_notebook.py final/NomorPeserta_NamaTim_Notebook.ipynb --train-check
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--train-check", action="store_true")
    a = ap.parse_args()
    nb = json.load(open(a.notebook, encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    stop = next(i for i, c in enumerate(cells) if "ECM = build_ecm(" in c)
    ns = {"__name__": "__notebook__"}
    t0 = time.time()
    for i, c in enumerate(cells[: stop + 1]):
        exec(compile(c, f"<sel {i}>", "exec"), ns)
    print(f"sel 0..{stop} selesai ({(time.time() - t0) / 60:.1f} mnt)", flush=True)
    ok = True
    for name, df in ns["_CACHE"].items():
        ref = pd.read_parquet(ROOT / "features" / f"{name}.parquet")
        same = df.equals(ref) and list(df.columns) == list(ref.columns) and (df.dtypes == ref.dtypes).all() \
            and df.index.equals(ref.index)
        if not same:
            ok = False
            diff = [c for c in ref.columns if c not in df.columns or not df[c].equals(ref[c])]
            print(f"  BEDA {name}: kolom berbeda {diff[:8]}")
    print(f"grup fitur identik dengan cache: {ok} ({len(ns['_CACHE'])} grup)", flush=True)
    sys.path.insert(0, str(ROOT))
    import run
    from src.features.registry import GROUPS
    from src.io_utils import load_apps
    apps = load_apps()
    Xr = run.build_matrix(list(GROUPS), apps)
    Er = pd.read_parquet(ROOT / "features" / "rp_events_v1.parquet").reindex(apps["application_id"])
    Xr = pd.concat([Xr, Er], axis=1)
    X = ns["X"]
    print("matriks 546 kolom identik:", X.equals(Xr) and list(X.columns) == list(Xr.columns), X.shape, flush=True)
    z = np.load(ROOT / "features" / "e121_ordermodel_v1.npz", allow_pickle=True)
    E = ns["ECM"]
    same = list(E["cols"]) == list(z["cols"]) and all(
        np.array_equal(E[k], z[k], equal_nan=True) for k in z.files if k != "cols")
    print("fitur ecm nested identik dengan npz:", same, flush=True)
    if not same:
        for k in z.files:
            if k != "cols":
                d = np.nanmax(np.abs(E[k].astype("float64") - z[k].astype("float64")))
                print(f"  {k}: maks selisih {d:.3g}")
    if a.train_check:
        nxt = next(i for i, c in enumerate(cells) if "PARAMS_LGB =" in c)
        exec(compile(cells[nxt], "<sel params>", "exec"), ns)
        exec(compile("""
tr, va, A, B, C = fold_mats(0)
dtr = lgb.Dataset(A, y[tr], params={"max_bin": PARAMS_LGB.get("max_bin", 255)})
dva = lgb.Dataset(B, y[va], reference=dtr)
_m = lgb.train({**PARAMS_LGB, "seed": 42}, dtr, 15000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
_p = _m.predict(B, num_iteration=_m.best_iteration)
""", "<cek latih>", "exec"), ns)
        ref = np.load(ROOT / "oof" / "E205_s42.npy")[ns["fold"] == 0]
        print(f"GBDT seed 42 fold 0: iterasi {ns['_m'].best_iteration}, prediksi identik dengan OOF E205: "
              f"{np.array_equal(ns['_p'], ref)} (maks selisih {np.abs(ns['_p'] - ref).max():.3g})", flush=True)
    print(f"selesai ({(time.time() - t0) / 60:.1f} mnt)")


if __name__ == "__main__":
    main()
