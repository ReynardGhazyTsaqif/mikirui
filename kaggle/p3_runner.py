# %% [markdown]
# # PROMPT 3 runner — CV model GBDT berat di Kaggle (DSC LOGIKA UI 2026)
#
# Job yang lambat di laptop (DART 5000 ronde, CatBoost, XGBoost) dijalankan paralel di Kaggle CPU. Logika latih sama
# persis dengan pipeline lokal (`e121.py` / `e141.py` / `src/tuning.fit_fold` / `src/models.py`):
# - fitur: `X_all` (528 kolom, kolom kategori dikembalikan ke tipe `category` dengan level yang sama seperti lokal),
#   + parquet tambahan (`rp_events_v1`, `ab_nonin_v1`), + fitur nested per fold (tr{k} latih, va{k} validasi, te{k} test);
# - parameter final diambil dari `p3_params.json` (hasil Optuna Fase 5, diekspor dari lokal);
# - fold tim (`apps_meta.fold`), atau fold baru `StratifiedKFold(5, shuffle=True, random_state=2026)` untuk FEATSET
#   "newfold" (hanya fitur non-nested, karena fitur nested terikat fold tim).
# Model dilatih dari nol pada data lomba; tidak ada data luar.
#
# **Keluaran** (`/kaggle/working`): `<NAME>_oof_s<seed>.npy` (urutan baris train), `<NAME>_test_s<seed>.npy`
# (urutan baris test). Taruh di `kaggle/output/` lalu impor ke `oof/` dan `preds/` (nama `<NAME>_s<seed>.npy`).
#
# **Cara jalan.** Add Input → dataset privat `dsc-logika-exp` (versi dengan `p3_params.json`); Accelerator None (CPU);
# Internet On (pin versi pustaka = lokal); Save Version → Save & Run All. Notebook harus tetap privat.

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.7.0", "xgboost==3.1.2", "catboost==1.2.10"],
               check=False)

# %%
import gc
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NAME = "E212"
MODEL = "dart"  # "lgb" | "dart" | "xgb" | "cat"
FEATSET = "fstar"  # "fstar" | "fstar_cpu" | "newfold" (non-nested, fold seed 2026)
SEEDS = [42, 43, 44]
SUBFOLDS = 1  # 2 = 10-fold bersarang seperti e121.py --subfolds 2 (hanya MODEL "lgb")
SMOKE = os.environ.get("P3_SMOKE") == "1"  # uji lokal: 1 seed, fold 0 saja
if os.environ.get("P3_MODEL"):
    MODEL, NAME = os.environ["P3_MODEL"], "SMOKE_" + os.environ["P3_MODEL"]
NESTED_ALL = ["e121_ordermodel_v1.npz", "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",
              "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz"]
EXTRA = ["rp_events_v1.parquet", "ab_nonin_v1.parquet"]
hits = glob.glob("/kaggle/input/**/apps_meta.parquet", recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ.get("P3_DATA", "kaggle/dataset")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("P3_OUT", ".")
print(NAME, MODEL, FEATSET, SEEDS, "| lightgbm", lgb.__version__, "| data", DATA)

# %% [markdown]
# ## 1. Data dan parameter

# %%
P = json.load(open(f"{DATA}/p3_params.json"))
meta = pd.read_parquet(f"{DATA}/apps_meta.parquet")
X = pd.read_parquet(f"{DATA}/X_all.parquet")
assert (X.pop("application_id").values == meta.application_id.values).all()
for c, levels in P["categorical"].items():  # kode angka -> category dengan level yang sama seperti lokal
    codes = X[c].fillna(-1).astype(int).values
    X[c] = pd.Categorical.from_codes(codes, categories=levels)
for f in EXTRA:
    E = pd.read_parquet(f"{DATA}/{f}").reindex(meta.application_id.values).reset_index(drop=True)
    X = pd.concat([X, E], axis=1)
nested = {"fstar": NESTED_ALL, "fstar_cpu": NESTED_ALL[:1], "newfold": []}[FEATSET]
zs = [np.load(f"{DATA}/{f}", allow_pickle=True) for f in nested]
ncols = [f"{c}{i}" for i, z in enumerate(zs) for c in z["cols"]]
is_tr = meta.is_test.values == 0
Xtr, Xte = X[is_tr].reset_index(drop=True), X[~is_tr].reset_index(drop=True)
y = meta.target.values[is_tr].astype(int)
fold = meta.fold.values[is_tr]
if FEATSET == "newfold":
    fold = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(len(y)), y)):
        fold[va] = k
del X
gc.collect()
print("train", Xtr.shape, "test", Xte.shape, "nested", len(ncols))


def block(key, n_rows):
    return pd.DataFrame(np.hstack([z[key] for z in zs]), columns=ncols) if zs else pd.DataFrame(index=range(n_rows))


def fit_predict(A, ya, B, yb, C, seed):
    """Satu model: kembalikan (pred_B, pred_C, iterasi). Sama dengan fit_fold / e141.py lokal."""
    if MODEL == "lgb":
        p = {**P["lgb"], "seed": seed}
        dtr = lgb.Dataset(A, ya, params={"max_bin": p.get("max_bin", 255)})
        m = lgb.train(p, dtr, 15000, valid_sets=[lgb.Dataset(B, yb, reference=dtr)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict(B, num_iteration=m.best_iteration), m.predict(C, num_iteration=m.best_iteration), m.best_iteration
    if MODEL == "dart":
        p = {**P["dart"], "seed": seed, "drop_seed": seed}
        dtr = lgb.Dataset(A, ya, params={"max_bin": p.get("max_bin", 255)})
        m = lgb.train(p, dtr, 5000, valid_sets=[lgb.Dataset(B, yb, reference=dtr)])
        return m.predict(B), m.predict(C), 5000
    if MODEL == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(**{**P["xgb"], "random_state": seed})
        m.fit(A, ya, eval_set=[(B, yb)], verbose=False)
        rng = (0, m.best_iteration + 1)
        return m.predict_proba(B, iteration_range=rng)[:, 1], m.predict_proba(C, iteration_range=rng)[:, 1], m.best_iteration
    import catboost as cb
    cats = [c for c in A.columns if str(A[c].dtype) == "category"]

    def prep(D):
        D = D.copy()
        for c in cats:
            D[c] = D[c].astype(str)
        return D
    m = cb.CatBoostClassifier(**{**P["cat"], "random_seed": seed}, cat_features=cats)
    m.fit(prep(A), ya, eval_set=(prep(B), yb), use_best_model=True)
    return m.predict_proba(prep(B))[:, 1], m.predict_proba(prep(C))[:, 1], m.get_best_iteration()


# %% [markdown]
# ## 2. CV 5 fold × seed

# %%
t0 = time.time()
for s in (SEEDS[:1] if SMOKE else SEEDS):
    oof, pred, its = np.zeros(len(y)), np.zeros(len(Xte)), []
    for k in ([0] if SMOKE else range(5)):
        tr, va = fold != k, fold == k
        A = pd.concat([Xtr[tr].reset_index(drop=True), block(f"tr{k}", tr.sum())], axis=1)
        B = pd.concat([Xtr[va].reset_index(drop=True), block(f"va{k}", va.sum())], axis=1)
        C = pd.concat([Xte, block(f"te{k}", len(Xte))], axis=1)
        if SUBFOLDS <= 1:
            pv, pt, it = fit_predict(A, y[tr], B, y[va], C, s)
            oof[va], pred = pv, pred + pt / 5
            its.append(it)
        else:
            yk, idx_k = y[va], np.flatnonzero(va)
            part = np.zeros(len(yk), dtype=int)
            for h, (_, ix) in enumerate(StratifiedKFold(SUBFOLDS, shuffle=True, random_state=42).split(np.zeros(len(yk)), yk)):
                part[ix] = h
            for h in range(SUBFOLDS):
                vm = part == h
                A2 = pd.concat([A, B[~vm]], ignore_index=True)
                pv, pt, it = fit_predict(A2, np.concatenate([y[tr], yk[~vm]]), B[vm].reset_index(drop=True), yk[vm], C, s)
                oof[idx_k[vm]] = pv
                pred += pt / (5 * SUBFOLDS)
                its.append(it)
        print(f"seed {s} fold {k}: AUC {roc_auc_score(y[va], oof[va]):.5f} ({(time.time() - t0) / 60:.0f} mnt)", flush=True)
        del A, B, C
        gc.collect()
    if SMOKE:
        break
    np.save(f"{OUT}/{NAME}_oof_s{s}.npy", oof)
    np.save(f"{OUT}/{NAME}_test_s{s}.npy", pred)
    aucs = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(5)]
    print(f"{NAME} seed {s}: mean-fold {np.mean(aucs):.5f} iter {its}", flush=True)
