# %% [markdown]
# # E215 — AutoGluon kualitas tertinggi sebagai pembanding (DSC LOGIKA UI 2026)
#
# **Tujuan** (masukan mentor, ide 14): (1) cek apakah rekayasa fitur kita membuang informasi, (2) batas atas kasar
# dari AutoML yang kuat, (3) varian kuat yang berbeda untuk portofolio submission.
#
# - AutoGluon Tabular `presets="best_quality"` (bagging + stacking LightGBM/CatBoost/XGBoost/NN/RF/ET), **dilatih
#   dari nol** pada data lomba; tidak ada bobot pretrained atau data luar.
# - **OOF selaras fold kita**: untuk fold luar k, AutoGluon dilatih pada fold != k dan memprediksi fold k serta test.
#   Fitur nested (ecm_*, seq_*, seqx_*) memakai versi fold k (tr{k} untuk latih, va{k} validasi, te{k} test),
#   persis seperti pipeline lokal, sehingga tidak bocor.
# - Varian `FEATSET`: `"fstar"` = F* (528 kolom + rp_ev + ab_nonin + 7 blok nested) atau `"naive"` = 239 kolom
#   agregat naif/mentah saja (tanpa nested).
# - AutoGluon tidak menjamin hasil identik antar-run → hanya untuk perbandingan/portofolio; masuk notebook final
#   hanya bila lolos uji MD5.
#
# **Keluaran** (`/kaggle/working`): `e215_<featset>_oof.npy` (urutan baris train), `e215_<featset>_test.npy`
# (rata-rata 5 fold, urutan baris test), `e215_<featset>_leaderboard_fold<k>.csv`, `e215_ids.npz`.
#
# **Cara jalan di Kaggle.** Add Input → dataset privat `dsc-logika-exp` (versi dengan `rp_events_v1.parquet`,
# `ab_nonin_v1.parquet`, `naive_cols.json`); Settings → Accelerator → None (CPU, RAM 30 GB) atau GPU,
# Internet → On; Save Version → Save & Run All. Perkiraan: `TIME_PER_FOLD` × 5 (default 2 jam × 5 = 10 jam).
# Notebook dan dataset harus tetap privat.

# %%
import subprocess
import sys

try:
    import autogluon.tabular  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "autogluon.tabular[all]==1.2"], check=True)

# %%
import gc
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score

FEATSET = os.environ.get("E215_FEATSET", "fstar")  # "fstar" | "naive"
TIME_PER_FOLD = int(os.environ.get("E215_TIME", 7200))  # detik per fold luar
SMOKE = os.environ.get("E215_SMOKE") == "1"
N_FOLDS = 5
NESTED = ["e121_ordermodel_v1.npz", "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",
          "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz"]
hits = glob.glob("/kaggle/input/**/apps_meta.parquet", recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ.get("E215_DATA", "kaggle/dataset")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("E215_OUT", ".")
print("data:", DATA, "| output:", OUT, "| featset:", FEATSET, "| detik/fold:", TIME_PER_FOLD)

# %% [markdown]
# ## 1. Data
# Urutan baris = `apps_meta` (train lalu test). Kolom kategori di `X_all` sudah berupa kode angka.

# %%
meta = pd.read_parquet(f"{DATA}/apps_meta.parquet")
X = pd.read_parquet(f"{DATA}/X_all.parquet")
assert (X.pop("application_id").values == meta.application_id.values).all()
if FEATSET == "naive":
    keep = json.load(open(f"{DATA}/naive_cols.json"))
    X = X[[c for c in X.columns if c in set(keep)]]
    zs, ncols = [], []
else:
    for f in ["rp_events_v1.parquet", "ab_nonin_v1.parquet"]:
        E = pd.read_parquet(f"{DATA}/{f}").reindex(meta.application_id.values)
        X = pd.concat([X, E.reset_index(drop=True)], axis=1)
    zs = [np.load(f"{DATA}/{f}", allow_pickle=True) for f in NESTED]
    ncols = [f"{c}{i}" for i, z in enumerate(zs) for c in z["cols"]]
is_tr = meta.is_test.values == 0
tr_idx, te_idx = np.where(is_tr)[0], np.where(~is_tr)[0]
y = meta.target.values[tr_idx].astype(int)
fold = meta.fold.values[tr_idx]
Xtr, Xte = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
del X
gc.collect()
print("train", Xtr.shape, "test", Xte.shape, "nested", len(ncols))


def nested(key):
    return pd.DataFrame(np.hstack([z[key] for z in zs]), columns=ncols) if zs else None


# %% [markdown]
# ## 2. AutoGluon per fold luar

# %%
oof = np.full(len(y), np.nan)
test = np.zeros(len(te_idx))
t0 = time.time()
for k in range(N_FOLDS):
    tr, va = fold != k, fold == k
    A = Xtr[tr].reset_index(drop=True)
    B = Xtr[va].reset_index(drop=True)
    C = Xte.copy()
    if zs:
        A = pd.concat([A, nested(f"tr{k}")], axis=1)
        B = pd.concat([B, nested(f"va{k}")], axis=1)
        C = pd.concat([C, nested(f"te{k}")], axis=1)
    A["target"] = y[tr]
    if SMOKE:
        A = A.sample(3000, random_state=42)
    pred = TabularPredictor(label="target", eval_metric="roc_auc", path=f"{OUT}/ag_{FEATSET}_fold{k}",
                            verbosity=1).fit(A, presets="best_quality", time_limit=60 if SMOKE else TIME_PER_FOLD)
    oof[va] = pred.predict_proba(B)[1].values
    test += pred.predict_proba(C)[1].values / N_FOLDS
    pred.leaderboard(silent=True).to_csv(f"{OUT}/e215_{FEATSET}_leaderboard_fold{k}.csv", index=False)
    print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.5f} ({(time.time() - t0) / 60:.0f} mnt)", flush=True)
    np.save(f"{OUT}/e215_{FEATSET}_oof.npy", oof)
    np.save(f"{OUT}/e215_{FEATSET}_test.npy", test * N_FOLDS / (k + 1))
    os.system(f"rm -rf {OUT}/ag_{FEATSET}_fold{k}")  # model besar; hasil prediksi sudah disimpan
    del A, B, C, pred
    gc.collect()
aucs = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)]
print(f"E215 {FEATSET}: mean-fold AUC {np.mean(aucs):.5f} {np.round(aucs, 4).tolist()}")
np.savez(f"{OUT}/e215_ids.npz", train=meta.application_id.values[tr_idx], test=meta.application_id.values[te_idx])
