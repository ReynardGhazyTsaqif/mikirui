# %% [markdown]
# # E142 — Neural network tabular modern: RealMLP & TabM (DSC LOGIKA UI 2026)
#
# **Tujuan.** Semua model kita sejauh ini GBDT (LightGBM/XGBoost/CatBoost) yang prediksinya sangat mirip
# (Spearman 0,99), jadi ensemble-nya hanya menambah ±0,0003 AUC. Neural network belajar dengan cara berbeda
# (batas keputusan halus, bukan potongan pohon), sehingga kesalahannya lebih beragam dan berguna untuk blend.
#
# - **RealMLP** (Holzmüller dkk., NeurIPS 2024) dan **TabM** (Gorishniy dkk., ICLR 2025), lewat pustaka
#   `pytabkit`. Keduanya **dilatih dari nol** pada data lomba; tidak ada bobot pretrained atau data luar.
# - Fitur sama dengan acuan E130a: 528 kolom + 7 fitur `ecm_*` nested per fold. Fold sama dengan CV lokal.
#
# **Keluaran** (`/kaggle/working`): `e142_<model>_oof_s<seed>.npy` (urutan baris train) dan
# `e142_<model>_test_s<seed>.npy` (urutan baris test), plus `e142_ids.npz`.
#
# **Cara jalan di Kaggle.** Add Input → dataset privat `dsc-logika-exp`; Settings → Accelerator → GPU,
# Internet → On (untuk `pip install pytabkit`); lalu Save Version → Save & Run All. Notebook harus tetap privat.

# %%
import subprocess
import sys

try:
    import pytabkit  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit==1.7.3"], check=True)

# %%
import gc
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

SMOKE = os.environ.get("E142_SMOKE") == "1"  # uji kecil di CPU lokal
MODELS = ["realmlp", "tabm"]  # notebook berat: satu model per notebook, 5 seed
N_SEEDS = 3
# Fitur nested tambahan (format e121) di atas ecm_*; TAG ikut nama file keluaran. Varian "_seq" = + seq_gru/bigru/tf (E122c)
EXTRA = []
TAG = ""
# True: early stopping / pemilihan epoch di 10% holdout internal data latih fold (OOF jujur). False: pakai fold
# validasi luar (versi 14-15 Okt; OOF sedikit optimis karena epoch dipilih pada data yang juga dinilai).
INNER_VAL = True
if os.environ.get("E142_MODELS"):
    MODELS = os.environ["E142_MODELS"].split(",")
SEEDS = [42] if SMOKE else list(range(42, 42 + N_SEEDS))
N_FOLDS = 5
BUDGET_H = 10.0  # sesi Kaggle maks 12 jam: seed yang mulai setelah 10 jam dilewati, hasil sebelumnya tetap tersimpan

hits = glob.glob("/kaggle/input/**/apps_meta.parquet", recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ.get("E142_DATA", "kaggle/dataset")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("E142_OUT", ".")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("data:", DATA, "| output:", OUT, "| device:", DEVICE, "| models:", MODELS)

# %% [markdown]
# ## 1. Data
# Matriks fitur sama persis dengan pipeline lokal (urutan baris = `apps_meta`). Fitur `ecm_*` berbeda per
# fold (nested), jadi digabung di dalam loop fold.

# %%
meta = pd.read_parquet(f"{DATA}/apps_meta.parquet")
X = pd.read_parquet(f"{DATA}/X_all.parquet")
assert (X.pop("application_id").values == meta.application_id.values).all()
zs = [np.load(f"{DATA}/{f}", allow_pickle=True) for f in ["e121_ordermodel_v1.npz"] + EXTRA]
ecm_cols = [str(c) for z in zs for c in z["cols"]]
print("fitur nested:", ecm_cols)
is_tr = meta.is_test.values == 0
tr_idx, te_idx = np.where(is_tr)[0], np.where(~is_tr)[0]
y = meta.target.values[tr_idx].astype(int)
fold = meta.fold.values[tr_idx]
Xtr_all, Xte_all = X.values[tr_idx].astype("float32"), X.values[te_idx].astype("float32")
cols = list(X.columns)
del X
gc.collect()
if SMOKE:
    sub = np.random.default_rng(0).random(len(tr_idx)) < 0.08
    print("SMOKE: subset train", sub.sum())
print("train", Xtr_all.shape, "test", Xte_all.shape)

# %% [markdown]
# ## 2. Pra-proses (di-fit hanya pada data latih fold)
# Neural network tidak bisa menerima NaN. Untuk tiap pola missing yang berbeda dibuat satu kolom indikator
# (banyak kolom berbagi pola yang sama, misalnya semua fitur riwayat kosong untuk customer tanpa riwayat), lalu
# NaN diisi median data latih. Setelah itu QuantileTransformer (ke distribusi normal) meredam ekor panjang
# nominal rupiah.

# %%
def missing_patterns(A):
    """Indeks kolom wakil untuk tiap pola NaN unik (porsi NaN 0,5%–99,5%)."""
    nan = np.isnan(A)
    rate = nan.mean(0)
    seen, reps = {}, []
    for j in np.where((rate > 0.005) & (rate < 0.995))[0]:
        key = hash(nan[:, j].tobytes())
        if key not in seen:
            seen[key] = j
            reps.append(j)
    return reps


def prep(A_tr, A_others, seed):
    reps = missing_patterns(A_tr)
    med = np.nanmedian(A_tr, axis=0)
    med = np.where(np.isnan(med), 0, med)

    def tf(A):
        ind = np.isnan(A[:, reps]).astype("float32")
        B = np.where(np.isnan(A), med, A)
        return B, ind

    B_tr, I_tr = tf(A_tr)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=seed)
    B_tr = qt.fit_transform(B_tr)
    outs = []
    for A in A_others:
        B, I = tf(A)
        outs.append(np.hstack([qt.transform(B), I]).astype("float32"))
    return np.hstack([B_tr, I_tr]).astype("float32"), outs

# %% [markdown]
# ## 3. Model
# Pemilihan epoch terbaik / early stopping memakai 10% holdout internal dari data latih fold (`INNER_VAL`), jadi
# fold validasi benar-benar tidak terlihat dan skor OOF jujur (sama seperti model urutan GRU).

# %%
from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier  # noqa: E402


def make(name, seed):
    common = dict(device=DEVICE, random_state=seed, n_cv=1, n_refit=0, verbosity=0,
                  val_metric_name="1-auc_ovr")
    if name == "realmlp":
        # saran dokumentasi pytabkit untuk metrik AUC: tanpa label smoothing; batch lebih besar untuk >50k baris
        return RealMLP_TD_Classifier(**common, use_ls=False, batch_size=512, **({"n_epochs": 2} if SMOKE else {}))
    if name == "tabm":
        return TabM_D_Classifier(**common, **({"n_epochs": 2} if SMOKE else {}))
    raise ValueError(name)

# %%
t0 = time.time()
for name in MODELS:
    for s in SEEDS:
        if (time.time() - t0) / 3600 > BUDGET_H:
            print(f"anggaran {BUDGET_H} jam habis, {name} seed {s} dilewati", flush=True)
            continue
        oof, test_pred = np.full(len(tr_idx), np.nan), np.zeros(len(te_idx))
        for k in range(N_FOLDS):
            trn, va = fold != k, fold == k
            if SMOKE:
                trn, va = trn & sub, va & sub
            A_tr = np.hstack([Xtr_all[trn]] + [z[f"tr{k}"][(trn)[fold != k]] for z in zs])
            A_va = np.hstack([Xtr_all[va]] + [z[f"va{k}"][(va)[fold == k]] for z in zs])
            A_te = np.hstack([Xte_all] + [z[f"te{k}"] for z in zs])
            B_tr, (B_va, B_te) = prep(A_tr, [A_va, A_te], s)
            y_tr = y[trn]
            clf = make(name, s + k)
            if INNER_VAL:  # holdout bertingkat 10% dari data latih fold, terpisah dari fold validasi
                rng = np.random.default_rng(s * 10 + k)
                es = np.zeros(len(y_tr), bool)
                for c in (0, 1):
                    idx = np.where(y_tr == c)[0]
                    es[rng.choice(idx, len(idx) // 10, replace=False)] = True
                clf.fit(B_tr[~es], y_tr[~es], B_tr[es], y_tr[es])
            else:
                clf.fit(B_tr, y_tr, B_va, y[va])
            oof[va] = clf.predict_proba(B_va)[:, 1]
            test_pred += clf.predict_proba(B_te)[:, 1] / N_FOLDS
            print(f"{name} seed {s} fold {k}: AUC {roc_auc_score(y[va], oof[va]):.5f} "
                  f"({B_tr.shape[1]} kolom, {(time.time() - t0) / 60:.1f} mnt)", flush=True)
            del clf, A_tr, A_va, A_te, B_tr, B_va, B_te
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        ok = ~np.isnan(oof)
        aucs = [roc_auc_score(y[ok & (fold == k)], oof[ok & (fold == k)]) for k in range(N_FOLDS)]
        print(f"== {name} seed {s}: mean-fold AUC {np.mean(aucs):.5f}", flush=True)
        np.save(f"{OUT}/e142_{name}{TAG}_oof_s{s}.npy", oof)
        np.save(f"{OUT}/e142_{name}{TAG}_test_s{s}.npy", test_pred)

np.savez(f"{OUT}/e142_ids.npz", ids_train=meta.application_id.values[tr_idx], ids_test=meta.application_id.values[te_idx])
print("selesai:", sorted(os.listdir(OUT)))
