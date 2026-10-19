"""Rakit notebook final self-contained dari kode sumber yang SAMA dengan pipeline eksperimen.

Kode builder fitur (`src/features/*.py`), loader (`src/io_utils.py`), model level order e-commerce (`e121.py`),
pelatihan GBDT (`e121.py cv` + `src/tuning.fit_fold`), DART (`e141.py`), dan blend (`experiments/make_sub.py`)
disalin apa adanya ke sel notebook. Perubahan hanya: import relatif dibuang (semua nama ada di satu namespace),
cache parquet diganti cache di memori (nilai identik), konstanta WINDOWS diberi nama per modul (beda nilai per
modul), dan log ke file diganti print. Parameter hasil Optuna di-hardcode dari kaggle/dataset/p3_params.json.

  python final/build_notebook.py                   # -> final/NomorPeserta_NamaTim_Notebook.ipynb
  python final/build_notebook.py --out final/DSC0001_Tim_Notebook.ipynb
"""
import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [  # (berkas, ganti nama)
    ("src/features/app.py", {}),
    ("src/features/naive.py", {}),
    ("src/features/repayment.py", {"WINDOWS": "RP_WINDOWS"}),
    ("src/features/bureau.py", {}),
    ("src/features/previous.py", {"WINDOWS": "PA_WINDOWS"}),
    ("src/features/account.py", {}),
    ("src/features/ecommerce.py", {"WINDOWS": "EC_WINDOWS"}),
    ("src/features/cross.py", {}),
    ("src/features/events.py", {}),
]
TITLES = {
    "app": "5.1 Aplikasi sekarang dan profil (`app_`, `prof_`)",
    "naive": "5.2 Agregat naif semua tabel pendukung",
    "repayment": "5.3 Riwayat cicilan (`rp_`)",
    "bureau": "5.4 Biro kredit (`bu_`, `bp_`)",
    "previous": "5.5 Aplikasi sebelumnya (`pa_`)",
    "account": "5.6 Perilaku rekening (`ab_`)",
    "ecommerce": "5.7 E-commerce (`ec_`)",
    "cross": "5.8 Fitur lintas tabel (`x_`)",
    "events": "5.9 Rantai bayar bulanan level event (`rp_ev_`) — temuan utama eksplorasi",
}


def strip_module(path, renames):
    """Kembalikan (docstring, kode) modul: docstring modul dan semua import dibuang, nama diganti per modul."""
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno, node.end_lineno + 1))
    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
        drop.update(range(first.lineno, first.end_lineno + 1))
    lines = [ln for i, ln in enumerate(src.splitlines(), 1) if i not in drop]
    code = "\n".join(lines).strip("\n")
    code = re.sub(r"\n{3,}", "\n\n\n", code)
    for old, new in renames.items():
        code = re.sub(rf"\b{old}\b", new, code)
    return doc, code


def segment(path, name):
    src = (ROOT / path).read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(name)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build(out):
    P = json.loads((ROOT / "kaggle" / "dataset" / "p3_params.json").read_text())
    cells = []
    cells.append(md("""
# DSC LOGIKA UI 2026 — Prediksi Default Kredit (notebook final, kandidat A)

**Ringkasan.** Prediksi peluang default (DPD 90 dalam 12 bulan) untuk 30.000 aplikasi test, metrik KS.
Pipeline: 528 fitur rekayasa dari 8 tabel + fitur **rantai bayar bulanan level event** (temuan eksplorasi) + fitur
**model level order e-commerce** (nested per fold) → dua model LightGBM (GBDT tuned dan DART), masing-masing rata-rata
3 seed × 5 fold → blend rank 50:50.

| komponen | mean-fold AUC (OOF, 5-fold) |
|---|---|
| GBDT tuned (E205) | 0,84517 |
| DART 5000 ronde (E207) | 0,84536 |
| **blend rank 50:50 (submission)** | **0,84545** (KS OOF 0,5242; public 0,56428) |

**Reproduksibilitas.** Semua fitur dihitung dari data mentah di notebook ini (tanpa cache), seed tetap, LightGBM
deterministik dengan jumlah thread tetap, file ditulis dengan pembulatan 6 desimal dan akhir baris LF. Output harus
identik dengan submission terpilih: **MD5 `afc487f68e4603a38209f2d4a472312b`**. Semua model CPU (tanpa GPU).
Perkiraan waktu Run All: ±3–3,5 jam di CPU 4 thread (DART paling lama, ±2,3 jam). Ganti `DATA_DIR` bila perlu.
"""))
    cells.append(md("## 1. Setup\nVersi pustaka yang dipakai saat submission dibuat dicetak di bawah; pakai versi yang sama untuk hasil identik."))
    cells.append(code("""
import gc
import glob
import hashlib
import os
import platform
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
SEED = 42
N_FOLDS = 5
N_THREADS = 4
RUN_TUNING = False  # parameter hasil Optuna (Fase 5) di-hardcode di bagian 7; tuning tidak diulang

_hits = glob.glob("/kaggle/input/**/train_applications.csv", recursive=True)
DATA_DIR = Path(os.path.dirname(_hits[0])) if _hits else Path(os.environ.get("DSC_DATA_DIR", "Dataset"))
OUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".")
OUT_PATH = OUT_DIR / "submission.csv"
EXPECTED = {"python": "3.12", "lightgbm": "4.7.0", "numpy": "2.3.3", "pandas": "2.3.2", "scikit-learn": "1.8.0",
            "scipy": "1.16.3"}
print("python", platform.python_version(), "| lightgbm", lgb.__version__, "| numpy", np.__version__,
      "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| scipy", scipy.__version__)
print("versi saat submission dibuat:", EXPECTED)
print("DATA_DIR =", DATA_DIR.resolve(), "| OUT_PATH =", OUT_PATH.resolve())
T0 = time.time()
"""))
    io_src = (ROOT / "src" / "io_utils.py").read_text(encoding="utf-8")
    io_code = io_src.split('from .config import DATA_DIR\n', 1)[1].strip("\n")
    cells.append(md("""
## 2. Data

Delapan tabel panitia (folder `DATA_DIR`). Kunci join `customer_id` (1 customer = 1 aplikasi train/test).
Semua fitur tabel pendukung dihitung relatif terhadap `decision_datetime` aplikasi sekarang (cutoff); seluruh
timestamp tabel pendukung sudah < cutoff (diverifikasi saat EDA). Tanggal di-parse dan kolom kategori dibaca
sebagai `category` supaya hemat memori.
"""))
    cells.append(code(io_code + """


apps = load_apps()
m_tr = apps["is_test"].eq(0).values
print(apps.shape, "| target rate train", round(apps.loc[m_tr, "target"].mean(), 4))
"""))
    cells.append(md("""
## 3. Ringkasan EDA (temuan yang membentuk pipeline)

- Split train/test acak (bukan temporal), customer tidak overlap → StratifiedKFold 5 fold valid sebagai CV.
- Target 13,58%; 30% customer tanpa riwayat (target lebih rendah, 11,7%) → missing dibiarkan NaN untuk GBDT.
- Data dibangkitkan dengan **risiko laten customer yang stabil**: label default antarpinjaman lama sangat berulang
  (ICC 0,40); target sekarang berperilaku seperti "satu pinjaman lagi".
- **Pembayaran cicilan hanya terjadi tepat di ulang bulan pencairan** (rantai Markov bulanan; FIFO; 0–3 cicilan per
  event) → representasi level event (bagian 5.9) jauh lebih informatif daripada agregat baris cicilan.
- E-commerce dipotong tepat 24 jam sebelum keputusan; order defaulter lebih rapat menjelang keputusan ("ramp") →
  fitur jendela pendek dan model level order (bagian 6).
- `annual_interest_rate` = 5 tier dengan target rate monoton 5,5% → 50,3% (grade risiko lender).
"""))
    cells.append(code("""
tr_apps = apps[m_tr]
print("target per tier bunga:", tr_apps.groupby("annual_interest_rate")["target"].mean().round(3).to_dict())
print("aplikasi per bulan keputusan:", tr_apps.decision_datetime.dt.to_period("M").value_counts().sort_index().to_dict())
"""))
    common_src = (ROOT / "src" / "features" / "common.py").read_text(encoding="utf-8")
    fin = segment("src/features/common.py", "finalize")
    byc = segment("src/features/common.py", "by_customer")
    days = segment("src/features/common.py", "days_between")
    cells.append(md("""
## 4. Kontrak fitur

Setiap builder `build_<grup>(apps)` mengembalikan DataFrame ber-index `application_id`, kolom ber-prefix dan terurut.
`cached` di pipeline eksperimen menyimpan hasil ke parquet; di notebook ini hasil disimpan di memori (nilainya sama).
"""))
    cells.append(code(f"""
{fin}


{byc}


_CACHE = {{}}


def cached(name, builder, apps, rebuild=False):
    \"\"\"Pengganti cache parquet: hasil grup fitur disimpan di memori, di-reindex ke urutan apps.\"\"\"
    if name in _CACHE and not rebuild:
        return _CACHE[name]
    df = builder(apps)
    df = df.reindex(apps["application_id"])
    df.index.name = "application_id"
    _CACHE[name] = df
    return df


{days}
"""))
    assert "def cached" in common_src
    cells.append(md("## 5. Rekayasa fitur\nKode di bawah identik dengan modul `src/features/` pipeline eksperimen (hanya import yang dihapus)."))
    for path, ren in MODULES:
        doc, body = strip_module(path, ren)
        key = Path(path).stem
        cells.append(md(f"### {TITLES[key]}\n\n" + doc))
        cells.append(code(body))
    cells.append(md("""
### 5.10 Bangun matriks fitur

23 grup (528 kolom, sama dengan set "all" pipeline eksperimen) + 18 kolom rantai bayar event. Kolom diurutkan
(`sorted`) persis seperti `run.build_matrix`. Grup `x_` memakai hasil grup lain lewat cache memori.
"""))
    cells.append(code("""
GROUPS = {
    "app_raw_v1": build_app_raw, "prof_raw_v1": build_prof_raw, "app_ratios_v1": build_app_ratios,
    "pa_naive_v1": build_pa_naive, "rp_naive_v1": build_rp_naive, "bu_naive_v1": build_bu_naive,
    "bp_naive_v1": build_bp_naive, "ab_naive_v1": build_ab_naive, "ec_naive_v1": build_ec_naive,
    "rp_core_v1": build_rp_core, "rp_window_v1": build_rp_window, "rp_loan_v1": build_rp_loan,
    "rp_trend_v1": build_rp_trend, "bu_extra_v1": build_bu_extra, "bp_trend_v1": build_bp_trend,
    "pa_window_v1": build_pa_window, "pa_burden_v1": build_pa_burden, "ab_window_v1": build_ab_window,
    "ab_income_v1": build_ab_income, "ec_window_v1": build_ec_window, "ec_mix_v1": build_ec_mix,
    "x_cross_v1": build_x_cross, "x_rate_resid_v1": build_x_rate_resid,
}
EXTRA = {"rp_events_v1": build_rp_events}


def build_matrix(groups, apps):
    parts = []
    for g in groups:
        t = time.time()
        parts.append(cached(g, GROUPS[g], apps))
        print(f"  fitur {g:<16} {parts[-1].shape[1]:>4} kolom  ({(time.time() - t) / 60:.1f} mnt)", flush=True)
    X = pd.concat(parts, axis=1)
    assert X.columns.is_unique, "nama kolom duplikat antar grup"
    return X[sorted(X.columns)]


X = build_matrix(list(GROUPS), apps)
E = pd.concat([cached(n, f, apps) for n, f in EXTRA.items()], axis=1)
X = pd.concat([X, E], axis=1)
del E
gc.collect()
print("matriks fitur:", X.shape, f"({(time.time() - T0) / 60:.1f} mnt sejak awal)")
"""))
    cells.append(md("""
## 6. Validasi dan fitur nested model level order e-commerce

CV = StratifiedKFold 5 fold (shuffle, `random_state=42`) di train, sama untuk semua model. Fitur `ecm_*` berasal dari
model LightGBM level **order** (1 baris = 1 order, label = target customer) yang diagregasi per customer. Supaya tidak
bocor, fitur ini **nested**: untuk fold luar k, baris validasi fold k dan test diprediksi model yang dilatih di fold ≠ k;
baris latih fold j ≠ k diprediksi model yang dilatih di fold di luar {k, j}. Early stopping model order memakai 10%
customer holdout di dalam data latihnya sendiri. Hasil per fold k: `tr{k}`, `va{k}`, `te{k}`.
"""))
    bo = segment("e121.py", "build_orders")
    bo = bo.replace('    if ORDERS.exists():\n        return pd.read_parquet(ORDERS)\n', "")
    bo = bo.replace("    od.to_parquet(ORDERS)\n", "")
    fr = segment("e121.py", "fit_rows")
    ag = segment("e121.py", "aggregate")
    cells.append(code(f"""
def make_folds(train):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(skf.split(train, train["target"])):
        fold[va] = k
    return pd.DataFrame({{"application_id": train["application_id"].values, "fold": fold}})


y = apps.loc[m_tr, "target"].astype(int).values
fold = make_folds(apps[m_tr])["fold"].values
test_ids = apps.loc[~m_tr, "application_id"].values
print("ukuran fold:", np.bincount(fold).tolist(), "| target rate per fold:", [round(y[fold == k].mean(), 4) for k in range(N_FOLDS)])

CATS = {json.dumps(["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"])}
FEATS = ["recency_h", "total", "total_vi", "pm", "st", "n_items", "qty", "hour", "dow"] + CATS
ROW_P = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=500,
             feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1,
             seed=SEED, deterministic=True, force_row_wise=True, num_threads=4)
AGGX = False


{bo}


{fr}


{ag}


def build_ecm(apps, y, fold):
    \"\"\"Fitur ecm_* nested per fold (sama dengan e121.py --build, hasil di memori).\"\"\"
    tr_apps = apps[apps["is_test"].eq(0).values].reset_index(drop=True)
    od = build_orders(apps)
    cfold = pd.Series(fold, index=tr_apps["customer_id"])
    cy = pd.Series(y, index=tr_apps["customer_id"])
    row_fold = od["customer_id"].map(cfold).fillna(-1).astype(int).values  # -1 = customer test
    y_row = od["customer_id"].map(cy).fillna(0).astype(int).values
    tr_c, te_c = tr_apps["customer_id"].values, apps.loc[apps["is_test"].eq(1).values, "customer_id"].values
    out = {{}}
    for k in range(N_FOLDS):
        t = time.time()
        mk = fit_rows(od, (row_fold >= 0) & (row_fold != k), y_row, SEED + k)
        rows_va, rows_te = row_fold == k, row_fold == -1
        f_va = aggregate(od, mk.predict(od.loc[rows_va, FEATS], raw_score=True), rows_va, tr_c[fold == k])
        f_te = aggregate(od, mk.predict(od.loc[rows_te, FEATS], raw_score=True), rows_te, te_c)
        parts = []
        for j in range(N_FOLDS):
            if j == k:
                continue
            mj = fit_rows(od, (row_fold >= 0) & (row_fold != k) & (row_fold != j), y_row, SEED + 10 * k + j)
            rows_j = row_fold == j
            parts.append(aggregate(od, mj.predict(od.loc[rows_j, FEATS], raw_score=True), rows_j, tr_c[fold == j]))
        f_tr = pd.concat(parts).reindex(tr_c[fold != k])
        out[f"tr{{k}}"], out[f"va{{k}}"], out[f"te{{k}}"] = f_tr.values, f_va.values, f_te.values
        print(f"ecm fold {{k}}: AUC ecm_mean val {{roc_auc_score(y[fold == k], f_va['ecm_mean'].fillna(-9)):.4f}} "
              f"({{(time.time() - t) / 60:.1f}} mnt)", flush=True)
    out["cols"] = np.array(f_va.columns)
    del od
    gc.collect()
    return out


ECM = build_ecm(apps, y, fold)
"""))
    cells.append(md(f"""
## 7. Model

Matriks latih per fold = 546 kolom basis + 7 kolom `ecm_*` (nama berakhiran `0`, sama dengan pipeline eksperimen).
Parameter hasil Optuna Fase 5 (study `tune_lgb`, 100 trial, set B3) — `RUN_TUNING = False`, jadi di-hardcode:

- **GBDT** (lr final 0,02, batas 15000 iterasi, early stopping 200 di fold validasi): `{P['lgb']}`
- **DART** (lr 0,05, 5000 ronde tetap, drop_rate 0,1, skip_drop 0,5, max_drop 50): parameter pohon sama.

Tiap model dilatih 3 seed (42, 43, 44) × 5 fold; prediksi test = rata-rata 5 model fold per seed.
"""))
    cells.append(code(f"""
PARAMS_LGB = {P['lgb']!r}
PARAMS_DART = {P['dart']!r}
SEEDS = [42, 43, 44]
X_tr, X_te = X[m_tr].reset_index(drop=True), X[~m_tr].reset_index(drop=True)
cols_ecm = [f"{{c}}0" for c in ECM["cols"]]


def fold_mats(k):
    tr, va = fold != k, fold == k
    A = pd.concat([X_tr[tr].reset_index(drop=True), pd.DataFrame(ECM[f"tr{{k}}"], columns=cols_ecm)], axis=1)
    B = pd.concat([X_tr[va].reset_index(drop=True), pd.DataFrame(ECM[f"va{{k}}"], columns=cols_ecm)], axis=1)
    C = pd.concat([X_te, pd.DataFrame(ECM[f"te{{k}}"], columns=cols_ecm)], axis=1)
    return tr, va, A, B, C


def ks_score(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def report(name, oof):
    aucs = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)]
    print(f"{{name}}: mean-fold AUC {{np.mean(aucs):.5f}} {{np.round(aucs, 4).tolist()}} | KS OOF {{ks_score(y, oof):.4f}}", flush=True)
"""))
    cells.append(md("### 7.1 GBDT tuned (E205)"))
    cells.append(code("""
OOF_GBDT, PRED_GBDT = {}, {}
for s in SEEDS:
    oof, pred, its = np.zeros(len(y)), np.zeros(len(X_te)), []
    for k in range(N_FOLDS):
        tr, va, A, B, C = fold_mats(k)
        dtr = lgb.Dataset(A, y[tr], params={"max_bin": PARAMS_LGB.get("max_bin", 255)})
        dva = lgb.Dataset(B, y[va], reference=dtr)
        mdl = lgb.train({**PARAMS_LGB, "seed": s}, dtr, 15000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = mdl.predict(B, num_iteration=mdl.best_iteration)
        pred += mdl.predict(C, num_iteration=mdl.best_iteration) / N_FOLDS
        its.append(mdl.best_iteration)
        del A, B, C, dtr, dva, mdl
        gc.collect()
    OOF_GBDT[s], PRED_GBDT[s] = oof, pred
    report(f"GBDT seed {s} (iter {its})", oof)
report("GBDT rata-rata 3 seed", np.mean([OOF_GBDT[s] for s in SEEDS], 0))
print(f"({(time.time() - T0) / 60:.1f} mnt sejak awal)")
"""))
    cells.append(md("### 7.2 DART (E207)\nDART tidak memakai early stopping (jumlah ronde tetap 5000); prediksi memakai semua ronde."))
    cells.append(code("""
OOF_DART, PRED_DART = {}, {}
for s in SEEDS:
    oof, pred = np.zeros(len(y)), np.zeros(len(X_te))
    for k in range(N_FOLDS):
        tr, va, A, B, C = fold_mats(k)
        dtr = lgb.Dataset(A, y[tr], params={"max_bin": PARAMS_DART.get("max_bin", 255)})
        dva = lgb.Dataset(B, y[va], reference=dtr)
        ev = {}
        mdl = lgb.train({**PARAMS_DART, "seed": s, "drop_seed": s}, dtr, 5000, valid_sets=[dva],
                        callbacks=[lgb.record_evaluation(ev)])
        oof[va] = mdl.predict(B)
        pred += mdl.predict(C) / N_FOLDS
        print(f"  DART seed {s} fold {k}: AUC {roc_auc_score(y[va], oof[va]):.5f} ({(time.time() - T0) / 60:.1f} mnt)", flush=True)
        del A, B, C, dtr, dva, mdl
        gc.collect()
    OOF_DART[s], PRED_DART[s] = oof, pred
    report(f"DART seed {s}", oof)
report("DART rata-rata 3 seed", np.mean([OOF_DART[s] for s in SEEDS], 0))
"""))
    cells.append(md("""
## 8. Ensemble dan submission

Per model: rank dari rata-rata rank antar seed (urutan seed 42, 43, 44). Blend = rata-rata berbobot 0,5/0,5 dari rank
ternormalisasi; skor akhir = rank blend / 30.000 dibulatkan 6 desimal (KS hanya bergantung pada urutan). File ditulis
dengan urutan `sample_submission` dan akhir baris LF.
"""))
    cells.append(code("""
def bag(arrs):
    return rankdata(np.mean([rankdata(a) for a in arrs], 0))


parts = [(bag([OOF_GBDT[s] for s in SEEDS]), bag([PRED_GBDT[s] for s in SEEDS]), 0.5),
         (bag([OOF_DART[s] for s in SEEDS]), bag([PRED_DART[s] for s in SEEDS]), 0.5)]
W = sum(w for _, _, w in parts)
O, T = 0, 0
for o, t, w in parts:
    O, T = O + w / W * o / len(o), T + w / W * t / len(t)
report("Blend GBDT + DART (OOF)", O)
print("KS OOF (ks_2samp):", round(ks_2samp(O[y == 1], O[y == 0]).statistic, 4))
t = (rankdata(T) / len(T)).round(6)
ss = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub = ss[["application_id"]].merge(pd.DataFrame({"application_id": test_ids, "default_probability": t}),
                                   on="application_id", how="left")
assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.notna().all()
assert sub.default_probability.between(0, 1).all()
sub.to_csv(OUT_PATH, index=False, lineterminator="\\n")
md5 = hashlib.md5(open(OUT_PATH, "rb").read()).hexdigest()
print("submission:", OUT_PATH, "| MD5", md5, "| identik dengan submission terpilih:", md5 == "afc487f68e4603a38209f2d4a472312b")
print(f"total waktu {(time.time() - T0) / 60:.1f} mnt")
"""))
    cells.append(md("""
## 9. Interpretasi (SHAP)

Kontribusi fitur model GBDT fold 0 (TreeSHAP bawaan LightGBM, `pred_contrib`). Sel ini dijalankan setelah submission
ditulis dan tidak mengubahnya.
"""))
    cells.append(code("""
tr, va, A, B, C = fold_mats(0)
dtr = lgb.Dataset(A, y[tr], params={"max_bin": 255})
mdl = lgb.train({**PARAMS_LGB, "seed": 42}, dtr, 15000, valid_sets=[lgb.Dataset(B, y[va], reference=dtr)],
                callbacks=[lgb.early_stopping(200, verbose=False)])
sv = mdl.predict(B, num_iteration=mdl.best_iteration, pred_contrib=True)[:, :-1]
imp = pd.Series(np.abs(sv).mean(0), index=B.columns).sort_values(ascending=False)
print(imp.head(20).round(4).to_string())
"""))
    cells.append(md("""
## 10. Kesimpulan

- Sinyal terkuat: riwayat bayar (direpresentasikan sebagai rantai event bulanan), model level order e-commerce
  (ramp belanja menjelang keputusan), skor biro, dan tier bunga lender.
- Residual model tidak lagi bisa dipelajari (uji kapasitas, ±60 uji sinyal, 10.760 fitur otomatis) → model berada dekat
  batas informasi data untuk representasi ini; kenaikan terakhir datang dari representasi yang lebih tepat (rantai bayar)
  dan ensemble GBDT + DART.
- Submission di atas dihasilkan sepenuhnya oleh notebook ini (CPU, deterministik) dengan MD5 yang dicetak di bagian 8.
"""))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out} ({len(cells)} sel)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "final" / "NomorPeserta_NamaTim_Notebook.ipynb"))
    build(Path(ap.parse_args().out))
