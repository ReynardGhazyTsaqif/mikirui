"""Audit angka untuk matriks fitur level aplikasi (train + test).

Memeriksa:
  - cakupan application_id: semua train dan test ada, tanpa duplikat, tanpa ID asing
  - nama kolom: *_id / target terlarang, prefix tak dikenal, urutan tidak sorted
  - per kolom: konstan, inf, beda tingkat NaN train vs test, beda distribusi
    train vs test (statistik KS dua sampel), AUC univariat terhadap target train

Exit code 1 jika ada ERROR.

Contoh:
    python .claude/skills/audit-leakage/scripts/audit_features.py --features features/ec_window_v2.parquet --prefix ec_
    python .claude/skills/audit-leakage/scripts/audit_features.py --features X.csv --report experiments/audit_ec.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

PREFIXES = ("app_", "prof_", "pa_", "rp_", "bu_", "bp_", "ab_", "ec_", "x_")

AUC_ERR, AUC_WARN = 0.80, 0.75          # rp_share_dpd1 = 0,784 adalah fitur sah terkuat
KS_ERR, KS_WARN = 0.05, 0.02            # noise KS 70k vs 30k ~ 0,009
NAN_DIFF_ERR = 0.02


def read_table(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if "application_id" in df.columns:
        df = df.set_index("application_id")
    elif df.index.name != "application_id":
        sys.exit("ERROR: file fitur butuh kolom atau index 'application_id'")
    return df


def numeric(s: pd.Series) -> np.ndarray:
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.astype("category").cat.codes.replace(-1, np.nan).to_numpy(dtype=float)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="parquet/csv dengan application_id + fitur")
    ap.add_argument("--data-dir", default="Dataset")
    ap.add_argument("--cols", nargs="*")
    ap.add_argument("--prefix", help="hanya audit kolom dengan prefix ini (mis. kolom baru)")
    ap.add_argument("--report", help="tulis tabel per kolom ke CSV")
    ap.add_argument("--allow-partial", action="store_true",
                    help="izinkan file yang tidak mencakup semua aplikasi (cakupan jadi WARN)")
    a = ap.parse_args()

    tr = pd.read_csv(os.path.join(a.data_dir, "train_applications.csv"), usecols=["application_id", "target"])
    te = pd.read_csv(os.path.join(a.data_dir, "test_applications.csv"), usecols=["application_id"])
    F = read_table(a.features)
    issues = []  # (level, scope, pesan)

    def add(level, scope, msg):
        issues.append((level, scope, msg))

    # ---- cakupan ID
    dup = int(F.index.duplicated().sum())
    if dup:
        add("ERROR", "id", f"{dup} application_id duplikat")
    ids = set(F.index)
    miss_tr = len(set(tr.application_id) - ids)
    miss_te = len(set(te.application_id) - ids)
    extra = len(ids - set(tr.application_id) - set(te.application_id))
    lvl = "WARN" if a.allow_partial else "ERROR"
    if miss_tr:
        add(lvl, "id", f"{miss_tr} aplikasi train tidak ada di file fitur")
    if miss_te:
        add(lvl, "id", f"{miss_te} aplikasi test tidak ada di file fitur")
    if extra:
        add("ERROR", "id", f"{extra} application_id bukan train/test (mis. ID previous application)")
    F = F[~F.index.duplicated()]
    if a.allow_partial:  # audit hanya baris yang memang ada di file
        tr = tr[tr.application_id.isin(ids)]
        te = te[te.application_id.isin(ids)]

    # ---- nama kolom
    cols = a.cols or list(F.columns)
    if "target" in F.columns:
        add("ERROR", "kolom", "kolom 'target' ada di file fitur")
    forbidden = [c for c in cols if c.lower().endswith("_id") or c.lower() in ("id", "target")]
    for c in forbidden:
        add("ERROR", c, "kolom ID/target tidak boleh jadi fitur")
    cols = [c for c in cols if c not in forbidden]
    if a.prefix:
        cols = [c for c in cols if c.startswith(a.prefix)]
    if not cols:
        sys.exit("ERROR: tidak ada kolom untuk diaudit")
    unknown = [c for c in cols if not c.startswith(PREFIXES)]
    if unknown:
        add("WARN", "kolom", f"{len(unknown)} kolom tanpa prefix standar {PREFIXES}: {unknown[:8]}")
    if list(F.columns) != sorted(F.columns):
        add("WARN", "kolom", "urutan kolom file tidak sorted() - urutkan agar deterministik")

    # ---- per kolom
    Xtr = tr.join(F[cols], on="application_id")
    Xte = te.join(F[cols], on="application_id")
    y = Xtr["target"].to_numpy()
    rows = []
    for c in cols:
        a_tr, a_te = numeric(Xtr[c]), numeric(Xte[c])
        row = dict(kolom=c)
        n_inf = int(np.isinf(a_tr).sum() + np.isinf(a_te).sum())
        a_tr = np.where(np.isinf(a_tr), np.nan, a_tr)
        a_te = np.where(np.isinf(a_te), np.nan, a_te)
        nan_tr, nan_te = np.isnan(a_tr).mean(), np.isnan(a_te).mean()
        row.update(nan_train=nan_tr, nan_test=nan_te, inf=n_inf)
        allv = np.concatenate([a_tr, a_te])
        nun = pd.Series(allv).nunique(dropna=True)
        row["n_unik"] = nun
        if n_inf:
            add("WARN", c, f"{n_inf} nilai inf (ganti NaN; cek pembagian nol)")
        if nun <= 1:
            add("WARN", c, "konstan (atau semua NaN) - buang")
            rows.append(row)
            continue
        if abs(nan_tr - nan_te) > NAN_DIFF_ERR:
            add("ERROR", c, f"tingkat NaN beda: train {nan_tr:.1%} vs test {nan_te:.1%}")
        ok_tr, ok_te = ~np.isnan(a_tr), ~np.isnan(a_te)
        if ok_tr.sum() > 50 and ok_te.sum() > 50:
            kst = ks_2samp(a_tr[ok_tr], a_te[ok_te]).statistic
            row["ks_train_test"] = kst
            if kst > KS_ERR:
                add("ERROR", c, f"distribusi train vs test beda (KS {kst:.3f})")
            elif kst > KS_WARN:
                add("WARN", c, f"distribusi train vs test agak beda (KS {kst:.3f})")
        if ok_tr.sum() > 100 and len(np.unique(y[ok_tr])) == 2:
            auc = roc_auc_score(y[ok_tr], a_tr[ok_tr])
            auc = max(auc, 1 - auc)
            row["auc_univariat"] = auc
            if auc >= AUC_ERR:
                add("ERROR", c, f"AUC univariat {auc:.3f} >= {AUC_ERR} - lebih kuat dari fitur sah terkuat; cari kebocoran")
            elif auc >= AUC_WARN:
                add("WARN", c, f"AUC univariat {auc:.3f} - periksa jalur ke target/masa depan")
        if 0.01 < nan_tr < 0.99:
            auc_nan = roc_auc_score(y, np.isnan(a_tr).astype(float))
            row["auc_flag_nan"] = max(auc_nan, 1 - auc_nan)
        rows.append(row)

    rep = pd.DataFrame(rows)
    if a.report:
        rep.to_csv(a.report, index=False)
    print(f"File   : {a.features}  ({F.shape[0]} baris, {len(cols)} kolom diaudit)")
    print(f"Cakupan: train hilang {miss_tr}, test hilang {miss_te}, ID asing {extra}, duplikat {dup}")
    show = rep.sort_values("auc_univariat", ascending=False) if "auc_univariat" in rep else rep
    with pd.option_context("display.width", 160, "display.max_rows", 30):
        print("\nKolom dengan AUC univariat tertinggi:")
        print(show.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    errs = [i for i in issues if i[0] == "ERROR"]
    warns = [i for i in issues if i[0] == "WARN"]
    print(f"\nERROR {len(errs)}, WARN {len(warns)}")
    for lvl_, scope, msg in errs + warns[:40]:
        print(f"  [{lvl_}] {scope}: {msg}")
    if len(warns) > 40:
        print(f"  ... {len(warns) - 40} WARN lain (lihat --report)")
    print("\nSTATUS: " + ("PERLU PERBAIKAN" if errs else "Lapis 2 aman (tetap lakukan Lapis 1 dan adversarial validation)"))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
