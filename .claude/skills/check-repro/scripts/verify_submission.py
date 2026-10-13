"""Validasi file submission DSC LOGIKA UI 2026, cetak MD5, dan (opsional) bandingkan.

Contoh:
    python .claude/skills/check-repro/scripts/verify_submission.py submissions/P5_single.csv
    python .claude/skills/check-repro/scripts/verify_submission.py run2.csv --compare run1.csv
    python .claude/skills/check-repro/scripts/verify_submission.py sub.csv --log submissions/LOG.md --note "E140a, AUC 0.8385"

Exit code 1 jika ada ERROR atau (dengan --compare) MD5 berbeda.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

COLS = ["application_id", "default_probability"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def line_ending(path):
    with open(path, "rb") as fh:
        head = fh.read(1 << 16)
    if b"\r\n" in head:
        return "CRLF"
    return "LF" if b"\n" in head else "?"


def max_decimals(path, n=2000):
    raw = pd.read_csv(path, dtype=str, nrows=n)
    if "default_probability" not in raw:
        return None
    s = raw["default_probability"].dropna()
    s = s[~s.str.contains("e", case=False)]
    dec = s.str.split(".").str[1].fillna("").str.len()
    return int(dec.max()) if len(dec) else 0


def validate(path, data_dir):
    errs, warns = [], []
    df = pd.read_csv(path)
    if list(df.columns) != COLS:
        errs.append(f"kolom {list(df.columns)} != {COLS}")
        return df, errs, warns
    ref_path = os.path.join(data_dir, "sample_submission.csv")
    if os.path.exists(ref_path):
        ref_ids = pd.read_csv(ref_path, usecols=["application_id"])["application_id"]
    else:
        ref_ids = pd.read_csv(os.path.join(data_dir, "test_applications.csv"),
                              usecols=["application_id"])["application_id"]
        warns.append("sample_submission.csv tidak ditemukan; ID dicek terhadap test_applications.csv")
    test_path = os.path.join(data_dir, "test_applications.csv")
    if len(df) != len(ref_ids):
        errs.append(f"{len(df)} baris, seharusnya {len(ref_ids)}")
    if df["application_id"].duplicated().any():
        errs.append(f"{int(df['application_id'].duplicated().sum())} application_id duplikat")
    s_sub, s_ref = set(df["application_id"]), set(ref_ids)
    if s_sub != s_ref:
        errs.append(f"ID beda dengan acuan: {len(s_ref - s_sub)} hilang, {len(s_sub - s_ref)} asing")
    if os.path.exists(test_path) and os.path.exists(ref_path):
        t_ids = set(pd.read_csv(test_path, usecols=["application_id"])["application_id"])
        if t_ids != s_ref:
            warns.append("sample_submission dan test_applications berisi ID berbeda - cek data")
    p = pd.to_numeric(df["default_probability"], errors="coerce")
    if p.isna().any():
        errs.append(f"{int(p.isna().sum())} probabilitas NaN/non-numerik")
    if ((p < 0) | (p > 1)).any():
        errs.append(f"{int(((p < 0) | (p > 1)).sum())} nilai di luar [0, 1]")
    if p.nunique() <= 1:
        warns.append("prediksi konstan (KS = 0)")
    elif p.nunique() < len(p) * 0.5:
        warns.append(f"hanya {p.nunique()} nilai unik - banyak seri, KS bisa terpengaruh pembulatan")
    nd = max_decimals(path)
    if nd is not None and nd > 6:
        warns.append(f"hingga {nd} desimal; bulatkan ke 6 (round(6)) agar MD5 stabil antar mesin")
    return df, errs, warns


def compare(path, other):
    print(f"\nBandingkan dengan {other}")
    m1, m2 = md5(path), md5(other)
    print(f"  MD5 {m1} vs {m2} -> {'SAMA' if m1 == m2 else 'BEDA'}")
    if m1 == m2:
        return True
    le1, le2 = line_ending(path), line_ending(other)
    a = pd.read_csv(path)
    b = pd.read_csv(other)
    m = a.merge(b, on="application_id", suffixes=("_a", "_b"))
    print(f"  akhir baris {le1} vs {le2}; baris {len(a)} vs {len(b)}; cocok ID {len(m)}")
    if not len(m):
        print("  DIAGNOSIS: tidak ada ID yang sama - file dari pipeline berbeda")
        return False
    pa, pb = m["default_probability_a"].to_numpy(), m["default_probability_b"].to_numpy()
    diff = np.abs(pa - pb)
    rho = spearmanr(pa, pb).statistic
    print(f"  selisih maks {diff.max():.3g}, rata-rata {diff.mean():.3g}, Spearman {rho:.6f}")
    print(f"  urutan baris sama: {list(a['application_id']) == list(b['application_id'])}")
    if diff.max() == 0 and len(a) == len(b) and list(a["application_id"]) == list(b["application_id"]):
        diag = ("isi identik; beda hanya format file (akhir baris/presisi teks). "
                "Pakai to_csv(..., lineterminator='\\n') dan round(6)")
    elif diff.max() == 0:
        diag = "nilai identik tapi urutan baris/ID beda - samakan urutan (merge ke sample_submission)"
    elif diff.max() <= 2e-6 and rho > 0.99999:
        diag = "floating point/jumlah thread - bulatkan 6 desimal dan kunci num_threads"
    elif rho > 0.9999:
        diag = "hampir sama; kemungkinan thread atau urutan operasi - kunci num_threads, deterministic=True"
    else:
        diag = ("sumber acak tidak terkunci atau pipeline beda - cari seed yang hilang, iterasi set(), "
                "time_limit, urutan kolom, fitur bergantung urutan baris")
    print(f"  DIAGNOSIS: {diag}")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--data-dir", default="Dataset")
    ap.add_argument("--compare", help="file lain untuk dicocokkan (MD5 dan isi)")
    ap.add_argument("--log", help="tambah satu baris ke log markdown, mis. submissions/LOG.md")
    ap.add_argument("--note", default="", help="catatan untuk baris log (exp_id, AUC, KS, ...)")
    a = ap.parse_args()

    df, errs, warns = validate(a.path, a.data_dir)
    h = md5(a.path)
    print(f"File   : {a.path}")
    print(f"MD5    : {h}")
    print(f"Baris  : {len(df)}   akhir baris: {line_ending(a.path)}")
    if "default_probability" in df:
        p = pd.to_numeric(df["default_probability"], errors="coerce")
        print(f"Prob   : min {p.min():.6f}  rata-rata {p.mean():.6f}  maks {p.max():.6f}  unik {p.nunique()}")
    for e in errs:
        print(f"[ERROR] {e}")
    for w in warns:
        print(f"[WARN] {w}")
    ok = not errs
    print("VALIDASI: " + ("LOLOS" if ok else "GAGAL - jangan upload"))

    same = True
    if a.compare:
        same = compare(a.path, a.compare)

    if a.log and ok:
        new = not os.path.exists(a.log)
        with open(a.log, "a", encoding="utf-8", newline="\n") as fh:
            if new:
                fh.write("| waktu | file | md5 | baris | catatan | public |\n|---|---|---|---|---|---|\n")
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            fh.write(f"| {ts} | {os.path.basename(a.path)} | {h} | {len(df)} | {a.note} |  |\n")
        print(f"Dicatat ke {a.log} (isi kolom public setelah skor keluar)")
    sys.exit(0 if ok and same else 1)


if __name__ == "__main__":
    main()
