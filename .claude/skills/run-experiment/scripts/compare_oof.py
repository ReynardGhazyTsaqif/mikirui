"""Bandingkan OOF dua eksperimen secara berpasangan dan terapkan aturan terima.

Contoh:
    python .claude/skills/run-experiment/scripts/compare_oof.py --base T_lgb_b3_4f3c7f --cand E140a
    python .claude/skills/run-experiment/scripts/compare_oof.py --base oof/a.npy --cand oof/b.npy --blend --bootstrap 300

Nama eksperimen dicari sebagai oof/<nama>_s<seed>.npy (>= 2 file -> dirata-rata)
atau oof/<nama>.npy. OOF diasumsikan berurutan sama dengan baris
Dataset/train_applications.csv. Cek dulu dengan --base X --cand X: mean-fold AUC
harus sama dengan angka di experiments/log.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5


def resolve_oof(name: str, oof_dir: str, seeds: list[int] | None):
    """Kembalikan (array, daftar file, jumlah seed)."""
    if name.endswith(".npy") and os.path.exists(name):
        files = [name]
    else:
        seed_files = sorted(glob.glob(os.path.join(oof_dir, f"{glob.escape(name)}_s*.npy")))
        seed_files = [f for f in seed_files
                      if re.fullmatch(rf"{re.escape(name)}_s\d+\.npy", os.path.basename(f))]
        if seeds:
            seed_files = [f for f in seed_files
                          if int(re.search(r"_s(\d+)\.npy$", f).group(1)) in seeds]
        plain = os.path.join(oof_dir, f"{name}.npy")
        if len(seed_files) >= 2:
            files = seed_files
        elif os.path.exists(plain):
            files = [plain]
        elif seed_files:
            files = seed_files
        else:
            sys.exit(f"ERROR: OOF untuk '{name}' tidak ditemukan di {oof_dir} "
                     f"(dicari {name}_s*.npy dan {name}.npy)")
    arrs = []
    for f in files:
        a = np.load(f)
        a = np.asarray(a, dtype=float).reshape(len(a), -1)
        if a.shape[1] != 1:
            sys.exit(f"ERROR: {f} berbentuk {a.shape}; harus 1 kolom probabilitas")
        arrs.append(a[:, 0])
    n_seed = len(files)
    if len(files) == 1 and not re.search(r"_s\d+\.npy$", files[0]):
        n_seed = None  # tidak diketahui: bisa jadi file bag
    return np.mean(arrs, axis=0), files, n_seed


def load_target_and_folds(data_dir: str, folds_path: str | None):
    train = pd.read_csv(os.path.join(data_dir, "train_applications.csv"),
                        usecols=["application_id", "target"])
    y = train["target"].to_numpy()
    if folds_path and os.path.exists(folds_path):
        f = pd.read_csv(folds_path)
        if not {"application_id", "fold"} <= set(f.columns):
            sys.exit(f"ERROR: {folds_path} harus punya kolom application_id, fold")
        fold = train[["application_id"]].merge(f, on="application_id", how="left")["fold"]
        if fold.isna().any():
            sys.exit(f"ERROR: {fold.isna().sum()} baris train tidak punya fold di {folds_path}")
        src = folds_path
        fold = fold.to_numpy().astype(int)
    else:
        fold = np.full(len(y), -1)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold[va] = k
        src = f"StratifiedKFold({N_FOLDS}, shuffle=True, random_state={SEED}) dibuat ulang"
    return y, fold, src


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def per_fold(y, p, fold):
    ks_list, auc_list = [], []
    for k in np.unique(fold):
        m = fold == k
        auc_list.append(roc_auc_score(y[m], p[m]))
        ks_list.append(ks(y[m], p[m]))
    return np.array(auc_list), np.array(ks_list)


def bootstrap_delta(y, pb, pc, fold, n, seed=SEED):
    """Bootstrap berstrata per fold untuk delta mean-fold AUC."""
    rng = np.random.default_rng(seed)
    idx_by_fold = [np.flatnonzero(fold == k) for k in np.unique(fold)]
    out = []
    for _ in range(n):
        d = []
        for idx in idx_by_fold:
            s = rng.choice(idx, size=len(idx), replace=True)
            if y[s].min() == y[s].max():
                continue
            d.append(roc_auc_score(y[s], pc[s]) - roc_auc_score(y[s], pb[s]))
        out.append(np.mean(d))
    return np.percentile(out, [2.5, 97.5]), float(np.mean(np.array(out) > 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="nama eksperimen acuan atau path .npy")
    ap.add_argument("--cand", required=True, help="nama eksperimen kandidat atau path .npy")
    ap.add_argument("--oof-dir", default="oof")
    ap.add_argument("--data-dir", default="Dataset")
    ap.add_argument("--folds", default="folds/folds.csv",
                    help="CSV application_id,fold; jika tidak ada, fold dibuat ulang (seed 42)")
    ap.add_argument("--seeds", type=int, nargs="*", help="batasi ke seed tertentu")
    ap.add_argument("--min-delta", type=float, default=0.0005)
    ap.add_argument("--min-folds", type=int, default=4)
    ap.add_argument("--thin", type=float, default=0.001,
                    help="di bawah ini keputusan dianggap tipis -> wajib 3 seed")
    ap.add_argument("--blend", action="store_true", help="kandidat adalah blend: wajib >= 3 seed")
    ap.add_argument("--bootstrap", type=int, default=0, help="jumlah resample bootstrap (0 = mati)")
    ap.add_argument("--json", help="tulis ringkasan ke file JSON")
    a = ap.parse_args()

    y, fold, fold_src = load_target_and_folds(a.data_dir, a.folds)
    pb, fb, nb = resolve_oof(a.base, a.oof_dir, a.seeds)
    pc, fc, nc = resolve_oof(a.cand, a.oof_dir, a.seeds)
    for nm, p in (("base", pb), ("cand", pc)):
        if len(p) != len(y):
            sys.exit(f"ERROR: panjang OOF {nm} = {len(p)}, train = {len(y)}")
        if np.isnan(p).any():
            sys.exit(f"ERROR: OOF {nm} mengandung NaN")

    auc_b, ks_b = per_fold(y, pb, fold)
    auc_c, ks_c = per_fold(y, pc, fold)
    d_fold = auc_c - auc_b
    delta = float(auc_c.mean() - auc_b.mean())
    n_up = int((d_fold > 0).sum())
    rho = spearmanr(pb, pc).statistic

    def seeds_txt(n):
        return "?" if n is None else str(n)

    print(f"Fold     : {fold_src}")
    print(f"Base     : {a.base}  ({len(fb)} file, seed {seeds_txt(nb)})")
    print(f"Cand     : {a.cand}  ({len(fc)} file, seed {seeds_txt(nc)})")
    print()
    print(f"{'fold':>4} {'AUC base':>9} {'AUC cand':>9} {'delta':>9} {'KS base':>8} {'KS cand':>8}")
    for k in range(len(auc_b)):
        print(f"{k:>4} {auc_b[k]:9.5f} {auc_c[k]:9.5f} {d_fold[k]:+9.5f} {ks_b[k]:8.4f} {ks_c[k]:8.4f}")
    print(f"{'mean':>4} {auc_b.mean():9.5f} {auc_c.mean():9.5f} {delta:+9.5f} {ks_b.mean():8.4f} {ks_c.mean():8.4f}")
    print()
    print(f"OOF pooled AUC : {roc_auc_score(y, pb):.5f} -> {roc_auc_score(y, pc):.5f}")
    print(f"OOF pooled KS  : {ks(y, pb):.4f} -> {ks(y, pc):.4f}")
    print(f"Spearman base~cand: {rho:.4f}")

    ci = p_pos = None
    if a.bootstrap:
        ci, p_pos = bootstrap_delta(y, pb, pc, fold, a.bootstrap)
        print(f"Bootstrap {a.bootstrap}x: 95% CI delta [{ci[0]:+.5f}, {ci[1]:+.5f}], P(delta>0) = {p_pos:.2f}")

    known = [n for n in (nb, nc) if n is not None]
    min_seed = min(known) if known else None
    notes = []
    passed = delta >= a.min_delta and n_up >= a.min_folds
    if passed and delta < a.thin and (min_seed is None or min_seed < 3):
        decision = "TIPIS - ulangi dengan 3 seed sebelum diterima"
    elif passed:
        decision = "DITERIMA"
    else:
        decision = "DITOLAK"
    if a.blend and (min_seed is None or min_seed < 3):
        notes.append("Blend dinilai dengan OOF < 3 seed (atau jumlah seed tidak diketahui): "
                     "kenaikan blend bisa ~2x terlalu optimis. Ulangi dengan OOF rata-rata >= 3 seed.")
        if decision == "DITERIMA":
            decision = "TIPIS - blend wajib dinilai dengan OOF >= 3 seed"
    if delta > 0.005:
        notes.append("Delta > +0,005 dari satu perubahan: jalankan audit-leakage sebelum melapor diterima.")
    if a.cand == a.base:
        notes.append("Base = cand: ini cek penyelarasan. Bandingkan mean-fold AUC di atas dengan log.csv.")

    print()
    print(f"Aturan   : delta mean-fold AUC >= {a.min_delta:+.4f} dan naik di >= {a.min_folds}/{len(auc_b)} fold")
    print(f"Hasil    : delta {delta:+.5f}, naik {n_up}/{len(auc_b)} fold")
    print(f"KEPUTUSAN: {decision}")
    for n in notes:
        print(f"CATATAN  : {n}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({
                "base": a.base, "cand": a.cand, "base_files": fb, "cand_files": fc,
                "fold_source": fold_src,
                "auc_base": auc_b.tolist(), "auc_cand": auc_c.tolist(),
                "ks_base": ks_b.tolist(), "ks_cand": ks_c.tolist(),
                "mean_fold_auc_base": float(auc_b.mean()), "mean_fold_auc_cand": float(auc_c.mean()),
                "delta": delta, "folds_up": n_up, "spearman": float(rho),
                "bootstrap_ci": None if ci is None else [float(ci[0]), float(ci[1])],
                "decision": decision, "notes": notes,
            }, fh, indent=2)


if __name__ == "__main__":
    main()
