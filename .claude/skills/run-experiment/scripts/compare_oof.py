"""Bandingkan OOF dua eksperimen secara berpasangan dan terapkan aturan keputusan (berlaku 17 Okt 2026).

Aturan (PROMPT_UTAMA Langkah 4):
  - Kandidat dan acuan dibandingkan memakai OOF rata-rata >= 3 seed, pada fold yang sama.
  - Delta = selisih mean-fold AUC. Bootstrap berpasangan berstrata per fold (1000x, seed 42) -> P(delta > 0), CI 95%.
  - DITERIMA : P >= 0,95 dan delta >= +0,0003
  - KANDIDAT : 0,80 <= P < 0,95, atau P >= 0,95 tetapi delta < +0,0003 (simpan OOF & preds untuk digabung nanti)
  - DITOLAK  : P < 0,80
  - Jumlah fold yang naik hanya informasi. P >= 0,95 dengan hanya 1-2 fold naik -> periksa fold anomali dulu.
  - Delta > +0,005 dari satu perubahan -> audit kebocoran dulu sebelum diterima.
  - KS dicatat, keputusan memakai mean-fold AUC.

Contoh:
    python .claude/skills/run-experiment/scripts/compare_oof.py --base E122x1 --cand E200
    python .claude/skills/run-experiment/scripts/compare_oof.py --base oof/a.npy --cand oof/b.npy --json hasil.json

Nama eksperimen dicari sebagai oof/<nama>_s<seed>.npy (>= 2 file -> dirata-rata)
atau oof/<nama>.npy (jumlah seed tidak diketahui). OOF diasumsikan berurutan sama dengan baris
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


class _WeightedAUC:
    """AUC berbobot cepat untuk satu fold: urutan skor dihitung sekali, tiap resample hanya bincount O(n).

    Bobot = berapa kali baris terambil dalam resample bootstrap; nilai seri diberi setengah (sama dengan roc_auc_score).
    """

    def __init__(self, y, p):
        self.vals, self.inv = np.unique(p, return_inverse=True)
        self.pos = (y == 1).astype(float)
        self.neg = 1.0 - self.pos
        self.g = len(self.vals)

    def __call__(self, w):
        pw = np.bincount(self.inv, weights=w * self.pos, minlength=self.g)
        nw = np.bincount(self.inv, weights=w * self.neg, minlength=self.g)
        below = np.cumsum(nw) - nw
        tp, tn = pw.sum(), nw.sum()
        return float(np.dot(pw, below + 0.5 * nw) / (tp * tn))


def bootstrap_delta(y, pb, pc, fold, n, seed=SEED):
    """Bootstrap berpasangan berstrata per fold untuk delta mean-fold AUC.

    Tiap resample: di setiap fold baris diambil ulang dengan pengembalian (ukuran sama), baris yang sama dipakai
    untuk acuan dan kandidat; delta = rata-rata (AUC cand - AUC base) antar fold.
    Return: (array delta bootstrap, CI 95%, P(delta > 0)).
    """
    rng = np.random.default_rng(seed)
    folds = []
    for k in np.unique(fold):
        idx = np.flatnonzero(fold == k)
        folds.append((len(idx), _WeightedAUC(y[idx], pb[idx]), _WeightedAUC(y[idx], pc[idx])))
    out = np.empty(n)
    for b in range(n):
        d = []
        for m, fb, fc in folds:
            w = np.bincount(rng.integers(0, m, m), minlength=m).astype(float)
            d.append(fc(w) - fb(w))
        out[b] = np.mean(d)
    return out, np.percentile(out, [2.5, 97.5]), float(np.mean(out > 0))


def decide(delta, p_pos, n_up, n_folds, min_seed, a):
    """Terapkan aturan 17 Okt. Return (keputusan, catatan)."""
    notes = []
    if p_pos >= a.p_accept and delta >= a.min_delta:
        decision = "DITERIMA"
        if n_up <= 2:
            decision = "DITERIMA (BERSYARAT)"
            notes.append(f"P >= {a.p_accept} tetapi hanya {n_up}/{n_folds} fold naik: periksa fold anomali sebelum menerima.")
    elif p_pos >= a.p_accept:
        decision = "KANDIDAT"
        notes.append(f"P >= {a.p_accept} tetapi delta < {a.min_delta:+.4f}: simpan OOF & preds untuk digabung nanti.")
    elif p_pos >= a.p_candidate:
        decision = "KANDIDAT"
        notes.append("Simpan OOF & preds untuk digabung nanti.")
    else:
        decision = "DITOLAK"
    if min_seed is None or min_seed < a.min_seeds:
        notes.append(f"Salah satu sisi < {a.min_seeds} seed (atau jumlah seed tidak diketahui): keputusan BELUM SAH; "
                     f"ulangi dengan OOF rata-rata >= {a.min_seeds} seed.")
        decision = f"BELUM SAH ({decision} bila seed cukup)"
    if delta > 0.005:
        notes.append("Delta > +0,005 dari satu perubahan: jalankan audit-leakage sebelum melapor diterima.")
    return decision, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="nama eksperimen acuan atau path .npy")
    ap.add_argument("--cand", required=True, help="nama eksperimen kandidat atau path .npy")
    ap.add_argument("--oof-dir", default="oof")
    ap.add_argument("--data-dir", default="Dataset")
    ap.add_argument("--folds", default="folds/folds.csv",
                    help="CSV application_id,fold; jika tidak ada, fold dibuat ulang (seed 42)")
    ap.add_argument("--seeds", type=int, nargs="*", help="batasi ke seed tertentu")
    ap.add_argument("--bootstrap", type=int, default=1000, help="jumlah resample bootstrap (default 1000)")
    ap.add_argument("--p-accept", type=float, default=0.95, help="ambang P(delta>0) untuk DITERIMA")
    ap.add_argument("--p-candidate", type=float, default=0.80, help="ambang P(delta>0) minimal untuk KANDIDAT")
    ap.add_argument("--min-delta", type=float, default=0.0003, help="delta minimal untuk DITERIMA")
    ap.add_argument("--min-seeds", type=int, default=3, help="seed minimal tiap sisi agar keputusan sah")
    ap.add_argument("--blend", action="store_true", help="(kompatibilitas) kandidat adalah blend; aturan seed sama")
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

    boots, ci, p_pos = bootstrap_delta(y, pb, pc, fold, a.bootstrap)
    print(f"Bootstrap {a.bootstrap}x (berpasangan, berstrata per fold, seed {SEED}): "
          f"CI 95% delta [{ci[0]:+.5f}, {ci[1]:+.5f}], P(delta>0) = {p_pos:.3f}")

    known = [n for n in (nb, nc) if n is not None]
    min_seed = min(known) if len(known) == 2 else None
    decision, notes = decide(delta, p_pos, n_up, len(auc_b), min_seed, a)
    if a.cand == a.base:
        notes.append("Base = cand: ini cek penyelarasan. Bandingkan mean-fold AUC di atas dengan log.csv.")

    print()
    print(f"Aturan   : DITERIMA jika P(delta>0) >= {a.p_accept} dan delta >= {a.min_delta:+.4f}; "
          f"KANDIDAT jika P >= {a.p_candidate}; selain itu DITOLAK; OOF >= {a.min_seeds} seed")
    print(f"Hasil    : delta {delta:+.5f}, P {p_pos:.3f}, naik {n_up}/{len(auc_b)} fold (informasi)")
    print(f"KEPUTUSAN: {decision}")
    for n in notes:
        print(f"CATATAN  : {n}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({
                "base": a.base, "cand": a.cand, "base_files": fb, "cand_files": fc,
                "n_seed_base": nb, "n_seed_cand": nc, "fold_source": fold_src,
                "auc_base": auc_b.tolist(), "auc_cand": auc_c.tolist(),
                "ks_base": ks_b.tolist(), "ks_cand": ks_c.tolist(),
                "mean_fold_auc_base": float(auc_b.mean()), "mean_fold_auc_cand": float(auc_c.mean()),
                "delta": delta, "folds_up": n_up, "spearman": float(rho),
                "bootstrap_n": a.bootstrap, "bootstrap_ci": [float(ci[0]), float(ci[1])], "p_delta_pos": p_pos,
                "decision": decision, "notes": notes,
            }, fh, indent=2)


if __name__ == "__main__":
    main()
