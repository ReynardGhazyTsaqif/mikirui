"""Runner eksperimen.

Contoh:
    python run.py E00                      # submission konstan (deteksi metrik)
    python run.py E01 --submit             # LightGBM dev + tulis submission
    python run.py E03 --seeds 42 43 44     # ukur noise antar seed
    python run.py E04                      # adversarial validation train vs test
    python run.py E03 --rebuild            # bangun ulang cache fitur
    python run.py E62                      # ablation per grup dari set terbaik
    python run.py E70                      # buat featset filter redundan + CV
    python run.py E71                      # null importance -> featset sel_full & sel_b2 + CV
    python run.py E75 --base E31           # analisis segmen OOF
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from src import selection as sel
from src.models import run_model_cv
from src.config import LGB_DEV, LOG_PATH, N_FOLDS, ROOT, SEED
from src.features.common import cached
from src.features.registry import GROUPS
from src.io_utils import load_apps
from src.train import (Timer, fold_ks, ks_score, log_experiment, make_folds, run_lgb_cv,
                       save_arrays, write_submission)

B0 = ["app_raw_v1", "prof_raw_v1"]
NAIVE = ["pa_naive_v1", "rp_naive_v1", "bu_naive_v1", "bp_naive_v1", "ab_naive_v1", "ec_naive_v1"]

# groups = set grup eksplisit; add = tambahkan ke set eksperimen acuan (base, default "BEST" =
# eksperimen diterima dengan mean-fold AUC tertinggi); cols = featset di experiments/featsets/.
EXPERIMENTS = {
    "E01": dict(groups=B0, desc="LightGBM dev, kolom mentah aplikasi + profil (B0)"),
    "E02": dict(groups=B0 + ["app_ratios_v1"], desc="+ rasio aplikasi/profil dan waktu"),
    "E03": dict(groups=B0 + ["app_ratios_v1"] + NAIVE, desc="+ agregat naif semua tabel (B1)"),
    "E10": dict(add=["rp_core_v1"], desc="rp_ DPD dan pembayaran all-time"),
    "E11": dict(add=["rp_window_v1"], desc="rp_ jendela 90/180/365 hari + recency"),
    "E12": dict(add=["rp_loan_v1"], desc="rp_ dua tingkat, ever90, pinjaman terakhir"),
    "E13": dict(add=["rp_trend_v1"], desc="rp_ tren, DPD berbobot waktu, streak"),
    "E50": dict(add=["ec_window_v1"], desc="ec_ RFM jendela 3-365 hari, recency, gap antarorder"),
    "E51": dict(add=["ec_mix_v1"], desc="ec_ pergeseran metode bayar & kategori, pola waktu"),
    "E20": dict(add=["bu_extra_v1"], desc="bu_ utilisasi total, sebaran skor, laporan segar"),
    "E21": dict(add=["bp_trend_v1"], desc="bp_ delta & tren skor, pertumbuhan utang"),
    "E30": dict(add=["pa_window_v1"], desc="pa_ jendela, recency, kecepatan, pinjaman baru"),
    "E31": dict(add=["pa_burden_v1"], desc="pa_ cicilan aktif, nominal & bunga vs sebelumnya"),
    "E40": dict(add=["ab_window_v1"], desc="ab_ jendela, tren, volatilitas"),
    "E41": dict(add=["ab_income_v1"], desc="ab_ verifikasi pendapatan, bantalan likuiditas"),
    "E60": dict(add=["x_cross_v1"], desc="x_ total PTI, utang/pendapatan, tiga sumber income, stres"),
    "E61": dict(add=["x_rate_resid_v1"], desc="x_ residual bunga (regresi tanpa target)"),
    # Fase 3
    "E70": dict(base="E31", cols="e70_b2", min_delta=-0.0003,
                desc="B2 tanpa kolom konstan/duplikat/korelasi > 0,995"),
    "E71a": dict(groups=list(GROUPS), desc="semua grup tanpa seleksi (pembanding E71)"),
    "E71b": dict(groups=list(GROUPS), cols="sel_full",
                 desc="null importance dari semua grup (menyelamatkan fitur grup ditolak)"),
    "E71c": dict(base="E31", cols="sel_b2", min_delta=-0.0003,
                 desc="null importance, dibatasi kolom B2"),
    "E72a": dict(base="B3", params={"is_unbalance": True}, desc="is_unbalance=True"),
    "E72b": dict(base="B3", params={"scale_pos_weight": 2.5}, desc="scale_pos_weight=2,5"),
    "E73": dict(base="B3", monotone=True, min_delta=-0.0003,
                desc="monotone constraints skor biro, DPD, bunga, recency e-commerce"),
    "E74a": dict(base="B3", cat_mode="onehot", desc="kategori one-hot"),
    "E74b": dict(base="B3", cat_mode="ordinal", desc="kategori kode ordinal"),
    "E74c": dict(base="B3", cat_mode="target", desc="kategori target encoding di dalam fold"),
    # Fase 4: diversitas model di atas B3. Dinilai dari kenaikan blend dengan E71b, bukan skor
    # tunggal (diterima = "kandidat" jika blend rank naik >= 0,0003 mean-fold AUC).
    "E80": dict(base="B3", model="catboost", blend_with="E71b", desc="CatBoost dev pada B3"),
    "E81": dict(base="B3", model="xgboost", blend_with="E71b", desc="XGBoost hist dev pada B3"),
    "E82": dict(base="B3", model="logreg", blend_with="E71b",
                desc="Logistic Regression + binning kuantil 16, L2 C=0,05"),
    "E83": dict(base="B3", model="mlp", blend_with="E71b",
                desc="MLP sklearn 256-128, QuantileTransformer normal"),
    "E84": dict(base="B3", model="extratrees", blend_with="E71b",
                desc="ExtraTrees 500 pohon, min_samples_leaf 30"),
    "E87": dict(base="B3", params={"extra_trees": True}, blend_with="E71b",
                desc="LightGBM extra_trees=True pada B3"),
}
B3_PATH = ROOT / "experiments" / "B3.json"


def accepted_rows(log):
    """Baris dengan diterima == True. Kolom berisi True/False/'kandidat' (Fase 4), sehingga pandas
    membacanya sebagai teks; bandingkan lewat string agar tidak diam-diam kosong."""
    return log[log["diterima"].astype(str) == "True"].copy()


def best_accepted():
    """Baris log eksperimen diterima dengan rata-rata AUC per fold tertinggi."""
    log = pd.read_csv(LOG_PATH)
    acc = accepted_rows(log)
    acc["mean_fold"] = acc[[f"auc_f{i}" for i in range(1, N_FOLDS + 1)]].mean(axis=1)
    return acc.loc[acc["mean_fold"].idxmax()]


def parse_feature_set(fs):
    """'g1+g2|cols=nama' -> ([g1, g2], 'nama'); tanpa '|cols=' -> cols None."""
    groups, _, cols = fs.partition("|cols=")
    return groups.split("+"), (cols or None)


def base_row(base):
    if base == "BEST":
        return best_accepted()
    if base == "B3":  # set acuan pilihan manual setelah E70/E71 (experiments/B3.json)
        with open(B3_PATH) as f:
            base = json.load(f)["exp_id"]
    log = pd.read_csv(LOG_PATH)
    return log[log["exp_id"] == base].iloc[-1]


def resolve(cfg):
    """Return (groups, cols, add): grup yang dimuat, nama featset (atau None), grup tambahan."""
    if "groups" in cfg:
        return list(cfg["groups"]), cfg.get("cols"), []
    groups, cols = parse_feature_set(base_row(cfg.get("base", "BEST"))["feature_set"])
    add = [g for g in cfg.get("add", []) if g not in groups]
    return groups + add, cfg.get("cols", cols), add


def build_matrix(groups, apps, rebuild=False):
    parts = []
    for g in groups:
        with Timer() as t:
            parts.append(cached(g, GROUPS[g], apps, rebuild=rebuild))
        print(f"  fitur {g:<16} {parts[-1].shape[1]:>4} kolom  ({t.minutes:.1f} mnt)", flush=True)
    X = pd.concat(parts, axis=1)
    assert X.columns.is_unique, "nama kolom duplikat antar grup"
    return X[sorted(X.columns)]


def terima(oof_baru, fold_baru, min_delta=0.0005):
    """Aturan terima dari planning: naik >= min_delta DAN membaik di >= 4 dari 5 fold.

    Delta dihitung dari rata-rata AUC per fold, bukan pooled OOF AUC. Pooled OOF bisa naik hanya
    karena kalibrasi antar fold lebih seragam (terlihat di E02: pooled +0,0021, per fold turun).
    Untuk eksperimen penyederhanaan (min_delta < 0), syarat jumlah fold yang membaik tidak dipakai.
    """
    if not LOG_PATH.exists():
        return True, "eksperimen pertama, jadi acuan"
    log = pd.read_csv(LOG_PATH)
    acc = accepted_rows(log)
    if acc.empty:
        return True, "belum ada acuan"
    fcols = [f"auc_f{i}" for i in range(1, N_FOLDS + 1)]
    acc["mean_fold"] = acc[fcols].mean(axis=1)
    best = acc.loc[acc["mean_fold"].idxmax()]
    naik = sum(b > a for b, a in zip(fold_baru, best[fcols]))
    delta = float(np.mean(fold_baru)) - best["mean_fold"]
    ok = delta >= min_delta and (naik >= 4 or min_delta < 0)
    return ok, (f"vs {best['exp_id']}: delta mean-fold {delta:+.5f} "
                f"(pooled OOF {oof_baru - best['oof_auc']:+.5f}), membaik di {naik}/{N_FOLDS} fold"
                f", batas {min_delta:+.4f}")


def mean_fold_auc(y, p, fold):
    return float(np.mean([roc_auc_score(y[fold == k], p[fold == k]) for k in range(N_FOLDS)]))


def blend_check(ref_exp, oof, y, fold, min_gain=0.0003):
    """Apakah OOF baru menaikkan blend rank dengan OOF ref_exp? Bobot dicari di grid 0..1 (0,05).

    Diagnostik cepat Fase 4; bobot final dicari di Fase 6 dengan nested CV.
    """
    ref = np.load(ROOT / "oof" / f"{ref_exp}.npy")
    r_ref, r_new = rankdata(ref) / len(ref), rankdata(oof) / len(oof)
    base = mean_fold_auc(y, ref, fold)
    grid = [(w, mean_fold_auc(y, (1 - w) * r_ref + w * r_new, fold))
            for w in np.round(np.arange(0, 1.0001, 0.05), 2)]
    w_best, auc_best = max(grid, key=lambda t: t[1])
    auc_half = dict(grid)[0.5]
    blend = (1 - w_best) * r_ref + w_best * r_new
    rho = spearmanr(ref, oof).statistic
    ok = (auc_best - base) >= min_gain and w_best > 0
    return ok, (f"vs {ref_exp}: Spearman {rho:.3f} | blend 50/50 {auc_half - base:+.5f} | "
                f"bobot terbaik {w_best:.2f} -> {auc_best - base:+.5f} mean-fold AUC, "
                f"KS {ks_score(y, ref):.4f} -> {ks_score(y, blend):.4f}")


def run_experiment(exp_id, seeds, submit, rebuild, pembuat):
    cfg = EXPERIMENTS[exp_id]
    groups, cols, add = resolve(cfg)
    apps = load_apps()
    train_mask = apps["is_test"].eq(0).values
    folds = make_folds(apps[train_mask])
    X = build_matrix(groups, apps, rebuild)
    if cols:
        keep = set(sel.load_featset(cols))
        for g in add:  # grup baru di atas featset terseleksi ikut masuk utuh
            keep |= set(cached(g, GROUPS[g], apps).columns)
        X = X[[c for c in X.columns if c in keep]]
    model = cfg.get("model", "lgb")
    params_base = {**LGB_DEV, **cfg.get("params", {})} if model == "lgb" else cfg.get("params", {})
    prep = None
    mode = cfg.get("cat_mode", "native")
    if mode in ("onehot", "ordinal"):
        X = sel.encode_static(X, mode)
    elif mode == "target":
        prep = sel.target_encode_prep
    if cfg.get("monotone"):
        mono = sel.monotone_vector(X.columns)
        params_base["monotone_constraints"] = mono
        print(f"  monotone constraints pada {sum(m != 0 for m in mono)} kolom")
    X_tr, X_te = X[train_mask], X[~train_mask]
    y = apps.loc[train_mask, "target"].astype(int).values
    fold = folds["fold"].values
    print(f"{exp_id}: {X.shape[1]} fitur, train {X_tr.shape}, test {X_te.shape}", flush=True)

    results = {}
    for s in seeds:
        with Timer() as t:
            if model == "lgb":
                oof, pred, aucs, iters, imp = run_lgb_cv({**params_base, "seed": s}, X_tr, y,
                                                         fold, X_te, prep=prep)
            else:
                oof, pred, aucs, iters, imp = run_model_cv(model, params_base, X_tr, y, fold,
                                                           X_te, s)
        auc, ks, kss = roc_auc_score(y, oof), ks_score(y, oof), fold_ks(y, oof, fold)
        results[s] = dict(oof=oof, pred=pred, aucs=aucs, auc=auc, ks=ks, kss=kss,
                          minutes=t.minutes, imp=imp)
        print(f"  seed {s}: OOF AUC {auc:.5f} | KS {ks:.5f} (mean-fold {np.mean(kss):.5f}) | "
              f"fold AUC {np.round(aucs, 4).tolist()} | iter {iters} | {t.minutes:.1f} mnt",
              flush=True)

    main = results[seeds[0]]
    (ROOT / "experiments").mkdir(exist_ok=True)
    if len(seeds) > 1:
        auc_s = [r["auc"] for r in results.values()]
        noise = dict(exp_id=exp_id, seeds=seeds, oof_auc=auc_s, std=float(np.std(auc_s, ddof=1)),
                     range=float(max(auc_s) - min(auc_s)))
        print(f"  noise antar seed: std {noise['std']:.5f}, rentang {noise['range']:.5f}")
        with open(ROOT / "experiments" / f"seed_noise_{exp_id}.json", "w") as f:
            json.dump(noise, f, indent=2)

    if cfg.get("blend_with"):
        ok, alasan = blend_check(cfg["blend_with"], main["oof"], y, fold)
        print(f"  keputusan: {'KANDIDAT ENSEMBLE' if ok else 'BUKAN KANDIDAT'} ({alasan})")
        ok = "kandidat" if ok else False
    else:
        ok, alasan = terima(main["auc"], main["aucs"], cfg.get("min_delta", 0.0005))
        print(f"  keputusan: {'DITERIMA' if ok else 'DITOLAK'} ({alasan})")

    save_arrays(exp_id, main["oof"], main["pred"])
    sub_name, sub_md5 = "", ""
    if submit:
        path, sub_md5 = write_submission(apps.loc[~train_mask, "application_id"], main["pred"],
                                         f"{exp_id}_lgb_oof{main['auc']:.5f}")
        sub_name = path.name
        print(f"  submission: {path}  md5 {sub_md5}")
    main["imp"].to_csv(ROOT / "experiments" / f"importance_{exp_id}.csv", header=["gain"])
    print("  top 15 fitur (gain):\n" + main["imp"].head(15).round(0).to_string())

    fs = "+".join(groups) + (f"|cols={cols}" if cols else "")
    log_params = {k: v for k, v in params_base.items() if k != "monotone_constraints"}
    log_params["monotone"] = bool(cfg.get("monotone"))
    log_params["cat_mode"] = mode
    log_experiment(exp_id, cfg["desc"], fs, X.shape[1], f"{model}_dev",
                   {**log_params, "seed": seeds[0]}, main["aucs"], main["auc"], main["minutes"],
                   ok, main["ks"], main["kss"], pembuat=pembuat, submission_file=sub_name,
                   md5_hash=sub_md5)


def run_e00():
    apps = load_apps()
    rate = apps.loc[apps["is_test"] == 0, "target"].mean()
    test_ids = apps.loc[apps["is_test"] == 1, "application_id"]
    path, h = write_submission(test_ids, np.full(len(test_ids), round(rate, 4)), "E00_constant")
    print(f"E00: konstanta {rate:.4f} -> {path}  md5 {h}")
    print("Tebakan metrik dari public score: 0.5 = AUC, 0 = Gini, ~0.397 = LogLoss, ~0.117 = Brier")


def run_e04(rebuild):
    """Adversarial validation: bisakah model membedakan baris train vs test?"""
    apps = load_apps()
    X = build_matrix(EXPERIMENTS["E03"]["groups"], apps, rebuild)
    y = apps["is_test"].values
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, N_FOLDS, len(y))
    params = {**LGB_DEV, "learning_rate": 0.1}
    oof = np.zeros(len(y))
    imp = pd.Series(0.0, index=X.columns)
    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300)
        oof[va] = m.predict(X[va])
        imp += pd.Series(m.feature_importance("gain"), index=X.columns)
    auc = roc_auc_score(y, oof)
    verdict = "aman" if auc < 0.55 else "PERIKSA fitur teratas di bawah"
    print(f"E04 adversarial AUC: {auc:.4f} -> {verdict}")
    print("Fitur paling membedakan train/test:\n"
          + imp.sort_values(ascending=False).head(10).round(0).to_string())
    (ROOT / "experiments").mkdir(exist_ok=True)
    with open(ROOT / "experiments" / "adversarial_E04.json", "w") as f:
        json.dump(dict(auc=auc, top=imp.sort_values(ascending=False).head(20).to_dict()), f,
                  indent=2)


def run_ablation(tol=-0.0001):
    """E62: hapus satu grup tiap run dari set terbaik. Grup yang penghapusannya tidak merugikan
    (delta mean-fold AUC >= tol) menjadi kandidat dibuang. Hasil: experiments/ablation_E62.csv dan
    usulan set B2 di experiments/B2_groups.json (dikonfirmasi dengan run terpisah)."""
    best = best_accepted()
    groups, _ = parse_feature_set(best["feature_set"])
    base_auc = np.mean([best[f"auc_f{i}"] for i in range(1, N_FOLDS + 1)])
    apps = load_apps()
    train_mask = apps["is_test"].eq(0).values
    y = apps.loc[train_mask, "target"].astype(int).values
    fold = make_folds(apps[train_mask])["fold"].values
    full = build_matrix(groups, apps)
    rows = []
    for g in groups:
        if g in B0:  # kolom inti aplikasi/profil selalu dipertahankan
            continue
        drop = cached(g, GROUPS[g], apps).columns
        X = full.drop(columns=drop)
        with Timer() as t:
            oof, _, aucs, _, _ = run_lgb_cv(LGB_DEV, X[train_mask], y, fold, X[~train_mask])
        d = float(np.mean(aucs)) - base_auc
        rows.append(dict(group=g, n_cols=len(drop), mean_fold_auc=np.mean(aucs), delta=d,
                         oof_ks=ks_score(y, oof)))
        print(f"  tanpa {g:<18} ({len(drop):>3} kol): delta mean-fold AUC {d:+.5f}, "
              f"KS {rows[-1]['oof_ks']:.4f}  ({t.minutes:.1f} mnt)", flush=True)
    res = pd.DataFrame(rows).sort_values("delta", ascending=False)
    res.to_csv(ROOT / "experiments" / "ablation_E62.csv", index=False)
    dropped = res.loc[res["delta"] >= tol, "group"].tolist()
    b2 = [g for g in groups if g not in dropped]
    with open(ROOT / "experiments" / "B2_groups.json", "w") as f:
        json.dump(dict(base=best["exp_id"], dropped=dropped, groups=b2), f, indent=2)
    print(f"Kandidat dibuang (delta >= {tol}): {dropped}")


def run_e70(pembuat):
    """Buat featset e70_b2 (filter redundan atas kolom B2/E31) lalu jalankan CV E70."""
    groups, _ = parse_feature_set(base_row("E31")["feature_set"])
    X = build_matrix(groups, load_apps())
    imp = pd.read_csv(ROOT / "experiments" / "importance_E31.csv", index_col=0)["gain"]
    keep, report = sel.redundancy_filter(X, imp)
    for k, v in report.items():
        print(f"  {k}: {len(v)} kolom {v[:8]}{' ...' if len(v) > 8 else ''}")
    sel.save_featset("e70_b2", keep, dict(source="E31", dropped=report))
    print(f"  featset e70_b2: {len(keep)} dari {X.shape[1]} kolom", flush=True)
    run_experiment("E70", [SEED], False, False, pembuat)


def run_e71(pembuat, n_null=40):
    """Null importance pada SEMUA grup -> featset sel_full; irisan dengan kolom B2 -> sel_b2."""
    apps = load_apps()
    train_mask = apps["is_test"].eq(0).values
    X = build_matrix(list(GROUPS), apps)
    y = apps.loc[train_mask, "target"].astype(int).values
    print(f"  null importance pada {X.shape[1]} kolom, {n_null} run acak", flush=True)
    with Timer() as t:
        res = sel.null_importance(X[train_mask], y, n_null=n_null,
                                  log=lambda m: print(m, flush=True))
    res.to_csv(ROOT / "experiments" / "null_importance_E71.csv")
    keep = res.index[res["actual"] > res["null_q75"]].tolist()
    b2_groups, _ = parse_feature_set(base_row("E31")["feature_set"])
    b2_cols = set(build_matrix(b2_groups, apps).columns)
    sel.save_featset("sel_full", keep, dict(rule="actual gain > null q75", n_null=n_null))
    sel.save_featset("sel_b2", [c for c in keep if c in b2_cols],
                     dict(source="kolom E31 yang juga lolos sel_full"))
    rescued = sorted(set(keep) - b2_cols)
    print(f"  selesai ({t.minutes:.1f} mnt): sel_full {len(keep)}/{X.shape[1]} kolom, "
          f"sel_b2 {len(set(keep) & b2_cols)}/{len(b2_cols)} kolom")
    print(f"  fitur dari grup ditolak yang lolos: {len(rescued)}, contoh {rescued[:10]}")
    for e in ["E71a", "E71b", "E71c"]:
        print(f"=== {e}", flush=True)
        run_experiment(e, [SEED], False, False, pembuat)


def run_e75(base):
    """Analisis segmen OOF: di segmen mana model paling lemah?"""
    apps = load_apps()
    is_tr = apps["is_test"].eq(0).values
    tr = apps[is_tr].reset_index(drop=True)
    y = tr["target"].astype(int).values
    oof = np.load(ROOT / "oof" / f"{base}.npy")
    feats = build_matrix(["prof_raw_v1", "rp_naive_v1", "pa_naive_v1", "bu_naive_v1",
                          "ab_naive_v1", "ec_naive_v1"], apps)
    f = feats[is_tr].reset_index(drop=True)
    seg = pd.DataFrame({
        "punya_riwayat_cicilan": f["rp_n_installments"].notna(),
        "punya_aplikasi_lama": f["pa_count"].notna(),
        "punya_data_rekening": f["ab_n_months"].notna(),
        "n_provider_biro": f["bu_n_providers"],
        "employment_type": f["prof_employment_type"],
        "region": f["prof_region"],
        "loan_purpose": tr["loan_purpose"],
        "tier_bunga": tr["annual_interest_rate"],
        "recency_ecommerce": pd.cut(f["ec_days_since_last_order"], [0, 2, 7, 30, 1e4],
                                    labels=["<=2h", "2-7h", "7-30h", ">30h"]),
        "usia": pd.cut(f["prof_age"], [0, 30, 40, 50, 100],
                       labels=["<30", "30-40", "40-50", "50+"]),
    })
    rep = sel.segment_report(seg, y, oof)
    rep.to_csv(ROOT / "experiments" / f"segments_{base}.csv", index=False)
    print(f"OOF {base}: AUC {roc_auc_score(y, oof):.4f}, KS {ks_score(y, oof):.4f}")
    print(rep.round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp_id")
    p.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    p.add_argument("--submit", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--pembuat", default="")
    p.add_argument("--base", default="E31")
    p.add_argument("--other", default=None)
    a = p.parse_args()
    if a.exp_id == "BLEND":  # python run.py BLEND --base E71b --other E71c
        apps_ = load_apps()
        y_ = apps_.loc[apps_["is_test"] == 0, "target"].astype(int).values
        fold_ = make_folds(apps_[apps_["is_test"] == 0])["fold"].values
        ok_, msg_ = blend_check(a.base, np.load(ROOT / "oof" / f"{a.other}.npy"), y_, fold_)
        print(f"{a.other} {'KANDIDAT' if ok_ else 'bukan kandidat'}: {msg_}")
    elif a.exp_id == "E00":
        run_e00()
    elif a.exp_id == "E04":
        run_e04(a.rebuild)
    elif a.exp_id == "E62":
        run_ablation()
    elif a.exp_id == "E70":
        run_e70(a.pembuat)
    elif a.exp_id == "E71":
        run_e71(a.pembuat)
    elif a.exp_id == "E75":
        run_e75(a.base)
    else:
        run_experiment(a.exp_id, a.seeds, a.submit, a.rebuild, a.pembuat)
