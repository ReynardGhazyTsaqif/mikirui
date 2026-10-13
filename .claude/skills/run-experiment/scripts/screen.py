"""Screen cepat fitur kandidat di atas OOF terbaik (Riset bagian 6).

Dua uji:
  1. Residual boosting: model kecil (7 daun) hanya dengan fitur kandidat,
     init_score = logit(OOF acuan), 5 fold tetap. Ukur delta AUC per fold.
  2. Korelasi residual: kalibrasi ulang OOF (logistik pada logit), lalu
     korelasi Pearson rank(fitur) dengan residual mentah (y - p). Noise sd ~ 1/sqrt(n).

Lolos jika delta mean-fold AUC >= +0,0005 ATAU ada kolom dengan |z| > 3.
Screen tidak nested dan sedikit optimis; keputusan akhir tetap dari CV penuh.

Contoh:
    python .claude/skills/run-experiment/scripts/screen.py --base-oof T_lgb_b3_4f3c7f \
        --features features/ec_ordermodel_v1.parquet --prefix ec_ordermodel_
    python .claude/skills/run-experiment/scripts/screen.py --selftest --engine sklearn
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_oof import load_target_and_folds, resolve_oof  # noqa: E402

SEED = 42
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def read_table(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "application_id" in df.columns:
        df = df.set_index("application_id")
    elif df.index.name != "application_id":
        sys.exit("ERROR: file fitur butuh kolom atau index 'application_id'")
    return df


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in X.columns:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s):
            out[c] = s.astype("category").cat.codes.replace(-1, np.nan).astype(float)
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(float)
        else:
            out[c] = pd.to_numeric(s, errors="coerce").astype(float)
    return pd.DataFrame(out, index=X.index)


# ---------------------------------------------------------------- boosting engines
def boost_lgb(Xtr, ytr, itr, Xva, yva, iva, seed):
    import lightgbm as lgb
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=7,
                  min_data_in_leaf=200, lambda_l2=1.0, feature_fraction=1.0,
                  seed=seed, deterministic=True, force_row_wise=True, num_threads=4, verbose=-1)
    dtr = lgb.Dataset(Xtr, ytr, init_score=itr)
    dva = lgb.Dataset(Xva, yva, init_score=iva, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return iva + m.predict(Xva, raw_score=True, num_iteration=m.best_iteration), m.best_iteration


def boost_sklearn(Xtr, ytr, itr, Xva, yva, iva, seed, lr=0.05, max_rounds=600, patience=60):
    """Newton boosting logloss dengan pohon 7 daun. Cadangan bila LightGBM tidak ada."""
    from sklearn.tree import DecisionTreeRegressor
    Ftr, Fva = itr.astype(float).copy(), iva.astype(float).copy()
    best_auc, best_F, best_it, since = roc_auc_score(yva, Fva), Fva.copy(), 0, 0
    for it in range(1, max_rounds + 1):
        p = sigmoid(Ftr)
        h = np.maximum(p * (1 - p), 1e-6)
        tree = DecisionTreeRegressor(max_leaf_nodes=7, min_samples_leaf=200, random_state=seed + it)
        tree.fit(Xtr, (ytr - p) / h, sample_weight=h)
        Ftr += lr * tree.predict(Xtr)
        Fva += lr * tree.predict(Xva)
        auc = roc_auc_score(yva, Fva)
        if auc > best_auc + 1e-7:
            best_auc, best_F, best_it, since = auc, Fva.copy(), it, 0
        else:
            since += 1
            if since >= patience:
                break
    return best_F, best_it


ENGINES = {"lgb": boost_lgb, "sklearn": boost_sklearn}


# ---------------------------------------------------------------- screen
def residual_corr(y, base, X: pd.DataFrame):
    z = logit(base).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    r = y - cal.predict_proba(z)[:, 1]
    rows = []
    for c in X.columns:
        x = X[c].to_numpy()
        ok = ~np.isnan(x)
        n_ok = int(ok.sum())
        if n_ok > 100 and np.nanstd(x) > 0:
            # Pearson rank(fitur) vs residual MENTAH (Riset bagian 6). Spearman me-rank residual juga,
            # sehingga rank residual kembali mengikuti y dan fitur kuat tampak "signifikan" palsu.
            rho = float(np.corrcoef(rankdata(x[ok]), r[ok])[0, 1])
            zval = rho * np.sqrt(n_ok)
            dec = pd.qcut(pd.Series(x[ok]).rank(method="first"), 10, labels=False)
            dmax = float(pd.Series(r[ok]).groupby(dec.to_numpy()).mean().abs().max())
        else:
            rho = zval = dmax = np.nan
        nan_rate = 1 - n_ok / len(x)
        z_nan = np.nan
        if 0.01 < nan_rate < 0.99:
            z_nan = float(np.corrcoef((~ok).astype(float), r)[0, 1]) * np.sqrt(len(x))
        rows.append(dict(kolom=c, nan=nan_rate, rho=rho, z=zval, z_nan=z_nan, resid_desil_max=dmax))
    return pd.DataFrame(rows)


def boosting_screen(y, base, fold, X: pd.DataFrame, engine, seed):
    init = logit(base)
    Xv = X.to_numpy(dtype=float)
    res = []
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        F, it = ENGINES[engine](Xv[tr], y[tr], init[tr], Xv[va], y[va], init[va], seed)
        a0 = roc_auc_score(y[va], base[va])
        res.append(dict(fold=int(k), auc_base=a0, auc_new=roc_auc_score(y[va], F),
                        delta=roc_auc_score(y[va], F) - a0, iter=it))
    return pd.DataFrame(res)


def run_screen(y, base, fold, X, engine, seed, min_delta=0.0005, z_thr=3.0, label=""):
    X = to_numeric(X)
    X = X[sorted(X.columns)]
    print(f"Screen {label}: {X.shape[1]} kolom, {len(y)} baris, engine={engine}")
    rc = residual_corr(y, base, X)
    rc["abs_z"] = rc[["z", "z_nan"]].abs().max(axis=1)
    rc = rc.sort_values("abs_z", ascending=False)
    with pd.option_context("display.width", 140, "display.max_rows", 40):
        print("\nKorelasi residual (|z| > 3 berarti sinyal di luar noise):")
        print(rc.head(25).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    bs = boosting_screen(y, base, fold, X, engine, seed)
    print("\nResidual boosting per fold:")
    print(bs.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    d = float(bs["delta"].mean())
    up = int((bs["delta"] > 0).sum())
    zmax = float(rc["abs_z"].max()) if len(rc) else 0.0
    passed = d >= min_delta or zmax > z_thr
    print(f"\nDelta mean-fold AUC {d:+.5f}, naik {up}/{len(bs)} fold, |z| maks {zmax:.2f}")
    print("SCREEN: " + ("LOLOS -> lanjut CV penuh 3 seed" if passed else "TIDAK LOLOS -> catat 'ditolak di screen'"))
    return dict(delta=d, folds_up=up, zmax=zmax, passed=passed)


def selftest(engine):
    rng = np.random.default_rng(SEED)
    n = 20000
    x0, x1, noise = rng.normal(size=(3, n))
    y = (rng.random(n) < sigmoid(-2 + 1.2 * x0 + 0.6 * x1)).astype(int)
    base = sigmoid(-2 + 1.2 * x0 + rng.normal(scale=0.05, size=n))  # acuan tanpa x1
    fold = np.arange(n) % 5
    rng.shuffle(fold)
    sig = run_screen(y, base, fold, pd.DataFrame({"t_signal": x1}), engine, SEED, label="(selftest sinyal)")
    print("\n" + "-" * 60)
    nul = run_screen(y, base, fold, pd.DataFrame({"t_noise": noise}), engine, SEED, label="(selftest noise)")
    ok = sig["passed"] and not nul["passed"]
    print("\nSELFTEST " + ("LULUS" if ok else "GAGAL") +
          f": sinyal delta {sig['delta']:+.4f} (harus lolos), noise delta {nul['delta']:+.5f} (harus tidak lolos)")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-oof", help="nama eksperimen acuan atau path .npy")
    ap.add_argument("--features", help="parquet/csv berisi application_id + fitur kandidat")
    ap.add_argument("--cols", nargs="*", help="kolom kandidat (default: semua)")
    ap.add_argument("--prefix", help="ambil kolom dengan prefix ini")
    ap.add_argument("--oof-dir", default="oof")
    ap.add_argument("--data-dir", default="Dataset")
    ap.add_argument("--folds", default="folds/folds.csv")
    ap.add_argument("--engine", choices=list(ENGINES), default="lgb")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true", help="uji dengan data sintetis lalu keluar")
    a = ap.parse_args()

    if a.engine == "lgb":
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            sys.exit("ERROR: lightgbm tidak terpasang. Pasang (python -m pip install lightgbm) atau pakai --engine sklearn.")
    if a.selftest:
        selftest(a.engine)
    if not (a.base_oof and a.features):
        ap.error("--base-oof dan --features wajib (kecuali --selftest)")

    y, fold, fold_src = load_target_and_folds(a.data_dir, a.folds)
    base, files, _ = resolve_oof(a.base_oof, a.oof_dir, None)
    if len(base) != len(y):
        sys.exit(f"ERROR: panjang OOF {len(base)} != train {len(y)}")
    train_ids = pd.read_csv(os.path.join(a.data_dir, "train_applications.csv"), usecols=["application_id"])
    F = read_table(a.features)
    if F.index.duplicated().any():
        sys.exit(f"ERROR: {int(F.index.duplicated().sum())} application_id duplikat di {a.features}")
    cols = a.cols or [c for c in F.columns if c != "target"]
    if a.prefix:
        cols = [c for c in cols if c.startswith(a.prefix)]
    bad = [c for c in cols if c.endswith("_id") or c == "target"]
    if bad:
        sys.exit(f"ERROR: kolom terlarang sebagai fitur: {bad}")
    if not cols:
        sys.exit("ERROR: tidak ada kolom kandidat")
    X = train_ids.join(F[cols], on="application_id").drop(columns="application_id")
    miss = X.isna().all(axis=1).mean()
    if miss > 0.5:
        print(f"PERINGATAN: {miss:.0%} baris train tidak punya nilai fitur sama sekali - cek join application_id")
    print(f"Fold: {fold_src}\nAcuan: {files}\n")
    run_screen(y, base, fold, X, a.engine, a.seed)


if __name__ == "__main__":
    main()
