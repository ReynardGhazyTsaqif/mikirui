"""Fase 3: seleksi fitur dan variasi pelatihan.

Set kolom hasil seleksi disimpan sebagai experiments/featsets/<nama>.json (list kolom terurut)
sehingga eksperimen di run.py bisa merujuknya lewat `cols="<nama>"`.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

from .config import CAT_COLS, LGB_DEV, ROOT, SEED
from .train import ks_score

FEATSET_DIR = ROOT / "experiments" / "featsets"


def save_featset(name, cols, meta=None):
    FEATSET_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATSET_DIR / f"{name}.json", "w") as f:
        json.dump(dict(columns=sorted(cols), **(meta or {})), f, indent=2)


def load_featset(name):
    with open(FEATSET_DIR / f"{name}.json") as f:
        return json.load(f)["columns"]


# ---------------------------------------------------------------- E70: filter redundan
def redundancy_filter(X, importance, corr_thr=0.995, sample=30000):
    """Buang kolom konstan, duplikat identik, NaN > 99%, dan satu dari pasangan |rho| > corr_thr.

    Dari pasangan berkorelasi, yang dibuang adalah yang gain importance-nya lebih kecil.
    """
    report = {}
    nan_share = X.isna().mean()
    report["nan_gt_99"] = sorted(nan_share[nan_share > 0.99].index)
    nunique = X.nunique(dropna=True)
    report["constant"] = sorted(nunique[nunique <= 1].index)
    keep = [c for c in X.columns if c not in set(report["nan_gt_99"]) | set(report["constant"])]

    num = [c for c in keep if c not in CAT_COLS and not str(X[c].dtype) == "category"]
    seen, dup = {}, []
    for c in sorted(num, key=lambda c: -importance.get(c, 0)):
        h = pd.util.hash_pandas_object(X[c], index=False).sum()
        if h in seen and X[c].equals(X[seen[h]]):
            dup.append(c)
        else:
            seen[h] = c
    report["duplicate"] = sorted(dup)
    num = [c for c in num if c not in set(dup)]

    S = X[num].sample(min(sample, len(X)), random_state=SEED).astype("float64")
    corr = S.corr().abs()
    order = sorted(num, key=lambda c: -importance.get(c, 0))  # paling penting diproses dulu
    dropped_corr = []
    kept = []
    for c in order:
        if any(corr.at[c, k] > corr_thr for k in kept):
            dropped_corr.append(c)
        else:
            kept.append(c)
    report["high_corr"] = sorted(dropped_corr)
    drop = set(report["nan_gt_99"]) | set(report["constant"]) | set(dup) | set(dropped_corr)
    return [c for c in X.columns if c not in drop], report


# ---------------------------------------------------------------- E71: null importance
def null_importance(X, y, n_actual=3, n_null=40, rounds=250, log=print):
    """Gain importance asli (rata-rata n_actual seed) vs distribusi gain dengan target diacak.

    Return DataFrame: actual, null_q75, null_max, score = log((1e-10 + actual) / (1 + null_q75)).
    """
    params = {**LGB_DEV}
    ds = lgb.Dataset(X, y, free_raw_data=False)

    def gains(target, seed):
        ds.set_label(target)
        m = lgb.train({**params, "seed": seed}, ds, rounds)
        return pd.Series(m.feature_importance("gain"), index=X.columns)

    actual = pd.concat([gains(y, SEED + i) for i in range(n_actual)], axis=1).mean(axis=1)
    log(f"  importance asli selesai ({n_actual} seed)")
    rng = np.random.RandomState(SEED)
    nulls = []
    for i in range(n_null):
        nulls.append(gains(rng.permutation(y), SEED + 100 + i))
        if (i + 1) % 10 == 0:
            log(f"  null importance {i + 1}/{n_null}")
    nulls = pd.concat(nulls, axis=1)
    ds.set_label(y)
    res = pd.DataFrame({"actual": actual, "null_q75": nulls.quantile(0.75, axis=1),
                        "null_max": nulls.max(axis=1)})
    res["score"] = np.log((1e-10 + res["actual"]) / (1 + res["null_q75"]))
    return res.sort_values("score", ascending=False)


# ---------------------------------------------------------------- E73: monotone constraints
MONOTONE = {
    # +1: nilai makin besar -> risiko makin besar; -1: sebaliknya. Arah diverifikasi di EDA.
    "app_annual_interest_rate": 1,
    "bu_score_mean": -1, "bu_score_min": -1, "bu_score_max": -1,
    "bu_score_A": -1, "bu_score_B": -1, "bu_score_C": -1,
    "bu_delinquent_accounts_last_12m_max": 1,
    "rp_max_days_past_due_max": 1, "rp_max_days_past_due_mean": 1,
    "rp_days_past_due_max": 1, "rp_n_dpd30": 1, "rp_n_dpd90": 1,
    "rp_share_dpd30": 1, "rp_share_dpd90": 1,
    "ec_days_since_last_order": -1,
}


def monotone_vector(columns):
    return [MONOTONE.get(c, 0) for c in columns]


# ---------------------------------------------------------------- E74: encoding kategori
def cat_columns(X):
    return [c for c in X.columns if str(X[c].dtype) == "category"]


def encode_static(X, mode):
    """Encoding tanpa target (boleh di luar fold): onehot atau ordinal (kode integer)."""
    cats = cat_columns(X)
    if mode == "onehot":
        return pd.get_dummies(X, columns=cats, dtype="int8")
    if mode == "ordinal":
        X = X.copy()
        for c in cats:
            X[c] = X[c].cat.codes.astype("int16")
        return X
    raise ValueError(mode)


def target_encode_prep(X_tr, y_tr, X_va, X_te):
    """Target encoding di dalam fold (sklearn TargetEncoder memakai cross-fitting internal)."""
    cats = cat_columns(X_tr)
    enc = TargetEncoder(target_type="binary", random_state=SEED)
    out = []
    tr_enc = enc.fit_transform(X_tr[cats].astype(str), y_tr)
    for X, vals in [(X_tr, tr_enc), (X_va, enc.transform(X_va[cats].astype(str))),
                    (X_te, enc.transform(X_te[cats].astype(str)))]:
        X = X.copy()
        X[cats] = vals
        out.append(X)
    return tuple(out)


# ---------------------------------------------------------------- E75: analisis segmen
def segment_report(df, y, oof):
    """AUC/KS OOF per segmen. df: DataFrame train ber-index sama dengan y/oof berisi kolom segmen."""
    rows = []
    for col in df.columns:
        for val, idx in df.groupby(col, observed=True).indices.items():
            yy, pp = y[idx], oof[idx]
            if len(idx) < 300 or yy.min() == yy.max():
                continue
            rows.append(dict(segment=col, value=str(val), n=len(idx), target_rate=yy.mean(),
                             auc=roc_auc_score(yy, pp), ks=ks_score(yy, pp)))
    return pd.DataFrame(rows)
