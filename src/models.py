"""Fase 4: model selain LightGBM, semuanya memakai fold yang sama (folds/folds.csv).

Setiap fungsi fit_predict_<model>(X_tr, y_tr, X_va, y_va, X_te, seed, params) mengembalikan
(pred_va, pred_te, best_iter, importance_series_atau_None). run_model_cv membungkusnya menjadi
5-fold dengan keluaran yang sama dengan train.run_lgb_cv.
"""
import catboost as cb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, QuantileTransformer

from .config import N_FOLDS, N_THREADS


def _cats(X):
    return [c for c in X.columns if str(X[c].dtype) == "category"]


# ------------------------------------------------------------------------------- CatBoost
CB_DEV = dict(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3, eval_metric="AUC",
              od_type="Iter", od_wait=200, thread_count=N_THREADS, verbose=0,
              allow_writing_files=False)


def fit_predict_catboost(X_tr, y_tr, X_va, y_va, X_te, seed, params):
    cats = _cats(X_tr)

    def prep(X):  # CatBoost: fitur kategori harus string tanpa NaN
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return X

    m = cb.CatBoostClassifier(**{**CB_DEV, **params, "random_seed": seed}, cat_features=cats)
    m.fit(prep(X_tr), y_tr, eval_set=(prep(X_va), y_va), use_best_model=True)
    imp = pd.Series(m.get_feature_importance(), index=X_tr.columns)
    return (m.predict_proba(prep(X_va))[:, 1], m.predict_proba(prep(X_te))[:, 1],
            m.get_best_iteration(), imp)


# -------------------------------------------------------------------------------- XGBoost
XGB_DEV = dict(n_estimators=5000, learning_rate=0.05, max_depth=6, min_child_weight=5,
               subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist",
               enable_categorical=True, max_cat_to_onehot=8, eval_metric="auc",
               early_stopping_rounds=200, n_jobs=N_THREADS)


def fit_predict_xgboost(X_tr, y_tr, X_va, y_va, X_te, seed, params):
    m = xgb.XGBClassifier(**{**XGB_DEV, **params, "random_state": seed})
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    imp = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(
        X_tr.columns).fillna(0)
    it = m.best_iteration
    rng = (0, it + 1)
    return (m.predict_proba(X_va, iteration_range=rng)[:, 1],
            m.predict_proba(X_te, iteration_range=rng)[:, 1], it, imp)


# ------------------------------------------------------- Logistic Regression + binning
def _numeric_prep(num_cols, binning, n_bins, seed):
    """Imputasi median + indikator missing, lalu binning kuantil (LR) atau transform kuantil (MLP)."""
    impute = Pipeline([("imp", SimpleImputer(strategy="median"))])
    if binning:
        impute.steps.append(("bin", KBinsDiscretizer(n_bins=n_bins, encode="onehot",
                                                     strategy="quantile",
                                                     quantile_method="averaged_inverted_cdf")))
    else:
        impute.steps.append(("qt", QuantileTransformer(n_quantiles=200,
                                                       output_distribution="normal",
                                                       subsample=100_000, random_state=seed)))
    return FeatureUnion([("val", impute),
                         ("miss", MissingIndicator(features="missing-only",
                                                   error_on_new=False))])


def _design(X, binning, n_bins, seed):
    cats = _cats(X)
    num = [c for c in X.columns if c not in cats]
    return ColumnTransformer([
        ("num", _numeric_prep(num, binning, n_bins, seed), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ], sparse_threshold=0.3 if binning else 0.0)


def fit_predict_logreg(X_tr, y_tr, X_va, y_va, X_te, seed, params):
    p = dict(n_bins=16, C=0.05)
    p.update(params)
    pipe = Pipeline([("design", _design(X_tr, True, p["n_bins"], seed)),
                     ("lr", LogisticRegression(C=p["C"], max_iter=3000, random_state=seed))])
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1], 0, None


# --------------------------------------------------------------------------------- MLP
def fit_predict_mlp(X_tr, y_tr, X_va, y_va, X_te, seed, params):
    p = dict(hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3,
             batch_size=512, max_iter=200, early_stopping=True, validation_fraction=0.1,
             n_iter_no_change=10)
    p.update(params)
    pipe = Pipeline([("design", _design(X_tr, False, 0, seed)),
                     ("mlp", MLPClassifier(**p, random_state=seed))])
    pipe.fit(X_tr, y_tr)
    return (pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1],
            pipe.named_steps["mlp"].n_iter_, None)


# -------------------------------------------------------------------------- ExtraTrees
def fit_predict_extratrees(X_tr, y_tr, X_va, y_va, X_te, seed, params):
    p = dict(n_estimators=500, min_samples_leaf=30, max_features=0.3)
    p.update(params)

    def prep(X):  # NaN -> sentinel, kategori -> kode integer
        X = X.copy()
        for c in _cats(X):
            X[c] = X[c].cat.codes
        return X.astype("float32").fillna(-999)

    m = ExtraTreesClassifier(**p, n_jobs=N_THREADS, random_state=seed)
    m.fit(prep(X_tr), y_tr)
    imp = pd.Series(m.feature_importances_, index=X_tr.columns)
    return m.predict_proba(prep(X_va))[:, 1], m.predict_proba(prep(X_te))[:, 1], 0, imp


MODELS = {
    "catboost": fit_predict_catboost,
    "xgboost": fit_predict_xgboost,
    "logreg": fit_predict_logreg,
    "mlp": fit_predict_mlp,
    "extratrees": fit_predict_extratrees,
}


def run_model_cv(model, params, X, y, fold, X_test, seed, log=print):
    """5-fold untuk model non-LightGBM. Keluaran sama dengan train.run_lgb_cv."""
    fn = MODELS[model]
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    aucs, iters = [], []
    imp = pd.Series(0.0, index=X.columns)
    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        p_va, p_te, it, im = fn(X[tr], y[tr], X[va], y[va], X_test, seed, params)
        oof[va] = p_va
        test_pred += p_te / N_FOLDS
        aucs.append(roc_auc_score(y[va], p_va))
        iters.append(it)
        if im is not None:
            imp += im.reindex(X.columns).fillna(0) / N_FOLDS
        log(f"    fold {k}: AUC {aucs[-1]:.4f}, iter {it}")
    return oof, test_pred, aucs, iters, imp.sort_values(ascending=False)
