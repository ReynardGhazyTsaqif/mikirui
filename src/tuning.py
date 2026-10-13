"""Fase 5: ruang pencarian Optuna dan perakitan parameter untuk LightGBM, CatBoost, XGBoost.

Pencarian memakai learning rate tetap yang agak besar (cepat); parameter terbaik lalu dilatih ulang
dengan learning rate kecil di tahap final (lihat tune.py). Ruang pencarian mengikuti planning.
"""
import hashlib
import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from .config import LGB_DEV, N_THREADS, SEED
from .models import fit_predict_catboost, fit_predict_xgboost

SEARCH_LR = {"lgb": 0.05, "cat": 0.08, "xgb": 0.05}
FINAL_LR = {"lgb": 0.02, "cat": 0.04, "xgb": 0.02}


def suggest(model, trial):
    """Parameter yang dituning saja (tanpa parameter tetap)."""
    if model == "lgb":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            max_depth=trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 8, 10, 12]),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 500, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.1, 0.8),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
            min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
            extra_trees=trial.suggest_categorical("extra_trees", [False, True]),
        )
    if model == "cat":
        p = dict(
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 50, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
            rsm=trial.suggest_float("rsm", 0.2, 1.0),
            grow_policy=trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
            bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        )
        if p["grow_policy"] == "Depthwise":
            p["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 10, 300, log=True)
        if p["bootstrap_type"] == "Bayesian":
            p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 3.0)
        else:
            p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        return p
    if model == "xgb":
        return dict(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 100, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        )
    raise ValueError(model)


def full_params(model, tuned, lr):
    """Gabungkan parameter hasil tuning dengan parameter tetap + learning rate."""
    if model == "lgb":
        return {**LGB_DEV, **tuned, "learning_rate": lr, "bagging_freq": 1}
    if model == "cat":
        return {**tuned, "learning_rate": lr, "od_wait": 200 if lr >= 0.06 else 300}
    if model == "xgb":
        return {**tuned, "learning_rate": lr}
    raise ValueError(model)


def params_hash(model, tuned):
    return hashlib.md5(json.dumps([model, tuned], sort_keys=True).encode()).hexdigest()[:6]


def fit_fold(model, params, X_tr, y_tr, X_va, y_va, X_te, seed, rounds=None):
    """Satu fold: return (pred_va, pred_te atau None, best_iter). rounds = batas iterasi (default 5000)."""
    if rounds and model != "lgb":
        params = {**params, ("iterations" if model == "cat" else "n_estimators"): rounds}
    if model == "lgb":
        dtr = lgb.Dataset(X_tr, y_tr, params={"max_bin": params.get("max_bin", 255)})
        dva = lgb.Dataset(X_va, y_va, reference=dtr)
        m = lgb.train({**params, "seed": seed}, dtr, rounds or 5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p_te = None if X_te is None else m.predict(X_te, num_iteration=m.best_iteration)
        return m.predict(X_va, num_iteration=m.best_iteration), p_te, m.best_iteration
    fn = fit_predict_catboost if model == "cat" else fit_predict_xgboost
    X_te_ = X_va.iloc[:1] if X_te is None else X_te
    p_va, p_te, it, _ = fn(X_tr, y_tr, X_va, y_va, X_te_, seed, params)
    return p_va, (None if X_te is None else p_te), it


def make_objective(model, X, y, fold, search_folds, log):
    """Objective Optuna: rata-rata AUC pada fold pencarian (subset dari 5 fold tetap)."""
    def objective(trial):
        params = full_params(model, suggest(model, trial), SEARCH_LR[model])
        aucs = []
        for step, k in enumerate(search_folds):
            tr, va = fold != k, fold == k
            p_va, _, it = fit_fold(model, params, X[tr], y[tr], X[va], y[va], None, SEED)
            aucs.append(roc_auc_score(y[va], p_va))
            trial.report(float(np.mean(aucs)), step)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_aucs", [round(a, 6) for a in aucs])
        return float(np.mean(aucs))
    return objective
