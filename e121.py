"""E121: model level order e-commerce (nested) sebagai fitur untuk LightGBM tuned.

1. Tabel order (1 baris = 1 order, semua customer train+test), hanya fitur intrinsik order: jam sebelum
   cutoff, nominal, nominal / income deklarasi, metode bayar, status, jumlah item, kuantitas, jam, hari,
   porsi nominal per kategori. Fitur konteks customer sengaja tidak dipakai (riset 12 Okt butir J).
2. Nested: untuk fold luar k, fitur baris validasi fold k dan test berasal dari model baris yang dilatih
   pada fold != k; fitur baris train (fold j != k) berasal dari model yang dilatih pada fold di luar {k, j}.
   Early stopping memakai 10% customer holdout di dalam data latih model baris itu sendiri.
3. Prediksi order (raw score) diagregasi per customer: mean, max, min, std, 3 dan 10 order terakhir,
   rata-rata order 30 hari terakhir -> 7 kolom ecm_*.
4. LightGBM tuned (parameter Optuna, lr 0.02) pada B3 + 7 kolom, 5 fold x seed.

Contoh:
  python e121.py --build                 # bangun fitur nested -> features/e121_ordermodel_v1.npz
  python e121.py --build --shuffle       # uji label acak (audit-leakage lapis 3) -> ..._shuffle.npz
  python e121.py --cv --rounds 15000 --seeds 42 43 44 [--submit]
"""
import argparse
import gc
import sys
from argparse import Namespace
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FEATURES_DIR, N_FOLDS, ROOT, SEED
from src.io_utils import load, load_apps
from src.train import Timer, ks_score, log_experiment, make_folds, write_submission

ORDERS = FEATURES_DIR / "ec_orders_v1.parquet"
CATS = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
FEATS = ["recency_h", "total", "total_vi", "pm", "st", "n_items", "qty", "hour", "dow"] + CATS
ROW_P = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=500,
             feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1,
             seed=SEED, deterministic=True, force_row_wise=True, num_threads=4)
LOG = ROOT / "experiments" / "e121.log"
AGGX = False


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_orders(apps):
    """Tabel order dengan fitur intrinsik; cutoff = decision_datetime aplikasi customer."""
    if ORDERS.exists():
        return pd.read_parquet(ORDERS)
    ec = load("ecommerce_transactions")
    share = pd.crosstab(ec["order_id"], ec["product_category"], values=ec["item_total"], aggfunc="sum",
                        normalize="index").astype("float32")
    share.columns = share.columns.astype(str)
    od = ec.groupby("order_id", sort=True).agg(
        customer_id=("customer_id", "first"), t=("transaction_datetime", "first"),
        total=("order_total", "first"), pm=("payment_method", "first"), st=("order_status", "first"),
        n_items=("order_item_id", "size"), qty=("quantity", "sum"))
    del ec
    gc.collect()
    od = od.join(share[CATS]).reset_index(drop=True)
    del share
    prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
    od = od.merge(apps[["customer_id", "decision_datetime"]], on="customer_id").merge(prof, on="customer_id")
    assert (od["t"] < od["decision_datetime"]).all()
    od["recency_h"] = (od["decision_datetime"] - od["t"]).dt.total_seconds() / 3600
    od["hour"] = od["t"].dt.hour
    od["dow"] = od["t"].dt.dayofweek
    od["pm"] = od["pm"].cat.codes
    od["st"] = od["st"].cat.codes
    od["total_vi"] = od["total"] / od["declared_monthly_income"]
    od = od[["customer_id"] + FEATS].sort_values(["customer_id", "recency_h"], kind="stable")
    od[FEATS] = od[FEATS].astype("float32")
    od = od.reset_index(drop=True)
    od.to_parquet(ORDERS)
    return od


ITEMS = FEATURES_DIR / "ec_items_v1.parquet"
ITEM_FEATS = ["recency_h", "item_total", "quantity", "unit_price", "cat", "pm", "st", "order_total",
              "share_in_order", "item_vi", "hour", "dow"]


def build_items(apps):
    """Tabel item (1 baris = 1 item order), fitur intrinsik item + order induknya."""
    if ITEMS.exists():
        return pd.read_parquet(ITEMS)
    it = load("ecommerce_transactions")
    prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
    it = it.merge(apps[["customer_id", "decision_datetime"]], on="customer_id").merge(prof, on="customer_id")
    assert (it["transaction_datetime"] < it["decision_datetime"]).all()
    it["recency_h"] = (it["decision_datetime"] - it["transaction_datetime"]).dt.total_seconds() / 3600
    it["unit_price"] = it["item_total"] / it["quantity"].where(it["quantity"] > 0)
    it["share_in_order"] = it["item_total"] / it["order_total"].where(it["order_total"] > 0)
    it["item_vi"] = it["item_total"] / it["declared_monthly_income"]
    it["hour"] = it["transaction_datetime"].dt.hour
    it["dow"] = it["transaction_datetime"].dt.dayofweek
    it["cat"] = it["product_category"].cat.codes
    it["pm"] = it["payment_method"].cat.codes
    it["st"] = it["order_status"].cat.codes
    it = it[["customer_id"] + ITEM_FEATS].sort_values(["customer_id", "recency_h"], kind="stable")
    it[ITEM_FEATS] = it[ITEM_FEATS].astype("float32")
    it = it.reset_index(drop=True)
    it.to_parquet(ITEMS)
    return it


def fit_rows(od, mask, y_row, seed):
    """Latih model baris pada baris mask; early stopping pada 10% customer holdout di dalamnya."""
    cust = od.loc[mask, "customer_id"].unique()
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(cust, len(cust) // 10, replace=False))
    is_hold = od["customer_id"].isin(hold).values
    tr, es = mask & ~is_hold, mask & is_hold
    cats = [c for c in ("pm", "st", "cat") if c in FEATS]
    dtr = lgb.Dataset(od.loc[tr, FEATS], y_row[tr], categorical_feature=cats, free_raw_data=True)
    des = lgb.Dataset(od.loc[es, FEATS], y_row[es], reference=dtr)
    m = lgb.train({**ROW_P, "seed": seed}, dtr, 4000, valid_sets=[des],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def aggregate(od, pred, rows, customers):
    """Agregasi prediksi order (raw) per customer -> DataFrame berindeks customer_id (urut customers)."""
    d = pd.DataFrame({"customer_id": od["customer_id"].values[rows], "p": pred,
                      "rec": od["recency_h"].values[rows]})
    d["rk"] = d.groupby("customer_id").cumcount()  # od sudah urut recency naik per customer
    g = d.groupby("customer_id")["p"]
    agg = pd.DataFrame({"mean": g.mean(), "max": g.max(), "min": g.min(), "std": g.std(),
                        "last3": d[d.rk < 3].groupby("customer_id")["p"].mean(),
                        "last10": d[d.rk < 10].groupby("customer_id")["p"].mean(),
                        "rec30": d[d.rec <= 720].groupby("customer_id")["p"].mean()})
    if AGGX:  # E121e: agregat waktu tambahan
        d["q"] = 1 / (1 + np.exp(-d["p"]))
        for hl in (168, 720):  # bobot peluruhan, half-life 7 / 30 hari
            d["w"] = 0.5 ** (d["rec"] / hl)
            d["wp"] = d["w"] * d["p"]
            s = d.groupby("customer_id")[["w", "wp"]].sum()
            agg[f"ewm{hl}"] = s["wp"] / s["w"]
        agg["rec7"] = d[d.rec <= 168].groupby("customer_id")["p"].mean()
        agg["rec90"] = d[d.rec <= 2160].groupby("customer_id")["p"].mean()
        agg["q90"] = g.quantile(0.9)
        top = d.sort_values(["customer_id", "p"], ascending=[True, False], kind="stable")
        agg["top3"] = top[top.groupby("customer_id").cumcount() < 3].groupby("customer_id")["p"].mean()
        agg["risk30"] = d[d.rec <= 720].groupby("customer_id")["q"].sum()
        agg["risk_all"] = d.groupby("customer_id")["q"].sum()
        agg["trend30"] = agg["rec30"] - agg["mean"]
    agg.columns = [f"ecm_{c}" for c in agg.columns]
    return agg.reindex(customers).astype("float32")


def build(shuffle=False, tag="", item=False):
    apps = load_apps()
    m = apps["is_test"].eq(0).values
    tr_apps = apps[m].reset_index(drop=True)
    y = tr_apps["target"].astype(int).values
    if shuffle:
        y = np.random.default_rng(42).permutation(y)
    fold = make_folds(tr_apps)["fold"].values
    od = build_items(apps) if item else build_orders(apps)
    cfold = pd.Series(fold, index=tr_apps["customer_id"])
    cy = pd.Series(y, index=tr_apps["customer_id"])
    row_fold = od["customer_id"].map(cfold).fillna(-1).astype(int).values  # -1 = customer test
    y_row = od["customer_id"].map(cy).fillna(0).astype(int).values
    tr_c, te_c = tr_apps["customer_id"].values, apps.loc[~m, "customer_id"].values
    out = {}
    for k in range(N_FOLDS):
        with Timer() as t:
            mk = fit_rows(od, (row_fold >= 0) & (row_fold != k), y_row, SEED + k)
            rows_va, rows_te = row_fold == k, row_fold == -1
            f_va = aggregate(od, mk.predict(od.loc[rows_va, FEATS], raw_score=True), rows_va,
                             tr_c[fold == k])
            f_te = aggregate(od, mk.predict(od.loc[rows_te, FEATS], raw_score=True), rows_te, te_c)
            parts = []
            for j in range(N_FOLDS):
                if j == k:
                    continue
                mj = fit_rows(od, (row_fold >= 0) & (row_fold != k) & (row_fold != j), y_row,
                              SEED + 10 * k + j)
                rows_j = row_fold == j
                parts.append(aggregate(od, mj.predict(od.loc[rows_j, FEATS], raw_score=True), rows_j,
                                       tr_c[fold == j]))
            f_tr = pd.concat(parts).reindex(tr_c[fold != k])
        out[f"tr{k}"], out[f"va{k}"], out[f"te{k}"] = f_tr.values, f_va.values, f_te.values
        log(f"build fold {k}: AUC ecm_mean val {roc_auc_score(y[fold == k], f_va['ecm_mean'].fillna(-9)):.4f} "
            f"({t.minutes:.1f} mnt)")
    out["cols"] = np.array(f_va.columns)
    path = FEATURES_DIR / f"e121_ordermodel_v1{tag}{'_shuffle' if shuffle else ''}.npz"
    np.savez(path, **out)
    log(f"fitur disimpan {path.name}")


def cv(rounds, seeds, submit, featfile, name=None, basefs="b3", study="lgb", extra=""):
    from tune import Ctx, best_tuned
    from src.tuning import FINAL_LR, full_params
    ctx = Ctx(Namespace(smoke=False, stop_at=None))
    del ctx.Xb2_tr, ctx.Xb2_te
    gc.collect()
    if basefs != "b3":  # "all" = 528 kolom semua grup registry, atau nama featset di experiments/featsets
        import run
        from src import selection as sel
        from src.features.registry import GROUPS
        apps = load_apps()
        m = apps["is_test"].eq(0).values
        X = run.build_matrix(list(GROUPS), apps)
        if basefs != "all":
            keep = set(sel.load_featset(basefs))
            X = X[[c for c in X.columns if c in keep]]
        ctx.X_tr, ctx.X_te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
        del X, apps
        gc.collect()
        log(f"basis fitur {basefs}: {ctx.X_tr.shape[1]} kolom")
    if extra:  # parquet tambahan di features/ (index application_id, urutan sama dengan apps)
        apps = load_apps()
        m = apps["is_test"].eq(0).values
        E = pd.concat([pd.read_parquet(FEATURES_DIR / f).reindex(apps["application_id"])
                       for f in extra.split("+")], axis=1)
        ctx.X_tr = pd.concat([ctx.X_tr, E[m].reset_index(drop=True)], axis=1)
        ctx.X_te = pd.concat([ctx.X_te, E[~m].reset_index(drop=True)], axis=1)
        del E, apps
        gc.collect()
        log(f"fitur tambahan {extra}: total {ctx.X_tr.shape[1]} kolom")
    tuned, _ = best_tuned(ctx, study, 15)  # study Optuna "tune_<study>"
    log(f"parameter dari study tune_{study}: {tuned}")
    model = study.split("_")[0]  # lgb / cat / xgb
    params = full_params(model, tuned, FINAL_LR[model])
    zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in featfile.split("+")]
    cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
    z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key != "cols"}
    featfile = featfile.split("+")[-1]
    name = name or "E121" + featfile.split("e121_ordermodel_v1")[1].replace(".npz", "")
    log(f"=== CV {name} rounds {rounds} seeds {seeds}")
    for s in seeds:
        path = ROOT / "oof" / f"{name}_s{s}.npy"
        if path.exists():
            continue
        oof, pred, aucs, iters = np.zeros(len(ctx.y)), np.zeros(len(ctx.X_te)), [], []
        with Timer() as t:
            for k in range(N_FOLDS):
                tr, va = ctx.fold != k, ctx.fold == k
                X_tr = pd.concat([ctx.X_tr[tr].reset_index(drop=True),
                                  pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
                X_va = pd.concat([ctx.X_tr[va].reset_index(drop=True),
                                  pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
                X_te = pd.concat([ctx.X_te, pd.DataFrame(z[f"te{k}"], columns=cols)], axis=1)
                from src.tuning import fit_fold
                p_va, p_te, it = fit_fold(model, params, X_tr, ctx.y[tr], X_va, ctx.y[va], X_te, s,
                                          rounds=rounds)
                oof[va] = p_va
                pred += p_te / N_FOLDS
                aucs.append(roc_auc_score(ctx.y[va], oof[va]))
                iters.append(it)
        np.save(path, oof)
        np.save(ROOT / "preds" / f"{name}_s{s}.npy", pred)
        log(f"{name} seed {s}: AUC mean-fold {np.mean(aucs):.5f} KS {ks_score(ctx.y, oof):.4f} iter {iters} "
            f"({t.minutes:.1f} mnt)")
    all_s = sorted(int(p.stem.rsplit("_s", 1)[1]) for p in (ROOT / "oof").glob(f"{name}_s*.npy"))
    oof = np.mean([np.load(ROOT / "oof" / f"{name}_s{s}.npy") for s in all_s], axis=0)
    pred = np.mean([np.load(ROOT / "preds" / f"{name}_s{s}.npy") for s in all_s], axis=0)
    aucs = [roc_auc_score(ctx.y[ctx.fold == k], oof[ctx.fold == k]) for k in range(N_FOLDS)]
    kss = [ks_score(ctx.y[ctx.fold == k], oof[ctx.fold == k]) for k in range(N_FOLDS)]
    log(f"{name} BAG {all_s}: AUC mean-fold {np.mean(aucs):.5f} KS {ks_score(ctx.y, oof):.4f}")
    sub, md5 = "", ""
    if submit:
        p, md5 = write_submission(ctx.test_ids, pred, f"{name}_bag{len(all_s)}")
        sub = p.name
        log(f"submission {sub} md5 {md5}")
    log_experiment(name, f"{model} tuned + 7 fitur model order e-commerce nested, batas iterasi {rounds}, "
                   f"bag {all_s}, basis {basefs}", f"{basefs}+ecm", ctx.X_tr.shape[1] + len(cols), f"{model}_tuned_bag", params, aucs,
                   roc_auc_score(ctx.y, oof), 0, "kandidat", ks_score(ctx.y, oof), kss,
                   submission_file=sub, md5_hash=md5)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--rounds", type=int, default=15000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="", help="akhiran nama file fitur/eksperimen, mis. _b")
    ap.add_argument("--item", action="store_true", help="model level item, bukan order")
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--minchild", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--use", default="", help="file fitur untuk --cv, digabung dengan '+'")
    ap.add_argument("--study", default="lgb", help="study Optuna tune_<study> sumber parameter")
    ap.add_argument("--basefs", default="b3", help="b3 | all | nama featset")
    ap.add_argument("--aggx", action="store_true", help="E121e: agregat waktu tambahan saat --build")
    ap.add_argument("--extra", default="", help="parquet tambahan di features/ untuk --cv, digabung '+'")
    ap.add_argument("--name", default=None, help="nama eksperimen untuk --cv (default E121<tag>)")
    a = ap.parse_args()
    ROW_P.update(num_leaves=a.leaves, min_child_samples=a.minchild, learning_rate=a.lr)
    if a.item:
        FEATS[:] = ITEM_FEATS
    global AGGX
    AGGX = a.aggx
    if a.build:
        build(a.shuffle, a.tag, a.item)
    if a.cv:
        cv(a.rounds, a.seeds, a.submit,
           a.use or f"e121_ordermodel_v1{a.tag}{'_shuffle' if a.shuffle else ''}.npz", a.name, a.basefs, a.study, a.extra)


if __name__ == "__main__":
    main()
