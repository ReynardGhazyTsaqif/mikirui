import sys, gc, time
sys.path.insert(0, r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.io_utils import load, load_apps
from src.train import ks_score
R = r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
apps = load_apps(); tr = apps[apps.is_test == 0].reset_index(drop=True); y = tr.target.values.astype(int)
fold = pd.read_csv(R + r"\folds\folds.csv").set_index("application_id").loc[tr.application_id, "fold"].values
cinfo = pd.DataFrame({"customer_id": tr.customer_id, "y": y, "fold": fold, "cut": tr.decision_datetime})
oof = np.load(R + r"\oof\T_lgb_b3_4f3c7f.npy"); lo = np.log(oof / (1 - oof))
base = np.mean([roc_auc_score(y[fold == k], oof[fold == k]) for k in range(5)])
basef = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(5)]
P = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=500, feature_fraction=0.8,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=42, num_threads=4)
prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
ec = load("ecommerce_transactions")
ec = ec[ec.customer_id.isin(set(cinfo.customer_id))]
cs = pd.crosstab(ec.order_id, ec.product_category, values=ec.item_total, aggfunc="sum", normalize="index").astype("float32")
cs.columns = cs.columns.astype(str)
od = ec.groupby("order_id").agg(customer_id=("customer_id", "first"), t=("transaction_datetime", "first"),
                                 total=("order_total", "first"), pm=("payment_method", "first"),
                                 st=("order_status", "first"), n_items=("order_item_id", "size"), qty=("quantity", "sum"))
del ec; gc.collect()
od = od.join(cs).reset_index(drop=True); del cs
cats7 = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
od = od.merge(cinfo[["customer_id", "cut", "y", "fold"]], on="customer_id").merge(prof, on="customer_id", how="left")
od["recency"] = (od.cut - od.t).dt.total_seconds() / 3600
od["hour"] = od.t.dt.hour; od["dow"] = od.t.dt.dayofweek; od["dom"] = od.t.dt.day
od["pm"] = od.pm.cat.codes; od["st"] = od.st.cat.codes
od["total_vi"] = od.total / od.declared_monthly_income
od = od.sort_values(["customer_id", "t"]).reset_index(drop=True)
g = od.groupby("customer_id")
od["gap_prev_h"] = g.t.diff().dt.total_seconds() / 3600
od["gap_next_h"] = -g.t.diff(-1).dt.total_seconds() / 3600
od["rank_from_end"] = g.cumcount(ascending=False)
od["n_cust"] = g.t.transform("size")
od["total_vs_med"] = od.total / g.total.transform("median")
# intensitas lokal: jumlah order customer dalam 7 dan 30 hari sebelum order ini
ts = od.t.values.astype("datetime64[s]").astype(np.int64)
cid = od.customer_id.values
starts = np.r_[0, np.flatnonzero(cid[1:] != cid[:-1]) + 1, len(cid)]
w7 = np.zeros(len(od), np.int32); w30 = np.zeros(len(od), np.int32)
for a, b in zip(starts[:-1], starts[1:]):
    tt = ts[a:b]
    w7[a:b] = np.arange(b - a) - np.searchsorted(tt, tt - 7 * 86400)
    w30[a:b] = np.arange(b - a) - np.searchsorted(tt, tt - 30 * 86400)
od["n_prev7"] = w7; od["n_prev30"] = w30
base_feats = ["recency", "total", "total_vi", "pm", "st", "n_items", "qty", "hour", "dow"] + cats7
rich_feats = base_feats + ["dom", "gap_prev_h", "gap_next_h", "rank_from_end", "n_cust", "total_vs_med", "n_prev7", "n_prev30"]

def run(feats, label):
    t0 = time.time(); pred = np.zeros(len(od), "float32"); imp = pd.Series(0.0, index=feats)
    for k in range(5):
        trn, va = (od.fold != k).values, (od.fold == k).values
        dt = lgb.Dataset(od.loc[trn, feats], od.y[trn], categorical_feature=["pm", "st"])
        dv = lgb.Dataset(od.loc[va, feats], od.y[va], reference=dt)
        m = lgb.train(P, dt, 600, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred[va] = m.predict(od.loc[va, feats], num_iteration=m.best_iteration, raw_score=True)
        imp += pd.Series(m.feature_importance("gain"), index=feats)
    od["p"] = pred
    gg = od.groupby("customer_id")["p"]
    agg = pd.DataFrame({"mean": gg.mean(), "max": gg.max(), "min": gg.min(), "std": gg.std(),
                        "last3": od[od.rank_from_end < 3].groupby("customer_id")["p"].mean(),
                        "last10": od[od.rank_from_end < 10].groupby("customer_id")["p"].mean(),
                        "rec30": od[od.recency <= 720].groupby("customer_id")["p"].mean()})
    X = cinfo[["customer_id"]].merge(agg, left_on="customer_id", right_index=True, how="left").drop(columns="customer_id").astype("float32")
    sp = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=300, lambda_l2=10, verbose=-1, seed=42, num_threads=4)
    new = np.zeros(len(y))
    for k in range(5):
        trn, va = fold != k, fold == k
        dt = lgb.Dataset(X[trn], y[trn], init_score=lo[trn]); dv = lgb.Dataset(X[va], y[va], init_score=lo[va], reference=dt)
        m = lgb.train(sp, dt, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(200, verbose=False)])
        new[va] = lo[va] + m.predict(X[va], num_iteration=m.best_iteration, raw_score=True)
    aft = [roc_auc_score(y[fold == k], new[fold == k]) for k in range(5)]
    print(f"[{label}] customer AUC (mean pred) {roc_auc_score(y, X['mean'].fillna(X['mean'].median())):.4f} | screen {base:.5f} -> {np.mean(aft):.5f} "
          f"(delta {np.mean(aft) - base:+.5f}, fold naik {sum(a > b for a, b in zip(aft, basef))}/5) | KS {ks_score(y, oof):.4f} -> {ks_score(y, new):.4f} | {(time.time() - t0) / 60:.1f} mnt")
    print("   importance (gain %):", (imp / imp.sum() * 100).sort_values(ascending=False).head(10).round(1).to_dict(), flush=True)

run(base_feats, "ec dasar")
run(rich_feats, "ec kaya")
