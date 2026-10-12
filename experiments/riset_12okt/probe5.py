"""Screening cepat: apakah fitur kandidat e-commerce menambah sinyal di atas OOF model terbaik?
LightGBM dengan init_score = logit(OOF) hanya memakai fitur kandidat (5 fold tetap)."""
import sys, gc
sys.path.insert(0, r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.io_utils import load, load_apps
from src.train import ks_score
apps = load_apps(); tr = apps[apps.is_test == 0].reset_index(drop=True); y = tr.target.values.astype(int)
fold = pd.read_csv(r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika\folds\folds.csv").set_index("application_id").loc[tr.application_id, "fold"].values
oof = np.load(r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika\oof\T_lgb_b3_4f3c7f.npy")
lo = np.log(oof / (1 - oof))
ec = load("ecommerce_transactions", usecols=["order_id", "customer_id", "transaction_datetime", "order_status", "order_total", "payment_method", "product_category"])
ec = ec.drop_duplicates("order_id")
e = ec.merge(tr[["customer_id", "decision_datetime"]], on="customer_id"); del ec; gc.collect()
e["h"] = (e.decision_datetime - e.transaction_datetime).dt.total_seconds() / 3600
e = e.sort_values(["customer_id", "h"]); e["rk"] = e.groupby("customer_id").cumcount() + 1
out = pd.DataFrame(index=tr.customer_id)
for k in [1, 2, 3, 5, 10]:
    out[f"h_k{k}"] = e[e.rk == k].set_index("customer_id")["h"]
for d in [3, 7, 14, 30, 90]:
    w = e[e.h <= 24 * d]
    out[f"n_{d}d"] = w.groupby("customer_id").size(); out[f"n_{d}d"] = out[f"n_{d}d"].fillna(0)
    out[f"amt_{d}d"] = w.groupby("customer_id").order_total.sum()
    out[f"cod_{d}d"] = (w.payment_method == "COD").groupby(w.customer_id).mean()
    out[f"cc_{d}d"] = (w.payment_method == "credit_card").groupby(w.customer_id).mean()
    out[f"canc_{d}d"] = (w.order_status == "cancelled").groupby(w.customer_id).mean()
span = e.groupby("customer_id")["h"].max()
out["rate_all"] = e.groupby("customer_id").size() / (span / 24)
out["accel_14_all"] = (out.n_14d / 14) / out.rate_all
out["mean_gap_last5"] = (out.h_k5 - out.h_k1) / 4
X = out.reset_index(drop=True).astype("float32")
base = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(5)]
p = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=300, feature_fraction=0.7,
         lambda_l2=10, verbose=-1, seed=42, num_threads=4)
new = np.zeros(len(y)); its = []
for k in range(5):
    trn, va = fold != k, fold == k
    dt = lgb.Dataset(X[trn], y[trn], init_score=lo[trn]); dv = lgb.Dataset(X[va], y[va], init_score=lo[va], reference=dt)
    m = lgb.train(p, dt, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(200, verbose=False)])
    new[va] = lo[va] + m.predict(X[va], num_iteration=m.best_iteration, raw_score=True); its.append(m.best_iteration)
aft = [roc_auc_score(y[fold == k], new[fold == k]) for k in range(5)]
print("AUC per fold OOF terbaik :", np.round(base, 5), "mean", round(np.mean(base), 5), "KS", round(ks_score(y, oof), 4))
print("AUC per fold + ec kandidat:", np.round(aft, 5), "mean", round(np.mean(aft), 5), "KS", round(ks_score(y, new), 4))
print("delta mean", round(np.mean(aft) - np.mean(base), 5), "| fold naik", sum(a > b for a, b in zip(aft, base)), "/5 | iter", its)
