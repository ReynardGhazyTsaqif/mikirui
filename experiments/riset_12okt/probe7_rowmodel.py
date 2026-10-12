"""Screening ide 'row-level child model': latih LightGBM di level baris tiap tabel (label = target customer),
prediksi OOF per fold customer, agregasi per customer, lalu uji apakah menambah sinyal di atas OOF terbaik
(LightGBM init_score = logit OOF). Hanya baris train. Catatan: screen sedikit optimis (tidak nested)."""
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
P = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=500, feature_fraction=0.8,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=42, num_threads=4)

def row_model(name, rows, feats, cats=()):
    t0 = time.time()
    rows = rows.merge(cinfo[["customer_id", "y", "fold"]], on="customer_id")
    pred = np.zeros(len(rows), dtype="float32"); aucs = []
    for k in range(5):
        trn, va = (rows.fold != k).values, (rows.fold == k).values
        dt = lgb.Dataset(rows.loc[trn, feats], rows.y[trn], categorical_feature=list(cats))
        dv = lgb.Dataset(rows.loc[va, feats], rows.y[va], reference=dt)
        m = lgb.train(P, dt, 600, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred[va] = m.predict(rows.loc[va, feats], num_iteration=m.best_iteration, raw_score=True)
        aucs.append(roc_auc_score(rows.y[va], pred[va]))
    rows["p"] = pred
    rows = rows.sort_values(["customer_id", "recency"])
    rows["rk"] = rows.groupby("customer_id").cumcount()
    g = rows.groupby("customer_id")["p"]
    agg = pd.DataFrame({"mean": g.mean(), "max": g.max(), "min": g.min(), "std": g.std(),
                        "last3": rows[rows.rk < 3].groupby("customer_id")["p"].mean(),
                        "last10": rows[rows.rk < 10].groupby("customer_id")["p"].mean()})
    X = cinfo[["customer_id"]].merge(agg, left_on="customer_id", right_index=True, how="left").drop(columns="customer_id").astype("float32")
    cust_auc = roc_auc_score(y[X["mean"].notna()], X["mean"][X["mean"].notna()])
    sp = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=300, lambda_l2=10, verbose=-1, seed=42, num_threads=4)
    new = np.zeros(len(y))
    for k in range(5):
        trn, va = fold != k, fold == k
        dt = lgb.Dataset(X[trn], y[trn], init_score=lo[trn]); dv = lgb.Dataset(X[va], y[va], init_score=lo[va], reference=dt)
        m = lgb.train(sp, dt, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(200, verbose=False)])
        new[va] = lo[va] + m.predict(X[va], num_iteration=m.best_iteration, raw_score=True)
    aft = [roc_auc_score(y[fold == k], new[fold == k]) for k in range(5)]
    basef = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(5)]
    print(f"[{name}] baris {len(rows):,} | AUC level-baris {np.mean(aucs):.4f} | AUC customer (mean pred) {cust_auc:.4f} | "
          f"screen: {base:.5f} -> {np.mean(aft):.5f} (delta {np.mean(aft) - base:+.5f}, fold naik {sum(a > b for a, b in zip(aft, basef))}/5), "
          f"KS {ks_score(y, oof):.4f} -> {ks_score(y, new):.4f} | {(time.time() - t0) / 60:.1f} mnt", flush=True)

prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
# 3) ecommerce: baris = order
ec = load("ecommerce_transactions")
ec = ec[ec.customer_id.isin(set(cinfo.customer_id))]
cat_share = pd.crosstab(ec.order_id, ec.product_category, values=ec.item_total, aggfunc="sum", normalize="index").astype("float32")
cat_share.columns = cat_share.columns.astype(str)
od = ec.groupby("order_id").agg(customer_id=("customer_id", "first"), t=("transaction_datetime", "first"),
                                 total=("order_total", "first"), pm=("payment_method", "first"),
                                 st=("order_status", "first"), n_items=("order_item_id", "size"), qty=("quantity", "sum"))
del ec; gc.collect()
od = od.join(cat_share).reset_index(drop=True); del cat_share
od = od.merge(cinfo[["customer_id", "cut"]], on="customer_id")
od["recency"] = (od.cut - od.t).dt.total_seconds() / 3600
od["hour"] = od.t.dt.hour; od["dow"] = od.t.dt.dayofweek
od["pm"] = od.pm.cat.codes; od["st"] = od.st.cat.codes
od = od.merge(prof, on="customer_id", how="left")
od["total_vi"] = od.total / od.declared_monthly_income
cats = [c for c in od.columns if c in ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]]
feats = ["recency", "total", "total_vi", "pm", "st", "n_items", "qty", "hour", "dow"] + cats
row_model("ecommerce (order)", od[["customer_id"] + feats].copy(), feats, cats=["pm", "st"])

del od; gc.collect()
# 4) previous_applications: baris = aplikasi lama
pa = load("previous_applications")
pa = pa.merge(cinfo[["customer_id", "cut"]], on="customer_id")
pa["recency"] = (pa.cut - pa.decision_datetime).dt.days
pa["proc_h"] = (pa.decision_datetime - pa.application_datetime).dt.total_seconds() / 3600
pa["dec"] = pa.decision.cat.codes
pa["appr_ratio"] = pa.approved_amount / pa.requested_amount
pa = pa.merge(prof, on="customer_id", how="left")
pa["req_vi"] = pa.requested_amount / pa.declared_monthly_income
pa["inst_vi"] = pa.scheduled_monthly_installment / pa.declared_monthly_income
pa["ends_after_cut"] = ((pa.decision_datetime + pd.to_timedelta(pa.term_months * 30.4, unit="D")) > pa.cut).astype("int8")
feats = ["recency", "proc_h", "dec", "requested_amount", "term_months", "approved_amount", "annual_interest_rate",
         "scheduled_monthly_installment", "appr_ratio", "req_vi", "inst_vi", "ends_after_cut"]
row_model("previous_applications", pa[["customer_id"] + feats].copy(), feats, cats=["dec"])
del pa; gc.collect()
# 5) bureau_observations: baris = laporan (saat ini + aplikasi lama)
bu = load("bureau_observations")
bu = bu.merge(cinfo[["customer_id", "cut"]], on="customer_id")
bu["recency"] = (bu.cut - bu.available_datetime).dt.days
bu["lag"] = (bu.available_datetime - bu.report_datetime).dt.days
bu["prov"] = bu.bureau_provider.cat.codes
bu["is_current"] = bu.application_id.isin(set(tr.application_id)).astype("int8")
feats = ["recency", "lag", "prov", "is_current", "score", "active_accounts", "outstanding_balance", "revolving_credit_limit",
         "revolving_outstanding_balance", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"]
row_model("bureau_observations", bu[["customer_id"] + feats].copy(), feats, cats=["prov"])
