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

# 1) account_behavior: baris = customer x bulan
ab = load("account_behavior")
ab = ab.merge(cinfo[["customer_id", "cut"]], on="customer_id")
ab["recency"] = (ab.cut.dt.year * 12 + ab.cut.dt.month) - (ab.month.dt.year * 12 + ab.month.dt.month)
prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"])
ab = ab.merge(prof, on="customer_id", how="left")
num = ["opening_balance", "closing_balance", "inflows", "outflows", "income_like_credits", "average_balance", "minimum_balance"]
for c in num:
    ab[c + "_vi"] = ab[c] / ab.declared_monthly_income
ab["net"] = (ab.inflows - ab.outflows) / ab.declared_monthly_income
ab["month_of_year"] = ab.month.dt.month
feats = ["recency", "low_balance_days", "net", "month_of_year"] + num + [c + "_vi" for c in num]
row_model("account_behavior", ab[["customer_id", "recency"] + [f for f in feats if f != "recency"]].copy(), feats)
del ab; gc.collect()

# 2) repayment_history: baris = cicilan
rp = load("repayment_history")
rp = rp.merge(cinfo[["customer_id", "cut"]], on="customer_id")
rp["recency"] = (rp.cut - rp.due_date).dt.days
rp["paid_ratio"] = rp.amount_paid / rp.amount_due.where(rp.amount_due > 0)
rp["delay"] = (rp.last_payment_datetime - rp.due_date).dt.total_seconds() / 86400
rp["unpaid"] = rp.last_payment_datetime.isna().astype("int8")
rp = rp.sort_values(["loan_id", "due_date"]); rp["inst_no"] = rp.groupby("loan_id").cumcount() + 1
rp["n_inst_loan"] = rp.groupby("loan_id")["inst_no"].transform("max")
pa = load("previous_applications", usecols=["loan_id", "approved_amount", "annual_interest_rate", "term_months"])
rp = rp.merge(pa.dropna(subset=["loan_id"]), on="loan_id", how="left"); del pa
feats = ["recency", "amount_due", "paid_ratio", "delay", "unpaid", "days_past_due", "max_days_past_due", "inst_no",
         "n_inst_loan", "approved_amount", "annual_interest_rate", "term_months"]
row_model("repayment_history", rp[["customer_id"] + feats].copy(), feats)
del rp; gc.collect()

# 3) ecommerce: baris = order
ec = load("ecommerce_transactions")
ec = ec[ec.customer_id.isin(set(cinfo.customer_id))]
cat_share = pd.crosstab(ec.order_id, ec.product_category, values=ec.item_total, aggfunc="sum", normalize="index").astype("float32")
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
