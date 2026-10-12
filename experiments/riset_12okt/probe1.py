import sys, gc
sys.path.insert(0, r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.io_utils import load, load_apps
from src.train import ks_score

def auc(y, x):
    m = ~pd.isna(x)
    a = roc_auc_score(y[m], np.asarray(x)[m]); return round(max(a, 1 - a), 4), round(m.mean(), 3)

apps = load_apps()
tr = apps[apps.is_test == 0].reset_index(drop=True)
y = tr["target"].values.astype(int)
print("== A. ID / urutan baris")
print("row order AUC", auc(y, np.arange(len(tr))))
print("app_id hex AUC", auc(y, tr.application_id.str[2:10].apply(lambda s: int(s, 16)).values))
print("cust_id hex AUC", auc(y, tr.customer_id.str[2:10].apply(lambda s: int(s, 16)).values))
ss = pd.read_csv(r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika\Dataset\sample_submission.csv")
te = apps[apps.is_test == 1]
print("sample_sub sorted by app_id?", ss.application_id.is_monotonic_increasing, "| test sorted by app_id?", te.application_id.is_monotonic_increasing, "| test sorted by decision?", te.decision_datetime.is_monotonic_increasing, "| train sorted by decision?", tr.decision_datetime.is_monotonic_increasing)
print("sample_sub unique preds", ss.default_probability.nunique())

print("== B. rate tier & OOF terbaik per tier")
oof = np.load(r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika\oof\T_lgb_b3_4f3c7f.npy")
print("KS rate saja", round(ks_score(y, tr.annual_interest_rate.values), 4), "| KS model", round(ks_score(y, oof), 4))
for r, g in tr.groupby("annual_interest_rate"):
    idx = g.index.values
    print(f"  rate {r}: n {len(idx)}, target {y[idx].mean():.3f}, AUC model dlm tier {roc_auc_score(y[idx], oof[idx]):.4f}")

print("== C. jeda aplikasi -> keputusan")
gap_h = (tr.decision_datetime - tr.application_datetime).dt.total_seconds() / 3600
print("process_hours AUC", auc(y, gap_h.values), "quantiles", np.round(np.quantile(gap_h, [0, .1, .5, .9, 1]), 1))
print("decision minute==0 share", (tr.decision_datetime.dt.minute == 0).mean().round(3))

print("== D. profil duplikat")
prof = load("customer_profiles")
cols = [c for c in prof.columns if c != "customer_id"]
dup = prof.duplicated(cols, keep=False)
print("baris profil duplikat (tanpa id):", int(dup.sum()))

print("== E. account_behavior: bulan terakhir vs bulan keputusan")
ab = load("account_behavior", usecols=["customer_id", "month"])
last = ab.groupby("customer_id")["month"].agg(["max", "min", "count"])
t2 = tr[["customer_id", "decision_datetime"]].merge(last, left_on="customer_id", right_index=True, how="left")
dm = (t2.decision_datetime.dt.year * 12 + t2.decision_datetime.dt.month) - (t2["max"].dt.year * 12 + t2["max"].dt.month)
print("selisih bulan (keputusan - bulan terakhir):", dm.value_counts(dropna=False).head(6).to_dict())
print("n bulan AUC", auc(y, t2["count"].values), "| bulan pertama (abs) AUC", auc(y, (t2["min"].dt.year * 12 + t2["min"].dt.month).values))
del ab; gc.collect()

print("== F. aktivitas di antara application_datetime dan decision_datetime")
pa = load("previous_applications", usecols=["customer_id", "application_datetime", "decision_datetime", "decision"])
m = pa.merge(tr[["customer_id", "application_datetime", "decision_datetime"]], on="customer_id", suffixes=("_p", "_c"))
inwin = (m.application_datetime_p >= m.application_datetime_c)
print("previous apps diajukan setelah application_datetime saat ini:", int(inwin.sum()))
print("previous apps DIPUTUS setelah application_datetime saat ini:", int((m.decision_datetime_p >= m.application_datetime_c).sum()))
bu = load("bureau_observations", usecols=["application_id", "customer_id", "available_datetime", "report_datetime", "score", "inquiries_last_6m"])
b = bu.merge(tr[["application_id", "application_datetime", "decision_datetime"]], on="application_id")
b["after_app"] = (b.report_datetime >= b.application_datetime).astype(int)
b["lag_dec_avail_d"] = (b.decision_datetime - b.available_datetime).dt.total_seconds() / 86400
g = b.groupby("application_id").agg(n=("score", "size"), after_app=("after_app", "mean"), lag=("lag_dec_avail_d", "min"))
t3 = tr[["application_id"]].merge(g, left_on="application_id", right_index=True, how="left")
print("laporan biro saat ini dibuat setelah aplikasi (share):", round(b.after_app.mean(), 3), "| n provider AUC", auc(y, t3.n.values), "| lag min AUC", auc(y, t3.lag.values))
del bu, b; gc.collect()

ec = load("ecommerce_transactions", usecols=["order_id", "customer_id", "transaction_datetime", "payment_method", "order_status", "order_total"])
ec = ec.drop_duplicates("order_id")
e = ec.merge(tr[["customer_id", "application_datetime", "decision_datetime"]], on="customer_id")
e["between"] = (e.transaction_datetime >= e.application_datetime).astype(int)
gb = e.groupby("customer_id")
f = pd.DataFrame({"n_between": gb["between"].sum(),
                  "amt_between": e.order_total.where(e.between == 1, 0).groupby(e.customer_id).sum()})
lastord = e.sort_values("transaction_datetime").groupby("customer_id").tail(1).set_index("customer_id")
f["last_pm"] = lastord.payment_method; f["last_status"] = lastord.order_status
f["h_since_last"] = (lastord.decision_datetime - lastord.transaction_datetime).dt.total_seconds() / 3600
f["h_last_vs_app"] = (lastord.transaction_datetime - lastord.application_datetime).dt.total_seconds() / 3600
t4 = tr[["customer_id"]].merge(f, left_on="customer_id", right_index=True, how="left")
print("order antara aplikasi & keputusan: share customer >0:", round((t4.n_between > 0).mean(), 3), "| AUC n_between", auc(y, t4.n_between.values), "| AUC amt_between", auc(y, t4.amt_between.values))
print("AUC jam sejak order terakhir", auc(y, t4.h_since_last.values), "| AUC jam order terakhir relatif ke aplikasi", auc(y, t4.h_last_vs_app.values))
print("target rate per metode bayar order terakhir:", pd.Series(y).groupby(t4.last_pm.values).agg(["mean", "size"]).round(3).to_dict("index"))
print("target rate per status order terakhir:", pd.Series(y).groupby(t4.last_status.values).agg(["mean", "size"]).round(3).to_dict("index"))
bins = pd.cut(t4.h_since_last, [-1, 1, 3, 6, 12, 24, 48, 96, 1e9])
print("target rate per jam sejak order terakhir:", pd.Series(y).groupby(bins.values, observed=True).agg(["mean", "size"]).round(3).to_dict("index"))
