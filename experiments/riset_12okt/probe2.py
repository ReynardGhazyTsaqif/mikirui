import sys, gc, glob, os
sys.path.insert(0, r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.io_utils import load, load_apps

def auc(y, x):
    x = np.asarray(x, dtype=float); m = ~np.isnan(x)
    if m.sum() < 100 or len(np.unique(x[m])) < 2: return np.nan
    a = roc_auc_score(y[m], x[m]); return round(max(a, 1 - a), 4)

apps = load_apps()
tr = apps[apps.is_test == 0].reset_index(drop=True)
y = tr["target"].values.astype(int)
ids_cur = set(apps.application_id)

print("== 1. Tumpang tindih ID antartabel")
pa = load("previous_applications", usecols=["application_id", "customer_id", "loan_id"])
print("prev app_id yang = app_id train/test:", len(set(pa.application_id) & ids_cur))
bu = load("bureau_observations", usecols=["application_id", "customer_id"])
bu_ids = set(bu.application_id)
print("bureau app_id bukan current & bukan previous:", len(bu_ids - ids_cur - set(pa.application_id)))
bmap = bu.drop_duplicates("application_id")
cur_c = apps.set_index("application_id").customer_id
mm = bmap[bmap.application_id.isin(ids_cur)]
print("bureau current: customer_id beda dgn aplikasi:", int((mm.customer_id.values != cur_c.loc[mm.application_id].values).sum()))
rp = load("repayment_history", usecols=["loan_id", "customer_id"])
print("loan_id repayment tanpa previous app:", len(set(rp.loan_id) - set(pa.loan_id.dropna())))
lc = rp.drop_duplicates("loan_id").merge(pa[["loan_id", "customer_id"]].dropna(), on="loan_id", suffixes=("_r", "_p"))
print("loan_id repayment customer beda:", int((lc.customer_id_r != lc.customer_id_p).sum()))
del pa, bu, rp, bmap, mm, lc; gc.collect()

print("== 2. AUC univariat fitur yang sudah ada (top 25 dari cache)")
res = []
for p in sorted(glob.glob(r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika\features\*.parquet")):
    df = pd.read_parquet(p).iloc[:len(tr)]
    for c in df.columns:
        if str(df[c].dtype) in ("category", "object", "bool"): continue
        res.append((os.path.basename(p)[:-8], c, auc(y, df[c].values)))
    del df; gc.collect()
r = pd.DataFrame(res, columns=["grup", "kol", "auc"]).sort_values("auc", ascending=False)
print(r.head(25).to_string(index=False))
print("jumlah kolom:", len(r), "| AUC>=0.65:", int((r.auc >= .65).sum()), "| 0.60-0.65:", int(((r.auc >= .6) & (r.auc < .65)).sum()))
print("top per grup:"); print(r.groupby("grup").head(1).to_string(index=False))

print("== 3. E-commerce: recency ke-k dan jendela sangat pendek")
ec = load("ecommerce_transactions", usecols=["order_id", "customer_id", "transaction_datetime", "order_status", "order_total"])
ec = ec.drop_duplicates("order_id")
e = ec.merge(tr[["customer_id", "decision_datetime"]], on="customer_id")
del ec; gc.collect()
e["h"] = (e.decision_datetime - e.transaction_datetime).dt.total_seconds() / 3600
e = e.sort_values(["customer_id", "h"])
e["rk"] = e.groupby("customer_id").cumcount() + 1
out = pd.DataFrame(index=tr.customer_id)
for k in [1, 2, 3, 5, 10, 20]:
    out[f"h_k{k}"] = e[e.rk == k].set_index("customer_id")["h"]
for d in [2, 3, 5, 7, 14, 30, 90]:
    out[f"n_{d}d"] = e[e.h <= 24 * d].groupby("customer_id").size()
    out[f"n_{d}d"] = out[f"n_{d}d"].fillna(0)
span = e.groupby("customer_id")["h"].max()
out["n_all"] = e.groupby("customer_id").size()
out["rate_all"] = out.n_all / (span / 24)
out["accel_14_all"] = (out.n_14d / 14) / out.rate_all
out["accel_7_90"] = (out.n_7d / 7) / (out.n_90d / 90)
out["gap_k1_k2"] = out.h_k2 - out.h_k1
out["mean_gap_last5"] = (out.h_k5 - out.h_k1) / 4
out["span_days"] = span / 24
for c in out.columns:
    print(f"  {c:16s} AUC {auc(y, out[c].values)}")
