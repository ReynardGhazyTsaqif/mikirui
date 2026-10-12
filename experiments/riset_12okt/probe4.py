import sys
sys.path.insert(0, r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.io_utils import load, load_apps
def auc(y, x):
    x = np.asarray(x, float); m = ~np.isnan(x)
    a = roc_auc_score(y[m], x[m]); return round(max(a, 1 - a), 4)
apps = load_apps(); tr = apps[apps.is_test == 0].reset_index(drop=True); y = tr.target.values
oof = np.load(r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika\oof\T_lgb_b3_4f3c7f.npy")
res = y - oof
bu = load("bureau_observations", usecols=["application_id", "report_datetime", "active_accounts", "outstanding_balance", "inquiries_last_6m", "delinquent_accounts_last_12m"])
cur = bu.merge(tr[["application_id", "customer_id"]], on="application_id")
pa = load("previous_applications", usecols=["customer_id", "application_datetime", "decision_datetime", "decision", "approved_amount", "term_months"])
m = cur.merge(pa, on="customer_id", how="left")
d = (m.report_datetime - m.application_datetime).dt.days
m["own_inq6"] = ((d >= 0) & (d <= 182)).astype(int)
endt = m.decision_datetime + pd.to_timedelta(m.term_months.fillna(0) * 30.4, unit="D")
m["own_active"] = ((m.decision == "approved") & (m.decision_datetime <= m.report_datetime) & (endt > m.report_datetime)).astype(int)
m["own_active_amt"] = m.approved_amount.where(m.own_active == 1, 0)
g = m.groupby(["application_id", "report_datetime"]).agg(own_inq6=("own_inq6", "sum"), own_active=("own_active", "sum"), own_amt=("own_active_amt", "sum")).reset_index()
c2 = cur.merge(g, on=["application_id", "report_datetime"], how="left")
c2["ext_inq"] = c2.inquiries_last_6m - c2.own_inq6
c2["ext_acc"] = c2.active_accounts - c2.own_active
c2["ext_out"] = c2.outstanding_balance - c2.own_amt
f = c2.groupby("application_id")[["ext_inq", "ext_acc", "ext_out", "inquiries_last_6m", "active_accounts", "own_inq6", "own_active"]].mean()
t = tr[["application_id"]].merge(f, left_on="application_id", right_index=True, how="left")
for c in f.columns:
    x = t[c].values.astype(float); mk = ~np.isnan(x)
    rc = np.corrcoef(pd.Series(x[mk]).rank(), pd.Series(res[mk]).rank())[0, 1]
    print(f"{c:20s} AUC {auc(y, x)}  korelasi-rank dgn residual {rc:+.4f}")
print("ref: |korelasi| ~0.0076 = 2 sd noise (n=70k)")
print("share ext_inq<0:", round((t.ext_inq < 0).mean(), 3), "| share ext_acc<0:", round((t.ext_acc < 0).mean(), 3))
