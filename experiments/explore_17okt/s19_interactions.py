"""Langkah 2r: interaksi ramp x pi, sinyal segmen tanpa riwayat, ramp biro & aplikasi lama menjelang keputusan."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import apps, load
from resid import reference

pd.set_option("display.width", 220)
ref = reference("E122x1").set_index("customer_id")
X = pd.read_parquet("cache_error_frame.parquet")
X["r"] = ref.r
X["p_cal"] = ref.p_cal


def zr(x, r):
    ok = ~(np.isnan(x) | np.isnan(r))
    if ok.sum() < 200:
        return np.nan
    return np.corrcoef(rankdata(x[ok]), r[ok])[0, 1] * np.sqrt(ok.sum())


# 1. ramp x strata risiko (pakai skor model terkalibrasi sebagai strata, bukan pi_post saja)
X["risk_q"] = pd.qcut(X.p_cal.rank(method="first"), 5, labels=False)
for f in ["ec_last_h", "ec_n7", "ab_out_ramp", "ec_n_orders"]:
    zs = {int(q): round(zr(g[f].values.astype(float), g.r.values), 2) for q, g in X.groupby("risk_q")}
    print(f"z residual {f} per kuintil risiko model:", zs)
for f in ["ec_last_h", "ec_n7"]:
    zs = {h: round(zr(g[f].values.astype(float), g.r.values), 2) for h, g in X.groupby("has_hist")}
    print(f"z residual {f} per punya riwayat:", zs)
# 2. segmen tanpa riwayat: baterai cepat dari fitur cache yang ada
nh = X[~X.has_hist]
print("tanpa riwayat n =", len(nh), "target rate", nh.target.mean().round(4))
# 3. ramp biro: laporan sekarang vs laporan lama (customer punya laporan lama)
A = apps()
BU = load("bureau_observations")
cur_ids = set(A.application_id)
BU["is_cur"] = BU.application_id.isin(cur_ids)
num = ["score", "active_accounts", "outstanding_balance", "inquiries_last_6m", "delinquent_accounts_last_12m", "utilization",
       "revolving_credit_limit"]
cur = BU[BU.is_cur].groupby("customer_id")[num].mean()
prev = BU[~BU.is_cur].sort_values("report_datetime").groupby("customer_id")[num].agg(["mean", "last"])
D = pd.DataFrame(index=cur.index)
for c in num:
    D[f"d_{c}_vs_mean"] = cur[c] - prev[(c, "mean")]
    D[f"d_{c}_vs_last"] = cur[c] - prev[(c, "last")]
D = D.join(X[["target", "r", "risk_q"]], how="inner").dropna(subset=["d_score_vs_mean"])
rows = []
for c in [c for c in D.columns if c.startswith("d_")]:
    rows.append(dict(fitur=c, n=int(D[c].notna().sum()), z=round(zr(D[c].values.astype(float), D.r.values), 2),
                     mean_y1=round(D.loc[D.target == 1, c].mean(), 3), mean_y0=round(D.loc[D.target == 0, c].mean(), 3)))
print("ramp biro (sekarang - lama):\n", pd.DataFrame(rows).sort_values("z", key=np.abs, ascending=False).to_string(index=False))
# 4. aplikasi lama menjelang keputusan (credit hunger)
PA = load("previous_applications").merge(A[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"}), on="customer_id")
PA["age_d"] = (PA.cut - PA.application_datetime).dt.total_seconds() / 86400
for w in (14, 30, 60, 90):
    s = PA[PA.age_d <= w].groupby("customer_id").size()
    x = X.index.to_series().map(s).fillna(0).values.astype(float)
    print(f"jumlah aplikasi lama {w} hari terakhir: z residual {zr(x, X.r.values):.2f}, rata y1 {x[X.target.values == 1].mean():.3f} vs y0 {x[X.target.values == 0].mean():.3f}")
