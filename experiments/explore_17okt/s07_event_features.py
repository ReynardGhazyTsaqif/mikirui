"""Langkah 2g: statistik proses bayar level event per customer vs fitur lama vs residual E122x1."""
import numpy as np
import pandas as pd

from common import MAIN
from resid import reference, table

pd.set_option("display.width", 220)
EV = pd.read_parquet("cache_events_full.parquet")
EV["catch"] = ((EV.A_prev >= 1) & (EV.X >= 2)).astype("int8")
EV["behind"] = (EV.A_prev >= 1).astype("int8")
EV["miss0"] = ((EV.A_prev == 0) & (EV.X == 0)).astype("int8")
EV["at0"] = (EV.A_prev == 0).astype("int8")
g = EV.groupby("customer_id")
S = pd.DataFrame({
    "n_ev": g.size(), "n_miss": g.miss.sum(),
    "n_at0": g.at0.sum(), "n_miss0": g.miss0.sum(),
    "n_behind": g.behind.sum(), "n_catch": g.catch.sum(),
    "maxA": g.A.max(),
})
S["miss_rate"] = S.n_miss / S.n_ev
S["miss0_rate"] = S.n_miss0 / S.n_at0.replace(0, np.nan)
S["catch_rate"] = S.n_catch / S.n_behind.replace(0, np.nan)
# shrinkage beta (method of moments pada customer dengan >= 6 event)
m = S[S.n_ev >= 6]
mu, var = m.miss_rate.mean(), m.miss_rate.var()
k = mu * (1 - mu) / var - 1
a0, b0 = mu * k, (1 - mu) * k
print(f"prior beta miss: a={a0:.2f} b={b0:.2f} (mean {mu:.3f})")
S["miss_post"] = (S.n_miss + a0) / (S.n_ev + a0 + b0)
m0 = S[S.n_at0 >= 6].miss0_rate
mu0, var0 = m0.mean(), m0.var()
k0 = mu0 * (1 - mu0) / var0 - 1
S["miss0_post"] = (S.n_miss0 + mu0 * k0) / (S.n_at0 + k0)
# status terakhir (event terakhir per pinjaman yang masih punya kewajiban) dan event kalender terbaru
last = EV.sort_values(["customer_id", "cal", "loan_id"]).groupby("customer_id").tail(1).set_index("customer_id")
S["last_m_before"] = last.m_before
S["A_now_sum"] = EV.loc[EV.groupby("loan_id").j.idxmax()].groupby("customer_id").A.sum()
for w in (3, 6, 12):
    e = EV[EV.m_before < w]
    S[f"miss_rate_{w}m"] = e.groupby("customer_id").miss.mean()
    S[f"n_ev_{w}m"] = e.groupby("customer_id").size()
S.to_parquet("cache_cust_event_stats.parquet")
ref = reference("E122x1")
print("acuan E122x1 bag", ref.attrs["n_seed"], "seed; AUC pooled", round(__import__("sklearn.metrics").metrics.roc_auc_score(ref.target, ref.p), 5))
old = pd.read_parquet(MAIN / "features" / "rp_core_v1.parquet")[["rp_share_dpd1", "rp_n_dpd1", "rp_share_dpd30"]]
app2c = pd.read_csv(MAIN / "Dataset" / "train_applications.csv", usecols=["application_id", "customer_id"]).set_index("application_id").customer_id
old = old[old.index.isin(app2c.index)]
old.index = app2c.reindex(old.index).values
feats = {c: S[c] for c in ["miss_rate", "miss_post", "miss0_rate", "miss0_post", "catch_rate", "maxA", "n_ev", "A_now_sum",
                            "miss_rate_3m", "miss_rate_6m", "miss_rate_12m"]}
feats.update({c: old[c] for c in old.columns})
print(table(ref, feats).to_string(index=False))
# korelasi miss_post dengan fitur lama
j = pd.concat([S.miss_post, old.rp_share_dpd1], axis=1).dropna()
print("Spearman miss_post ~ rp_share_dpd1:", j.corr("spearman").iloc[0, 1].round(4))
