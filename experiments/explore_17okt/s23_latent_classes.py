"""Ide eksternal 5: apakah customer berasal dari kelas laten diskrit? GMM tanpa target pada besaran pembangkit kunci.

Besaran per customer (semua 100 rb, tanpa target): logit q0 (rantai bayar, hanya yang punya riwayat), skor biro rata-rata,
porsi COD e-commerce, log nominal order rata-rata, LLR waktu order rata-rata (kurva global dari s09, tanpa target per fold:
di sini memakai bentuk densitas waktu order semua customer, bukan label), tier bunga sekarang.
Uji: BIC 1..6 komponen; posterior GMM terbaik sebagai fitur -> z residual.
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import apps, load
from resid import reference, table

A = apps().set_index("customer_id")
ST = pd.read_parquet("cache_struct.parquet")
od = pd.read_parquet("cache_ec_orders.parquet")
BU = load("bureau_observations")
bu = BU[BU.application_id.isin(set(A.application_id))].groupby("customer_id").score.mean()
g = od.groupby("customer_id")
X = pd.DataFrame(index=A.index)
X["bu"] = bu.reindex(X.index)
X["cod"] = (od.pm == "COD").groupby(od.customer_id).mean().reindex(X.index)
X["lamt"] = np.log(od.total).groupby(od.customer_id).mean().reindex(X.index)
X["rec_med"] = np.log1p(g.rec_h.median()).reindex(X.index)
X["tier"] = A.annual_interest_rate
X["q0"] = np.log(ST.q0 / (1 - ST.q0)).reindex(X.index)
ref = reference("E122x1")
F = {}
for name, cols in [("hist", ["bu", "cod", "lamt", "rec_med", "tier", "q0"]), ("all", ["bu", "cod", "lamt", "rec_med", "tier"])]:
    D = X[cols].dropna()
    Z = StandardScaler().fit_transform(D)
    bics = []
    for k in range(1, 7):
        gm = GaussianMixture(k, covariance_type="full", random_state=42, n_init=2).fit(Z)
        bics.append(gm.bic(Z))
    kbest = int(np.argmin(bics)) + 1
    print(f"{name}: n {len(D)}, BIC per k=1..6:", np.round(np.array(bics) - min(bics), 0).tolist(), "-> k terbaik", kbest)
    gm = GaussianMixture(max(kbest, 2), covariance_type="full", random_state=42, n_init=2).fit(Z)
    post = gm.predict_proba(Z)
    for j in range(post.shape[1]):
        F[f"gmm_{name}_p{j}"] = pd.Series(post[:, j], index=D.index)
    F[f"gmm_{name}_entropy"] = pd.Series(-(post * np.log(post + 1e-12)).sum(1), index=D.index)
print(table(ref, F).to_string(index=False))
