"""Langkah 2n: posterior risiko default per customer dari hasil pinjaman lama (definisi target persis + sensor).

Asumsi kerja (diuji di s14): tiap pinjaman customer c default ~ Bernoulli(pi_c), pi_c ~ Beta(a, b).
Label pinjaman = A >= 3 pada event 1..9 (DPD 90 dalam 12 bulan).
- pinjaman lengkap (>= 9 event atau jendela 365 hari lewat): observasi 0/1
- pinjaman belum lengkap dengan A >= 3 sudah terjadi: observasi 1
- pinjaman belum lengkap tanpa default setelah m event: likelihood pi*S1(m) + (1 - pi), S1(m) = P(belum hit | flagged)
- pinjaman baru tanpa cicilan (fresh): tidak menambah likelihood
Posterior mean dihitung dengan integrasi grid. Fitur: posterior mean, log-odds, n efektif.
"""
import numpy as np
import pandas as pd

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 220)
A = apps()
EV = pd.read_parquet("cache_events_full.parquet")
L = EV.groupby("loan_id").agg(customer_id=("customer_id", "first"), dis=("dis", "first"), cutoff=("cutoff", "first"),
                              term=("term", "first"), m=("j", "max"))
hit = EV[(EV.j <= 9) & (EV.A >= 3)].groupby("loan_id").j.min()
L["hit_j"] = hit.reindex(L.index)
L["complete"] = (L.dis + pd.Timedelta(days=365)) <= L.cutoff
L["m9"] = np.minimum(L.m, 9)
L["state"] = np.where(L.hit_j.notna(), "default", np.where(L.complete | (L.m >= 9), "ok", "censored"))
print(L.state.value_counts().to_dict())
# S1(m): di antara pinjaman lengkap yang default, P(hit_j > m)
dl = L[L.complete & L.hit_j.notna()]
S1 = np.array([(dl.hit_j > mm).mean() for mm in range(0, 10)])
print("S1(m) = P(belum hit setelah m event | default):", np.round(S1, 3).tolist())
print("distribusi event hit pertama (pinjaman lengkap default):", dl.hit_j.value_counts().sort_index().to_dict())
# prior dari s14
a, b = 0.186, 1.330
grid = np.linspace(0.0005, 0.9995, 400)
lp_prior = (a - 1) * np.log(grid) + (b - 1) * np.log(1 - grid)
cust = L.groupby("customer_id")
logl = {}
for st, val in [("default", None), ("ok", None)]:
    pass
nd = L[L.state == "default"].groupby("customer_id").size()
no = L[L.state == "ok"].groupby("customer_id").size()
cens = L[L.state == "censored"]
cust_ids = L.customer_id.unique()
ll = np.zeros((len(cust_ids), len(grid)))
cidx = pd.Series(np.arange(len(cust_ids)), index=cust_ids)
ll += np.outer(nd.reindex(cust_ids).fillna(0).values, np.log(grid))
ll += np.outer(no.reindex(cust_ids).fillna(0).values, np.log(1 - grid))
for mm in range(1, 9):
    cnt = cens[cens.m9 == mm].groupby("customer_id").size().reindex(cust_ids).fillna(0).values
    ll += np.outer(cnt, np.log(grid * S1[mm] + (1 - grid)))
post = ll + lp_prior
post -= post.max(axis=1, keepdims=True)
w = np.exp(post)
w /= w.sum(axis=1, keepdims=True)
pm = w @ grid
F = pd.DataFrame(index=cust_ids)
F["lo_post_mean"] = pm
F["lo_post_logit"] = np.log(pm / (1 - pm))
F["lo_n_default"] = nd.reindex(cust_ids).fillna(0)
F["lo_n_ok"] = no.reindex(cust_ids).fillna(0)
F["lo_n_cens"] = cens.groupby("customer_id").size().reindex(cust_ids).fillna(0)
F["lo_share_default_complete"] = (F.lo_n_default / (F.lo_n_default + F.lo_n_ok)).where(F.lo_n_default + F.lo_n_ok > 0)
F.to_parquet("cache_loan_posterior.parquet")
ref = reference("E122x1")
from common import MAIN
old = pd.read_parquet(MAIN / "features" / "rp_loan_v1.parquet")[["rp_n_loans_ever90", "rp_share_loans_ever90"]]
ids = pd.read_csv(MAIN / "Dataset" / "train_applications.csv", usecols=["application_id", "customer_id"]).set_index("application_id").customer_id
old = old[old.index.isin(ids.index)]
old.index = ids.reindex(old.index).values
feats = {c: F[c] for c in F.columns}
feats.update({c: old[c] for c in old.columns})
print(table(ref, feats).to_string(index=False))
j = ref.set_index("customer_id").join(F, how="inner")
j["dec"] = pd.qcut(j.lo_post_mean.rank(method="first"), 10, labels=False)
print("desil posterior: rata posterior, target rate, rata p model")
print(j.groupby("dec").agg(post=("lo_post_mean", "mean"), target=("target", "mean"), p=("p", "mean")).round(4).T.to_string())
