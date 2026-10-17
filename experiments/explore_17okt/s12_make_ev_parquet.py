"""Tulis kandidat fitur proses bayar level event (tanpa target) ke parquet ber-index application_id untuk screen.py."""
import numpy as np
import pandas as pd

from common import apps

A = apps()
S = pd.read_parquet("cache_struct.parquet")
E = pd.read_parquet("cache_cust_event_stats.parquet")
EV = pd.read_parquet("cache_events_full.parquet")
F = pd.DataFrame(index=S.index)
F["ev_q0"], F["ev_q1"], F["ev_c"] = S.q0, S.q1, S.c
F["ev_n0"], F["ev_n1"] = S.n0, S.n1
F["ev_pd_struct"] = S.pd_struct
F["ev_miss_post"] = E.miss_post
F["ev_catch_rate"] = E.catch_rate
F["ev_miss_rate_6m"] = E.miss_rate_6m
F["ev_months_since_last_miss"] = EV[EV.miss == 1].groupby("customer_id").m_before.min()
F["ev_post_term_events"] = (EV.j > EV.term).groupby(EV.customer_id).sum()
out = A[["application_id", "customer_id"]].merge(F, left_on="customer_id", right_index=True, how="left")
out = out.drop(columns="customer_id").set_index("application_id").astype("float32")
out = out[sorted(out.columns)]
out.to_parquet("cand_ev_v1.parquet")
print(out.describe().T.round(3))
