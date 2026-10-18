"""Masukan user 18 Okt butir 2 (lanjutan s36): anomali e-commerce level item/order sebagai pola, bukan dibuang.

Dari cache features/ec_items_v1.parquet (1 baris = 1 item; order = (customer, recency_h, order_total)):
order_total vs jumlah item_total, harga satuan tidak bulat, kuantitas > 1, item bernilai 0, kategori ganda dalam order,
kombinasi status x metode bayar, order batal ber-COD, dsb. Per customer: porsi/jumlah, lalu AUC + z residual E202.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import MAIN
from resid import reference, table

pd.set_option("display.width", 250)
ref = reference("E202")
it = pq.read_table(MAIN / "features" / "ec_items_v1.parquet",
                   columns=["customer_id", "recency_h", "item_total", "quantity", "order_total", "cat", "pm", "st"],
                   read_dictionary=["customer_id"]).to_pandas()
it["oid"] = it.groupby(["customer_id", "recency_h", "order_total"], observed=True, sort=False).ngroup()
od = it.groupby("oid").agg(cid=("customer_id", "first"), tot=("order_total", "first"), s=("item_total", "sum"),
                           n=("item_total", "size"), ncat=("cat", "nunique"), st=("st", "first"), pm=("pm", "first"),
                           rec=("recency_h", "first"), qmax=("quantity", "max"))
print(f"item {len(it)}, order {len(od)}")
od["mismatch"] = (od.s - od.tot).abs() > 0.5
od["dup_cat"] = od.ncat < od.n
print("order_total != jumlah item:", round(od.mismatch.mean(), 5), "| kategori ganda dalam order:", round(od.dup_cat.mean(), 4))
print("status (kode):", od.st.value_counts(normalize=True).round(4).to_dict(), "| metode (kode):", od.pm.value_counts(normalize=True).round(4).to_dict())
it["unit"] = it.item_total / it.quantity.where(it.quantity > 0)
it["unit_frac"] = (it.unit % 1) > 0
it["unit_mod1000"] = (it.unit % 1000) == 0
it["item0"] = it.item_total <= 0
it["q_gt1"] = it.quantity > 1
cid_it = it.customer_id
F = {}
for f in ["unit_frac", "unit_mod1000", "item0", "q_gt1"]:
    F[f"ec_{f}"] = it[f].groupby(cid_it, observed=True).mean()
F["ec_q_max"] = it.quantity.groupby(cid_it, observed=True).max()
F["ec_order_mismatch"] = od.mismatch.groupby(od.cid, observed=True).mean()
F["ec_dup_cat"] = od.dup_cat.groupby(od.cid, observed=True).mean()
F["ec_items_per_order"] = od.n.groupby(od.cid, observed=True).mean()
for s in sorted(od.st.unique()):
    for p in sorted(od.pm.unique()):
        m = (od.st == s) & (od.pm == p)
        if 0.002 < m.mean():
            F[f"ec_st{int(s)}_pm{int(p)}"] = m.groupby(od.cid, observed=True).mean()
w = od.rec <= 24 * 30  # 30 hari terakhir
for s in sorted(od.st.unique()):
    F[f"ec_st{int(s)}_30d"] = ((od.st == s) & w).groupby(od.cid, observed=True).sum()
F["ec_tot_mod10k"] = ((od.tot % 10000) == 0).groupby(od.cid, observed=True).mean()
F["ec_tot_mod100k"] = ((od.tot % 100000) == 0).groupby(od.cid, observed=True).mean()
F = {k: pd.Series(v.values, index=v.index.astype(str)) for k, v in F.items()}
print(table(ref, F).to_string(index=False))
