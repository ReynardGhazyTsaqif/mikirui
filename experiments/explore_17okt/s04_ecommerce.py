"""Langkah 2d: struktur baris e-commerce dan hubungannya dengan waktu aplikasi/keputusan dan target."""
import numpy as np
import pandas as pd

from common import DATA, apps

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
A = apps()
EC = pd.read_csv(DATA / "ecommerce_transactions.csv", parse_dates=["transaction_datetime"],
                 dtype={"product_category": "category", "payment_method": "category", "order_status": "category",
                        "quantity": "int16"})
print("== ecommerce", EC.shape)
print(EC.head(8).to_string())
od = EC.groupby("order_id", sort=False).agg(customer_id=("customer_id", "first"), t=("transaction_datetime", "first"),
                                            t_nuniq=("transaction_datetime", "nunique"), total=("order_total", "first"),
                                            tot_nuniq=("order_total", "nunique"), item_sum=("item_total", "sum"),
                                            n_items=("order_item_id", "size"), pm=("payment_method", "first"),
                                            st=("order_status", "first"), st_nuniq=("order_status", "nunique"))
print("order:", len(od), "| waktu unik per order:", od.t_nuniq.max(), "| total unik per order:", od.tot_nuniq.max(),
      "| status unik:", od.st_nuniq.max(), "| total == jumlah item:", (od.total == od.item_sum).mean())
od = od.reset_index().merge(A[["customer_id", "application_datetime", "decision_datetime", "target", "is_test"]], on="customer_id")
od["rec_h"] = (od.decision_datetime - od.t).dt.total_seconds() / 3600
od["rec_app_h"] = (od.application_datetime - od.t).dt.total_seconds() / 3600
print("recency ke keputusan (jam) q:", od.rec_h.quantile([0, .001, .01, .1, .5, .9, .99, 1]).round(2).tolist())
print("recency ke aplikasi (jam) q:", od.rec_app_h.quantile([0, .001, .01, .1, .5, .9, .99, 1]).round(2).tolist())
print("order setelah application_datetime:", (od.rec_app_h < 0).mean().round(4))
g = od.groupby("customer_id")
c = pd.DataFrame({"n": g.size(), "first_d": g.rec_h.max() / 24, "last_h": g.rec_h.min(), "last_app_h": g.rec_app_h.min(),
                  "target": g.target.first(), "is_test": g.is_test.first()})
print("order per customer q:", c.n.quantile([0, .01, .1, .5, .9, .99, 1]).tolist())
print("rentang riwayat (hari dari order pertama ke keputusan) q:", c.first_d.quantile([0, .01, .1, .5, .9, .99, 1]).round(1).tolist())
print("order terakhir (jam sebelum keputusan) q:", c.last_h.quantile([0, .01, .05, .1, .25, .5, .75, .9, 1]).round(2).tolist())
print("order terakhir (jam sebelum aplikasi) q:", c.last_app_h.quantile([0, .01, .05, .1, .25, .5, .75, .9, 1]).round(2).tolist())
t = c[c.is_test == 0]
print("target per desil n order:", t.groupby(pd.qcut(t.n, 10, duplicates="drop"), observed=True).target.mean().round(3).tolist())
print("target per desil order terakhir sebelum KEPUTUSAN:", t.groupby(pd.qcut(t.last_h, 10), observed=True).target.mean().round(3).tolist())
print("target per desil order terakhir sebelum APLIKASI:", t.groupby(pd.qcut(t.last_app_h, 10), observed=True).target.mean().round(3).tolist())
from sklearn.metrics import roc_auc_score
print("AUC -last_h %.4f | AUC -last_app_h %.4f | AUC n %.4f | AUC -first_d %.4f" % (
    roc_auc_score(t.target, -t.last_h), roc_auc_score(t.target, -t.last_app_h), roc_auc_score(t.target, t.n),
    roc_auc_score(t.target, -t.first_d)))
# laju order relatif waktu: hitung order per hari pada jendela waktu mundur, dipisah per target
bins = [0, 1, 2, 3, 7, 14, 30, 60, 90, 180, 365, 730, 1e9]
tr = od[od.is_test == 0]
tr = tr.assign(b=pd.cut(tr.rec_h / 24, bins, right=False))
cnt = tr.groupby(["b", "target"], observed=True).size().unstack()
ncust = t.target.value_counts()
w = np.diff(bins)[: len(cnt)]
rate = cnt.div(ncust, axis=1).div(w, axis=0)
print("laju order per customer per hari (kolom target 0/1) per jendela hari sebelum keputusan:\n", rate.round(4))
print("jam order:", od.t.dt.hour.value_counts().sort_index().to_dict())
print("status:", od.st.value_counts().to_dict(), "| metode:", od.pm.value_counts().to_dict())
print("cancel rate per target:", tr.groupby("target").st.apply(lambda s: (s == "cancelled").mean()).round(4).to_dict())
EC["unit"] = EC.item_total / EC.quantity.replace(0, np.nan)
print("quantity:", EC.quantity.value_counts().sort_index().head(10).to_dict())
print("harga satuan per kategori q:\n", EC.groupby("product_category", observed=True).unit.quantile([.01, .5, .99]).unstack().round(0))
c.to_parquet("cache_ec_customer.parquet")
od[["order_id", "customer_id", "t", "rec_h", "rec_app_h", "total", "n_items", "pm", "st", "target", "is_test"]].to_parquet("cache_ec_orders.parquet")
