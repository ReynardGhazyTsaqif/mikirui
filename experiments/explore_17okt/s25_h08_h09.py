"""Bukti H08 & H09 (tanpa target) -> features/cand_h08h09_v1.parquet untuk screen di atas OOF E200.

H08: pinjaman yang masih berjalan di cutoff: peluang 'akan default dalam 12 bulan' diberi (event ke-m, tunggakan A saat ini)
     dari tabel empiris pinjaman lama LENGKAP (berapa yang akhirnya default dari keadaan (m, A) yang sama). Tanpa target.
H09: status bulan kalender 1-3 terakhir lintas semua pinjaman: jatuh tempo, lewat-bayar, tunggakan total.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.events import _events  # noqa: E402
from src.io_utils import load, load_apps  # noqa: E402

apps = load_apps()
ev = _events(apps)
rp = load("repayment_history", usecols=["loan_id", "due_date", "max_days_past_due"])
pa = load("previous_applications", usecols=["loan_id", "decision_datetime"]).dropna()
r = rp.merge(pa, on="loan_id")
elig = (r.due_date + pd.Timedelta(days=90)) <= (r.decision_datetime + pd.DateOffset(months=12))
lab = (elig & (r.max_days_past_due >= 90)).groupby(r.loan_id).any()
L = ev.groupby("loan_id").agg(dis=("dis", "first"), cutoff=("cutoff", "first"), customer_id=("customer_id", "first"),
                              J=("j", "max"))
L["complete"] = (L.dis + pd.DateOffset(months=12)) <= L.cutoff
L["def"] = lab.reindex(L.index).fillna(False).astype(int)
# tabel hazard empiris dari pinjaman lengkap: keadaan (m = event ke-m <= 9, A setelah event m) -> P(default akhirnya)
e = ev[ev.j <= 9].merge(L[["complete", "def"]], left_on="loan_id", right_index=True)
ec = e[e.complete]
tab = ec.groupby(["j", ec.A.clip(upper=3)])["def"].agg(["mean", "size"])
print("P(default | event m, tunggakan A) dari pinjaman lengkap (baris m, kolom A):")
print(tab["mean"].unstack().round(3).to_string())
# pinjaman berjalan (belum lengkap) di cutoff: keadaan terakhirnya
run_loans = L[~L.complete]
last = ev[ev.loan_id.isin(run_loans.index)].sort_values("j").groupby("loan_id").tail(1)
last = last[last.j <= 9]
key = list(zip(last.j, last.A.clip(upper=3)))
pmap = tab["mean"].to_dict()
base = ec.groupby("loan_id")["def"].first().mean()
last["p_def"] = [pmap.get(k, np.nan) for k in key]
g = last.groupby("customer_id")
F = pd.DataFrame({"h08_pdef_running_max": g.p_def.max(), "h08_pdef_running_sum": g.p_def.sum(),
                  "h08_n_running_inwindow": g.size()})
# H09: bulan kalender terakhir
for w in (1, 2, 3):
    x = ev[ev.m_before < w]
    gg = x.groupby("customer_id")
    F[f"h09_due_{w}m"] = gg.size()
    F[f"h09_miss_{w}m"] = gg.miss.sum()
    F[f"h09_allmiss_{w}m"] = (gg.miss.mean() == 1).astype(float)
lastA = ev.sort_values("j").groupby("loan_id").tail(1)
F["h09_arrears_now_total"] = lastA[lastA.m_before <= 1].groupby("customer_id").A.sum()
out = apps[["application_id", "customer_id"]].merge(F, left_on="customer_id", right_index=True, how="left")
out = out.drop(columns="customer_id").set_index("application_id").astype("float32")
out[sorted(out.columns)].to_parquet(ROOT / "features" / "cand_h08h09_v1.parquet")
print(out.describe().T.round(3).to_string())
