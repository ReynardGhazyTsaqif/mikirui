import sys
sys.path.insert(0, r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from src.io_utils import load, load_apps
apps = load_apps()
tr = apps[apps.is_test == 0].reset_index(drop=True)
prof = load("customer_profiles")
tr = tr.merge(prof, on="customer_id", how="left")
y = tr.target.values
oof = np.load(r"D:\Unand\Sems 7\Lomba\DSC\dsc-logika\oof\T_lgb_b3_4f3c7f.npy")
tr["res"] = y - oof
tr["ym"] = tr.decision_datetime.dt.to_period("M").astype(str)
tr["q"] = tr.decision_datetime.dt.to_period("Q").astype(str)
print("target & residual per bulan keputusan:")
print(tr.groupby("ym").agg(n=("target", "size"), rate=("target", "mean"), res=("res", "mean")).round(3).T.to_string())
for c in ["region", "employment_type", "loan_purpose", "housing_status"]:
    t = tr.pivot_table(index=c, columns="q", values="target", aggfunc="mean").round(3)
    r = tr.pivot_table(index=c, columns="q", values="res", aggfunc="mean").round(3)
    print(f"\n== {c} x kuartal: target rate"); print(t.to_string())
    print(f"   residual (y - oof):"); print(r.to_string())
print("\nstd residual sel region x bulan:", tr.groupby(["region", "ym"]).res.mean().std().round(4),
      "| std yg diharapkan krn noise ~", round(np.sqrt((oof * (1 - oof)).mean() / (len(tr) / (5 * 13))), 4))
