"""E120 langkah 1: di segmen mana model (E122x1, bag 3 seed) paling lemah dan paling banyak menyumbang kesalahan."""
import glob, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
meta = pd.read_parquet("kaggle/dataset/apps_meta.parquet"); tr = meta[meta.is_test == 0].reset_index(drop=True)
X = pd.read_parquet("kaggle/dataset/X_all.parquet"); X = X[meta.is_test.values == 0].reset_index(drop=True)
p = np.mean([rankdata(np.load(f)) for f in sorted(glob.glob("oof/E122x1_s4*.npy"))], 0) / len(tr)
y = tr.target.values.astype(int)
seg = pd.DataFrame({
    "riwayat": np.where(X["rp_hist_len_days"].notna() if "rp_hist_len_days" in X else X.filter(like="rp_").notna().any(axis=1), "ada cicilan", "tanpa cicilan"),
    "rate": X["app_annual_interest_rate"].round(2).astype(str),
    "ec_recency": pd.cut(X.filter(like="days_since_last_order").iloc[:, 0], [-1, 2, 7, 30, 90, 1e9], labels=["<=2h", "3-7h", "8-30h", "31-90h", ">90h"]).astype(str),
    "rekening": np.where(X.filter(like="ab_").notna().any(axis=1), "ada", "tidak"),
})
print("kolom recency e-commerce:", X.filter(like="days_since_last_order").columns[0])
print(f"keseluruhan: n {len(y)}, rate {y.mean():.3f}, AUC {roc_auc_score(y, p):.4f}")
for c in seg:
    print(f"\n== {c}")
    for v, idx in seg.groupby(c).groups.items():
        yy, pp = y[idx], p[idx]
        auc = roc_auc_score(yy, pp) if 0 < yy.mean() < 1 else np.nan
        # kontribusi kesalahan: defaulter dengan skor di bawah median global (FN berat) dan non-default di atas persentil 90 (FP berat)
        fn = ((yy == 1) & (pp < 0.5)).sum(); fp = ((yy == 0) & (pp > 0.9)).sum()
        print(f"  {v:14s} n {len(idx):6d} rate {yy.mean():.3f} AUC dalam segmen {auc:.4f} | defaulter skor<median {fn:5d} ({fn / max(1, (yy == 1).sum()):.0%} dari defaulter segmen)")
print(f"\ndefaulter dengan skor di bawah median: {((y == 1) & (p < .5)).sum()} dari {y.sum()} ({((y == 1) & (p < .5)).sum() / y.sum():.1%})")
np.save("experiments/e120/oof_rank.npy", p)
