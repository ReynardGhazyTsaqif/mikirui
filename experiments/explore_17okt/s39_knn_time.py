"""Struktur yang belum diuji: (1) kemiripan lokal antar customer (H20 KNN) dan (2) pengelompokan waktu keputusan.

(1) 30 fitur terpenting (rank-uniform, dari s38) -> untuk baris fold 3-4, rata-rata residual OOF E202 dari 50 tetangga
    terdekat di fold 0-2 (dan sebaliknya). Jika residual tetangga memprediksi residual sendiri -> ada struktur lokal
    yang terlewat (seperti tipe/arketipe customer).
(2) Residual menurut tanggal/minggu keputusan: ragam rata-rata residual per minggu vs harapan binomial, dan korelasi
    residual sendiri dengan rata-rata residual 50 customer terdekat waktunya (hanya baris fold lain).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from common import apps  # noqa: E402
from resid import reference  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load_apps  # noqa: E402

ref = reference("E202")
r, fold, y = ref.r.values, ref.fold.values, ref.target.values
TOP = ["seqx_rp4", "seq_bigru2", "seq_gru1", "seq_tf3", "rp_ev_miss_post", "x_rate_pred", "ecm_last100",
       "bu_score_fresh_weighted", "seqx_pa6", "x_rate_x_score"]
A2 = load_apps()
m = A2.is_test.eq(0).values
X = run.build_matrix(list(GROUPS), A2)
E = pd.read_parquet(FEATURES_DIR / "rp_events_v1.parquet").reindex(A2.application_id)
X = pd.concat([X, E], axis=1)[m].reset_index(drop=True)
USE = ["e121_ordermodel_v1.npz", "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",
       "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz"]
for i, f in enumerate(USE):
    z = np.load(FEATURES_DIR / f, allow_pickle=True)
    arr = np.full((len(X), len(z["cols"])), np.nan, dtype="float32")
    for k in range(5):
        arr[fold == k] = z[f"va{k}"]
    for j, c in enumerate(z["cols"]):
        X[f"{c}{i}"] = arr[:, j]
s38 = pd.read_csv(HERE / "s38_auto_fe.csv")  # urutan basis tidak disimpan -> pakai 10 teratas + fitur kuat lain
extra = ["rp_ev_q0_post", "ecm_mean0", "ecm_last30", "pa_active_months_left_sum", "ec_days_since_last_order",
         "bu_score_mean", "app_annual_interest_rate", "rp_ev_catch_rate", "rp_ev_pd_struct", "seqx_ab5",
         "app_lti", "prof_declared_monthly_income", "ab_all_inst_vs_income", "ec_share_night", "rp_ev_miss_rate_12m",
         "pa_n_prev_apps", "bu_utilization_mean", "x_total_pti", "prof_age", "rp_max_dpd_max"]
cols = [c for c in TOP + extra if c in X.columns]
print("fitur KNN:", len(cols), [c for c in TOP + extra if c not in X.columns])
Z = np.column_stack([rankdata(np.nan_to_num(X[c].values, nan=-1e9)) / len(X) for c in cols])


def knn_resid(q, b, k=50):
    nn = NearestNeighbors(n_neighbors=k).fit(Z[b])
    _, idx = nn.kneighbors(Z[q])
    return r[b][idx].mean(1)


out = np.full(len(r), np.nan)
q1, b1 = np.isin(fold, [3, 4]), np.isin(fold, [0, 1, 2])
out[q1] = knn_resid(q1, b1)
out[b1] = knn_resid(b1, q1)
for lab, msk in [("fold 3-4 (tetangga dari 0-2)", q1), ("fold 0-2 (tetangga dari 3-4)", b1)]:
    rho = np.corrcoef(out[msk], r[msk])[0, 1]
    print(f"KNN residual {lab}: korelasi {rho:+.4f}, z {rho * np.sqrt(msk.sum()):+.2f}")
# (2) waktu keputusan
A = apps()
t = A[A.is_test == 0].decision_datetime.reset_index(drop=True)
wk = t.dt.to_period("W").astype(str).values
g = pd.DataFrame({"wk": wk, "r": r, "v": ref.p_cal.values * (1 - ref.p_cal.values)}).groupby("wk")
mean_r, n_w, v_w = g.r.mean(), g.size(), g.v.mean()
chi = (mean_r ** 2 / (v_w / n_w)).sum()
print(f"minggu {len(n_w)}: chi2 rata-rata residual per minggu {chi:.1f} vs harapan {len(n_w)} (sd {np.sqrt(2 * len(n_w)):.1f})")
d = pd.DataFrame({"r": r, "p": ref.p_cal.values * (1 - ref.p_cal.values)}).groupby(t.dt.date.values)
chi_d = (d.r.mean() ** 2 / (d.p.mean() / d.size())).sum()
print(f"hari {d.ngroups}: chi2 {chi_d:.1f} vs harapan {d.ngroups} (sd {np.sqrt(2 * d.ngroups):.1f})")
o = np.argsort(t.values)
rs, fs = r[o], fold[o]
nb = np.full(len(r), np.nan)
for i in range(len(rs)):
    lo, hi = max(0, i - 60), min(len(rs), i + 61)
    w = [j for j in range(lo, hi) if fs[j] != fs[i]][:50]
    nb[i] = rs[w].mean()
rho = np.corrcoef(nb, rs)[0, 1]
print(f"residual vs rata-rata residual 50 customer terdekat waktunya (fold lain): korelasi {rho:+.4f}, z {rho * np.sqrt(len(rs)):+.2f}")
