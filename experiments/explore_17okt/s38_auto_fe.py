"""Masukan user 18 Okt butir 3: pencarian fitur otomatis ala OpenFE dengan validasi silang anti uji-berganda.

Basis: 80 fitur terpenting (gain LightGBM cepat di set E200 versi OOF: 528 + rp_ev + fitur nested va{k}).
Kandidat: semua pasangan (a/b, rank(a) - rank(b), (rank(a)-0,5)(rank(b)-0,5)) + transformasi kelompok
(a - rata-rata a per kelompok, peringkat a di dalam kelompok) untuk kelompok tier, wilayah, pekerjaan, rumah, tujuan,
punya riwayat, desil umur, desil income. Skor = z residual OOF E202 (Pearson rank vs residual x sqrt(n)), dihitung
TERPISAH di fold 0-2 (seleksi) dan fold 3-4 (konfirmasi). Lolos: |z_sel| >= 4 dan z_val searah dengan |z_val| >= 2,5.
Keluaran: experiments/explore_17okt/s38_auto_fe.csv (semua kandidat).
"""
import gc
import itertools
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from resid import reference  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load, load_apps  # noqa: E402

USE = ["e121_ordermodel_v1.npz", "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",
       "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz"]
apps = load_apps()
m = apps.is_test.eq(0).values
X = run.build_matrix(list(GROUPS), apps)
E = pd.read_parquet(FEATURES_DIR / "rp_events_v1.parquet").reindex(apps.application_id)
X = pd.concat([X, E], axis=1)[m]
tr_apps = apps[m].reset_index(drop=True)
del E
ref = reference("E202")
assert (ref.application_id.values == X.index.values).all()
X = X.reset_index(drop=True)
fold = ref.fold.values
for i, f in enumerate(USE):  # fitur nested versi OOF
    z = np.load(FEATURES_DIR / f, allow_pickle=True)
    arr = np.full((len(X), len(z["cols"])), np.nan, dtype="float32")
    for k in range(5):
        arr[fold == k] = z[f"va{k}"]
    for j, c in enumerate(z["cols"]):
        X[f"{c}{i}"] = arr[:, j]
cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
num = [c for c in X.columns if c not in cat_cols]
y, r = ref.target.values, ref.r.values
p = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=4, force_row_wise=True)
mdl = lgb.train(p, lgb.Dataset(X[num].astype("float32"), y), 300)
imp = pd.Series(mdl.feature_importance("gain"), index=num).sort_values(ascending=False)
base = list(imp.index[:80])
print("80 fitur basis (10 teratas):", base[:10], flush=True)
B = X[base].astype("float64")
del mdl, X
gc.collect()
sel, val = np.isin(fold, [0, 1, 2]), np.isin(fold, [3, 4])


def zscore(x):
    out = []
    for msk in (sel, val):
        ok = msk & np.isfinite(x)
        if ok.sum() < 2000 or np.nanstd(x[ok]) == 0:
            out.append(np.nan)
            continue
        rx = rankdata(x[ok])
        rho = np.corrcoef(rx, r[ok])[0, 1]
        out.append(rho * np.sqrt(ok.sum()))
    return out


R = {c: rankdata(B[c].values, nan_policy="omit") / np.isfinite(B[c].values).sum() for c in base}
rows = []
for a, b in itertools.combinations(base, 2):
    va, vb = B[a].values, B[b].values
    with np.errstate(divide="ignore", invalid="ignore"):
        cands = {f"{a} / {b}": np.where(vb != 0, va / vb, np.nan),
                 f"rank({a}) - rank({b})": R[a] - R[b],
                 f"(rank({a})-.5)*(rank({b})-.5)": (R[a] - 0.5) * (R[b] - 0.5)}
    for name, x in cands.items():
        zs, zv = zscore(x)
        rows.append((name, zs, zv))
prof = load("customer_profiles").set_index("customer_id")
cid = tr_apps.customer_id
grp = {"tier": tr_apps.annual_interest_rate.values, "region": cid.map(prof.region).astype(str).values,
       "emp": cid.map(prof.employment_type).astype(str).values, "house": cid.map(prof.housing_status).astype(str).values,
       "purpose": tr_apps.loan_purpose.astype(str).values,
       "hist": B[[c for c in base if c.startswith("pa_")][0]].notna().values if any(c.startswith("pa_") for c in base) else np.zeros(len(B)),
       "age10": pd.qcut(cid.map(prof.age), 10, labels=False, duplicates="drop").values,
       "inc10": pd.qcut(cid.map(prof.declared_monthly_income), 10, labels=False, duplicates="drop").values}
for a in base:
    s = pd.Series(B[a].values)
    for gname, g in grp.items():
        gm = s.groupby(g).transform("mean")
        gr = s.groupby(g).rank(pct=True)
        for name, x in {f"{a} - mean|{gname}": (s - gm).values, f"rank({a})|{gname}": gr.values}.items():
            zs, zv = zscore(x)
            rows.append((name, zs, zv))
T = pd.DataFrame(rows, columns=["kandidat", "z_sel", "z_val"])
T["lolos"] = (T.z_sel.abs() >= 4) & (np.sign(T.z_sel) == np.sign(T.z_val)) & (T.z_val.abs() >= 2.5)
T.to_csv(HERE / "s38_auto_fe.csv", index=False)
print(f"kandidat {len(T)}; |z_sel| >= 4: {(T.z_sel.abs() >= 4).sum()}; |z_sel| >= 3: {(T.z_sel.abs() >= 3).sum()}; lolos konfirmasi: {T.lolos.sum()}")
print("sebaran z_sel (harapan nol: sd 1):", T.z_sel.describe().round(3).to_dict())
print(T.reindex(T.z_sel.abs().sort_values(ascending=False).index).head(25).round(2).to_string(index=False))
