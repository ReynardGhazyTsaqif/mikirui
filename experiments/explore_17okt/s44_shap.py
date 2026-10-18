"""H26 (mentor ide 18): SHAP pada model set F* (E122x1 + rp_ev + ab_nonin), fold 0 sebagai validasi.

1. TreeSHAP bawaan LightGBM (pred_contrib) di 14k baris validasi -> kepentingan global (mean |SHAP|).
2. Bentuk dependence 20 fitur teratas: rata-rata SHAP per 20 kuantil -> ketidakmonotonan (jumlah pergantian arah)
   dan lompatan terbesar antar kuantil bertetangga relatif terhadap rentang (tanda ambang tajam / aturan pembangkit).
3. FN/FP terbesar vs TP/TN pembanding: fitur mana yang paling membedakan dekomposisi SHAP-nya.
4. Proksi interaksi dari struktur pohon (pasangan fitur induk-anak berbeda, dibobot gain; gaya xgbfi).
Keluaran: s44_shap_importance.csv, s44_shap_shape.csv, s44_interactions.csv, s44_shap_values_fold0.npz.
"""
import gc
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load_apps  # noqa: E402
from src.train import make_folds  # noqa: E402

USE = ["e121_ordermodel_v1.npz", "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",
       "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz"]
apps = load_apps()
m = apps.is_test.eq(0).values
X = run.build_matrix(list(GROUPS), apps)
E = pd.concat([pd.read_parquet(FEATURES_DIR / f).reindex(apps.application_id)
               for f in ["rp_events_v1.parquet", "ab_nonin_v1.parquet"]], axis=1)
X = pd.concat([X, E], axis=1)[m].reset_index(drop=True)
y = apps.loc[m, "target"].astype(int).values
fold = make_folds(apps[m])["fold"].values
del E
zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in USE]
cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
k = 0
tr, va = fold != k, fold == k
A_ = pd.concat([X[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1)
B_ = pd.concat([X[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1)
del X
gc.collect()
params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 35, "max_depth": 4,
          "min_child_samples": 34, "feature_fraction": 0.12870618888259533, "bagging_fraction": 0.9264683082866251,
          "bagging_freq": 1, "lambda_l2": 7.664965525069028, "lambda_l1": 0.011885886537829151,
          "min_gain_to_split": 0.9382501776308856, "extra_trees": True, "max_bin": 255, "seed": 42,
          "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1}
dtr = lgb.Dataset(A_, y[tr])
mdl = lgb.train(params, dtr, 8000, valid_sets=[lgb.Dataset(B_, y[va], reference=dtr)],
                callbacks=[lgb.early_stopping(200, verbose=False)])
p = mdl.predict(B_, num_iteration=mdl.best_iteration)
print(f"model fold 0: iter {mdl.best_iteration}, AUC {roc_auc_score(y[va], p):.5f}", flush=True)
S = mdl.predict(B_, num_iteration=mdl.best_iteration, pred_contrib=True)  # kolom terakhir = nilai dasar
names = list(B_.columns)
sv = S[:, :-1]
np.savez_compressed(HERE / "s44_shap_values_fold0.npz", shap=sv.astype("float32"), names=np.array(names),
                    y=y[va], p=p)
imp = pd.DataFrame({"fitur": names, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp.to_csv(HERE / "s44_shap_importance.csv", index=False)
print("20 fitur teratas (mean |SHAP|):")
print(imp.head(20).round(4).to_string(index=False))
# 2. bentuk dependence
rows = []
for f in imp.fitur.head(20):
    x = B_[f].values.astype(float)
    j = names.index(f)
    ok = np.isfinite(x)
    q = pd.qcut(pd.Series(x[ok]).rank(method="first"), 20, labels=False)
    prof = pd.Series(sv[ok, j]).groupby(q.values).mean().values
    d = np.diff(prof)
    turns = int((np.sign(d[1:]) != np.sign(d[:-1])).sum())
    rng = prof.max() - prof.min()
    jump = np.abs(d).max() / rng if rng > 0 else 0
    nan_shap = sv[~ok, j].mean() if (~ok).any() else np.nan
    rows.append(dict(fitur=f, rentang_shap=round(rng, 4), pergantian_arah=turns, lompatan_maks_rel=round(jump, 3),
                     kuantil_lompatan=int(np.abs(d).argmax()), shap_nan=round(nan_shap, 4), porsi_nan=round((~ok).mean(), 3)))
shape = pd.DataFrame(rows)
shape.to_csv(HERE / "s44_shap_shape.csv", index=False)
print("\nbentuk dependence (pergantian arah dari 18 kemungkinan; lompatan maks relatif rentang):")
print(shape.to_string(index=False))
# 3. FN/FP terbesar vs pembanding
yy = y[va]
med, p95 = np.median(p), np.quantile(p, 0.95)
grp = {"FN": (yy == 1) & (p < med), "TN": (yy == 0) & (p < med), "FP": (yy == 0) & (p > p95), "TP": (yy == 1) & (p > p95)}
D = pd.DataFrame({g: sv[msk].mean(0) for g, msk in grp.items()}, index=names)
D["FN_minus_TN"] = D.FN - D.TN
D["FP_minus_TP"] = D.FP - D.TP
print("\njumlah:", {g: int(msk.sum()) for g, msk in grp.items()})
print("fitur dengan selisih SHAP terbesar FN vs TN (skor sama-sama rendah):")
print(D.reindex(D.FN_minus_TN.abs().sort_values(ascending=False).index).head(8)[["FN", "TN", "FN_minus_TN"]].round(4).to_string())
print("fitur dengan selisih SHAP terbesar FP vs TP (skor sama-sama tinggi):")
print(D.reindex(D.FP_minus_TP.abs().sort_values(ascending=False).index).head(8)[["FP", "TP", "FP_minus_TP"]].round(4).to_string())
# 4. proksi interaksi dari struktur pohon
dump = mdl.dump_model(num_iteration=mdl.best_iteration)
fn = dump["feature_names"]
inter = defaultdict(float)


def walk(node, parent_feat):
    if "split_index" not in node:
        return
    f = fn[node["split_feature"]]
    if parent_feat is not None and parent_feat != f:
        inter[tuple(sorted((parent_feat, f)))] += node["split_gain"]
    walk(node["left_child"], f)
    walk(node["right_child"], f)


for t in dump["tree_info"]:
    walk(t["tree_structure"], None)
I = pd.Series(inter).sort_values(ascending=False)
I.rename("gain_pasangan").to_csv(HERE / "s44_interactions.csv")
print("\n15 pasangan interaksi terkuat (gain anak pada induk berbeda fitur):")
print((I.head(15) / I.sum()).round(4).to_string())
