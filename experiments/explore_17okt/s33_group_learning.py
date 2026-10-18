"""Putaran 2: di grup sumber mana model masih "belajar lambat"? (penunjuk arah rekayasa representasi)

Untuk tiap grup G (set E200), kontribusi D_G(n) = AUC(semua, n) - AUC(tanpa G, n) diukur pada 50% dan 100% data latih
fold (28k / 56k baris; subsampel, parameter, dan seed sama dengan s29 sehingga AUC "semua" diambil dari
learning_curve_e200.csv). Grup dengan D_G(56k) - D_G(28k) besar = model masih menambah cara memakai grup itu
seiring data bertambah -> representasi yang lebih baik untuk grup itu bisa setara "data gratis" (seperti rp_ev).
Noise 1 seed: selisih ganda +-0,0007 -> hanya efek besar yang bermakna.
Keluaran: experiments/explore_17okt/group_learning.csv
"""
import gc
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.features.registry import GROUPS  # noqa: E402
from src.io_utils import load_apps  # noqa: E402
from src.train import make_folds  # noqa: E402

USE = ["e121_ordermodel_v1.npz", "e122_gru_v1r.npz", "e122_gru_v1_bigrur.npz", "e122_gru_v1_tfr.npz",
       "e122x_rp_v1r.npz", "e122x_ab_v1r.npz", "e122x_pa_v1r.npz"]
NESTED = {"ecm": [0], "seq_ec": [1, 2, 3], "seqx": [4, 5, 6]}  # indeks file USE per grup nested
BASE = {"rp": ("rp_",), "rp_ev": ("rp_ev_",), "bu": ("bu_", "bp_"), "pa": ("pa_",), "ab": ("ab_",), "ec": ("ec_",),
        "app": ("app_", "prof_"), "x": ("x_",)}
apps = load_apps()
m = apps.is_test.eq(0).values
X = run.build_matrix(list(GROUPS), apps)
E = pd.read_parquet(FEATURES_DIR / "rp_events_v1.parquet").reindex(apps.application_id)
X = pd.concat([X.set_index(apps.application_id), E], axis=1)[m].reset_index(drop=True)
y = apps.loc[m, "target"].astype(int).values
fold = make_folds(apps[m])["fold"].values
del E, apps
gc.collect()
zs = [np.load(FEATURES_DIR / f, allow_pickle=True) for f in USE]
cols = [f"{c}{i}" for i, z_ in enumerate(zs) for c in z_["cols"]]
src = [i for i, z_ in enumerate(zs) for _ in z_["cols"]]
z = {key: np.hstack([z_[key] for z_ in zs]) for key in zs[0].files if key not in ("cols", "ids_train", "ids_test")}
groups = {g: [c for c in X.columns if c.startswith(pre) and not (g == "rp" and c.startswith("rp_ev_"))]
          for g, pre in BASE.items()}
groups.update({g: [c for c, s in zip(cols, src) if s in idx] for g, idx in NESTED.items()})
params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 35, "max_depth": 4,
          "min_child_samples": 34, "feature_fraction": 0.12870618888259533, "bagging_fraction": 0.9264683082866251,
          "bagging_freq": 1, "lambda_l2": 7.664965525069028, "lambda_l1": 0.011885886537829151,
          "min_gain_to_split": 0.9382501776308856, "extra_trees": True, "max_bin": 255, "seed": 42,
          "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1}
lc = pd.read_csv(HERE / "learning_curve_e200.csv").set_index(["fold", "frac"]).auc
rows = []
for g, gcols in groups.items():
    print(f"grup {g}: {len(gcols)} kolom", flush=True)
    for k in range(5):
        tr, va = fold != k, fold == k
        Atr = pd.concat([X[tr].reset_index(drop=True), pd.DataFrame(z[f"tr{k}"], columns=cols)], axis=1).drop(columns=gcols)
        Bva = pd.concat([X[va].reset_index(drop=True), pd.DataFrame(z[f"va{k}"], columns=cols)], axis=1).drop(columns=gcols)
        ytr, yva = y[tr], y[va]
        perm = np.random.default_rng(42 + k).permutation(len(ytr))  # sama dengan s29
        for frac in (0.5, 1.0):
            idx = np.sort(perm[: int(len(ytr) * frac)])
            dtr = lgb.Dataset(Atr.iloc[idx], ytr[idx], params={"max_bin": 255})
            dva = lgb.Dataset(Bva, yva, reference=dtr)
            mdl = lgb.train(params, dtr, 8000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            auc = roc_auc_score(yva, mdl.predict(Bva, num_iteration=mdl.best_iteration))
            rows.append(dict(group=g, fold=k, frac=frac, auc_wo=auc, auc_all=lc[(k, frac)], d=lc[(k, frac)] - auc))
            del dtr, dva, mdl
        del Atr, Bva
        gc.collect()
    pd.DataFrame(rows).to_csv(HERE / "group_learning.csv", index=False)
r = pd.DataFrame(rows)
s = r.pivot_table(index="group", columns="frac", values="d", aggfunc="mean")
s.columns = ["D_28k", "D_56k"]
s["tumbuh"] = s.D_56k - s.D_28k
sd = r.pivot_table(index=["group", "fold"], columns="frac", values="d").diff(axis=1)[1.0].groupby("group").std()
s["sd_tumbuh_mean"] = sd / np.sqrt(5)
s["n_kolom"] = pd.Series({g: len(c) for g, c in groups.items()})
print(s.sort_values("tumbuh", ascending=False).round(5).to_string())
