"""E130b: null importance ulang memakai parameter LightGBM tuned (Fase 5) di atas 528 kolom semua grup.

Featset hasil: experiments/featsets/sel_tuned.json (aturan sama dengan E71: gain asli > kuantil 75% null).
Lalu CV memakai `python e121.py --cv --use e121_ordermodel_v1.npz --name E130b --basefs sel_tuned`.
"""
import sys
from argparse import Namespace

import run
from src import selection as sel
from src.config import ROOT
from src.features.registry import GROUPS
from src.io_utils import load_apps
from src.train import Timer
from src.tuning import full_params


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from tune import Ctx, best_tuned
    ctx = Ctx.__new__(Ctx)  # hanya butuh storage Optuna, tanpa memuat data
    ctx.storage = f"sqlite:///{(ROOT / 'experiments' / 'p5' / 'optuna.db').as_posix()}"
    ctx.smoke = False
    tuned, _ = best_tuned(ctx, "lgb", 15)
    params = full_params("lgb", tuned, 0.05)
    apps = load_apps()
    m = apps["is_test"].eq(0).values
    X = run.build_matrix(list(GROUPS), apps)
    y = apps.loc[m, "target"].astype(int).values
    with Timer() as t:
        res = sel.null_importance(X[m], y, n_actual=5, n_null=40, rounds=800, params=params,
                                  log=lambda s: print(s, flush=True))
    res.to_csv(ROOT / "experiments" / "null_importance_E130b.csv")
    keep = res.index[res["actual"] > res["null_q75"]].tolist()
    sel.save_featset("sel_tuned", keep, dict(rule="actual gain > null q75, params LGB tuned lr 0.05, 800 rounds",
                                             n_null=40))
    print(f"selesai ({t.minutes:.1f} mnt): sel_tuned {len(keep)}/{X.shape[1]} kolom", flush=True)


if __name__ == "__main__":
    main()
