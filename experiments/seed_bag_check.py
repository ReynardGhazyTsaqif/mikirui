"""Uji: apakah kenaikan blend di OOF menyusut jika OOF tiap model dirata-rata 3 seed (meniru
efek bagging 5-fold pada prediksi test)? OOF per seed disimpan ke oof/<exp>_s<seed>.npy."""
import numpy as np, pandas as pd, run
from scipy.stats import rankdata
from src import selection as sel
from src.config import LGB_DEV, ROOT
from src.io_utils import load_apps
from src.train import run_lgb_cv, make_folds, ks_score

apps = load_apps(); m = apps.is_test.eq(0).values
y = apps.loc[m, 'target'].astype(int).values
fold = make_folds(apps[m]).fold.values
mf = lambda p: np.mean([__import__('sklearn.metrics', fromlist=['x']).roc_auc_score(y[fold == k], p[fold == k]) for k in range(5)])
bag = {}
for e in ['E71b', 'E71c', 'E87']:
    cfg = run.EXPERIMENTS[e]
    g, c, _ = run.resolve(cfg)
    X = run.build_matrix(g, apps)
    X = X[[k for k in X.columns if k in set(sel.load_featset(c))]]
    params = {**LGB_DEV, **cfg.get('params', {})}
    oofs = [np.load(ROOT / 'oof' / f'{e}.npy')]
    for s in [43, 44]:
        path = ROOT / 'oof' / f'{e}_s{s}.npy'
        if not path.exists():
            o, p, *_ = run_lgb_cv({**params, 'seed': s}, X[m], y, fold, X[~m])
            np.save(path, o); np.save(ROOT / 'preds' / f'{e}_s{s}.npy', p)
        oofs.append(np.load(path))
    bag[e] = (oofs[0], np.mean(oofs, axis=0))
    print(f'{e}: 1 seed AUC {mf(oofs[0]):.5f} KS {ks_score(y, oofs[0]):.4f} | 3 seed AUC {mf(bag[e][1]):.5f} KS {ks_score(y, bag[e][1]):.4f}', flush=True)
for lbl, i in [('1 seed', 0), ('3 seed', 1)]:
    b = sum(rankdata(bag[e][i]) for e in bag) / (3 * len(y))
    best = max(mf(bag[e][i]) for e in bag)
    print(f'{lbl}: blend 3 LGB AUC {mf(b):.5f} KS {ks_score(y, b):.4f} | model terbaik {best:.5f} | kenaikan blend {mf(b) - best:+.5f}')
