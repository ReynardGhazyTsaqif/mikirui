"""Cek stabilitas kandidat B3 (E31, E70, E71b, E71c) pada seed 42/43/44. Tidak menulis ke log."""
import json
import numpy as np, pandas as pd, run
from src import selection as sel
from src.config import LGB_DEV, ROOT
from src.io_utils import load_apps
from src.train import run_lgb_cv, make_folds, ks_score

apps = load_apps(); m = apps.is_test.eq(0).values
y = apps.loc[m, 'target'].astype(int).values
fold = make_folds(apps[m]).fold.values
rows = []
for e in ['E31', 'E70', 'E71b', 'E71c']:
    cfg = run.EXPERIMENTS[e]
    g, c, _ = run.resolve(cfg if e != 'E31' else dict(base='E31'))
    X = run.build_matrix(g, apps)
    if c:
        keep = set(sel.load_featset(c)); X = X[[k for k in X.columns if k in keep]]
    for s in [42, 43, 44]:
        oof, _, aucs, _, _ = run_lgb_cv({**LGB_DEV, 'seed': s}, X[m], y, fold, X[~m])
        rows.append(dict(exp=e, n=X.shape[1], seed=s, mean_fold_auc=np.mean(aucs), oof_ks=ks_score(y, oof)))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows)
summ = r.groupby('exp').agg(n=('n', 'first'), auc_mean=('mean_fold_auc', 'mean'), auc_sd=('mean_fold_auc', 'std'),
                            ks_mean=('oof_ks', 'mean'), ks_sd=('oof_ks', 'std'))
base = r[r.exp == 'E31'].set_index('seed').mean_fold_auc
summ['menang_vs_E31_per_seed'] = [int((r[r.exp == e].set_index('seed').mean_fold_auc > base).sum()) for e in summ.index]
print(summ.round(5).to_string())
r.to_csv(ROOT / 'experiments' / 'seed_check_b3.csv', index=False)
