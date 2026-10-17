"""Residual OOF model acuan (terkalibrasi) + uji sinyal tambahan untuk kandidat fitur (tanpa melatih model utama)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import apps, bag_oof

EPS = 1e-6


def reference(name="E122x1"):
    """DataFrame train: customer_id, application_id, target, fold, p (OOF bag), p_cal, r (residual), lg (logit)."""
    A = apps()
    tr = A[A.is_test == 0].reset_index(drop=True)
    o, n = bag_oof(name)
    assert len(o) == len(tr)
    lg = np.log(np.clip(o, EPS, 1 - EPS) / (1 - np.clip(o, EPS, 1 - EPS)))
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), tr.target)
    pc = cal.predict_proba(lg.reshape(-1, 1))[:, 1]
    out = tr[["customer_id", "application_id", "target", "fold"]].copy()
    out["p"], out["p_cal"], out["lg"] = o, pc, lg
    out["r"] = out.target - pc
    out.attrs["n_seed"] = n
    return out


def signal(ref, s, name="x"):
    """s: Series ber-index customer_id. Cetak AUC univariat, z residual (Pearson rank vs residual), selisih residual desil."""
    x = ref.customer_id.map(s).astype(float).values
    ok = ~np.isnan(x)
    y, r = ref.target.values, ref.r.values
    auc = roc_auc_score(y[ok], x[ok]) if ok.sum() > 100 else np.nan
    rho = np.corrcoef(rankdata(x[ok]), r[ok])[0, 1]
    z = rho * np.sqrt(ok.sum())
    dec = pd.qcut(pd.Series(rankdata(x[ok])), 10, labels=False, duplicates="drop")
    rd = pd.Series(r[ok]).groupby(dec.values).mean()
    return dict(fitur=name, n=int(ok.sum()), auc=round(max(auc, 1 - auc), 4), arah="+" if auc >= 0.5 else "-",
                z=round(z, 2), resid_desil_min=round(rd.min(), 4), resid_desil_max=round(rd.max(), 4))


def table(ref, feats):
    """feats: dict nama -> Series ber-index customer_id."""
    return pd.DataFrame([signal(ref, s, k) for k, s in feats.items()]).sort_values("z", key=np.abs, ascending=False)
