"""H02: model bantu PD pinjaman (x_auxpd_), dilatih pada pinjaman lama berlabel, bukti leave-one-loan-out.

Temuan eksplorasi 17 Okt (s14, s20): tiap customer punya risiko default laten yang stabil; label default antarpinjaman
customer yang sama berkorelasi (ICC 0,40) dan target sekarang berperilaku seperti "satu pinjaman lagi". Pinjaman lama
dengan jendela 12 bulan lengkap memberi ±80 rb label tambahan dari mekanisme yang sama.

Baris latih = pinjaman lama P yang jendela 12 bulannya lengkap sebelum cutoff (customer train DAN test, simetris).
Label = ada cicilan dengan due + 90 hari <= pencairan + 12 bulan kalender dan max_days_past_due >= 90.
Fitur baris P = bukti customer TANPA pinjaman itu:
  - hasil pinjaman lengkap lain (k, n, posterior beta-binomial), pinjaman lain yang masih berjalan,
  - rantai bayar pinjaman lain, TANPA event yang jatuh di jendela 12 bulan pinjaman P (guncangan bulanan bersama),
  - laporan biro as-of keputusan P (laporan sesudahnya bisa memuat tunggakan P sendiri),
  - profil disesuaikan ke waktu P, ciri pinjaman P, aplikasi lama lain,
  - sifat e-commerce stabil (tanpa 90 hari terakhir sebelum keputusan sekarang = zona ramp) dan rekening (tanpa 3 bulan terakhir).
Baris skor = aplikasi sekarang (train + test) dengan bukti penuh sebelum cutoff (definisi fitur sama).
Target sekarang TIDAK pernah dibaca -> fitur tidak perlu nested. Model: LightGBM, iterasi dari GroupKFold per customer,
refit 3 seed pada semua baris P.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ..config import DATA_DIR, N_THREADS, SEED
from ..io_utils import cutoffs, load
from .common import finalize
from .events import _beta_prior, _events

PRIOR_A, PRIOR_B = 0.186, 1.330  # Beta prior risiko default per customer (ICC 0,40; s14)
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1,
              deterministic=True, force_row_wise=True, num_threads=N_THREADS)


def _cal(ts):
    return ts.dt.year * 12 + ts.dt.month - 1


def _rows(apps, rp, pa):
    """Baris P (pinjaman lama lengkap, berlabel) dan C (aplikasi sekarang)."""
    cut = cutoffs(apps)[["customer_id", "application_id", "cutoff"]]
    ap = pa[pa.decision == "approved"].merge(cut[["customer_id", "cutoff"]], on="customer_id")
    end12 = ap.decision_datetime + pd.DateOffset(months=12)
    ap = ap[end12 <= ap.cutoff].copy()
    r = rp.merge(ap[["loan_id", "decision_datetime"]], on="loan_id")
    elig = (r.due_date + pd.Timedelta(days=90)) <= (r.decision_datetime + pd.DateOffset(months=12))
    lab = (elig & (r.max_days_past_due >= 90)).groupby(r.loan_id).any()
    ap["label"] = ap.loan_id.map(lab).fillna(False).astype(int)
    ap = ap[ap.loan_id.isin(lab.index)]  # punya cicilan
    P = pd.DataFrame({"row": "P:" + ap.loan_id, "customer_id": ap.customer_id.values, "t": ap.decision_datetime.values,
                      "excl_loan": ap.loan_id.values, "excl_app": ap.application_id.values,
                      "term": ap.term_months.values, "rate": ap.annual_interest_rate.values,
                      "amount": ap.approved_amount.values, "inst": ap.scheduled_monthly_installment.values,
                      "label": ap.label.values, "kind": 1})
    A = apps.merge(cut[["application_id", "cutoff"]], on="application_id")
    C = pd.DataFrame({"row": "C:" + A.application_id, "customer_id": A.customer_id.values, "t": A.cutoff.values,
                      "excl_loan": None, "excl_app": None, "term": A.term_months.values,
                      "rate": A.annual_interest_rate.values, "amount": A.loan_amount.values,
                      "inst": A.proposed_monthly_installment.values, "label": np.nan, "kind": 0})
    R = pd.concat([P, C], ignore_index=True)
    R["t"] = pd.to_datetime(R.t)
    R["t_cal"] = _cal(R.t)
    return R


def _loan_outcomes(R, ev, rp, pa, apps):
    """Hasil pinjaman lain (LOO) yang meniru posisi aplikasi sekarang.

    Untuk baris dengan waktu keputusan t (P: pencairan pinjaman P; C: cutoff):
      - lengkap & dihitung  : pinjaman lain yang jendela 12 bulannya TIDAK tumpang tindih dengan [t, t+12 bln)
                              (dicairkan <= t-12 bln, atau >= t+12 bln) dan hasilnya sudah teramati di cutoff sekarang;
      - berjalan saat t     : dicairkan di (t-12 bln, t) -> hanya tunggakan maksimum dari event sebelum t yang dilihat;
      - tumpang tindih sesudah t (dicairkan di [t, t+12 bln)) : tidak dipakai (guncangan bulanan bersama).
    Untuk baris C semua pinjaman lama dicairkan sebelum cutoff, jadi aturan ini sama dengan pemakaian biasa.
    """
    cut = cutoffs(apps)[["customer_id", "cutoff"]]
    ap = pa[pa.decision == "approved"].merge(cut, on="customer_id")
    end12 = ap.decision_datetime + pd.DateOffset(months=12)
    ap["complete"] = end12 <= ap.cutoff
    r = rp.merge(ap[["loan_id", "decision_datetime"]], on="loan_id")
    elig = (r.due_date + pd.Timedelta(days=90)) <= (r.decision_datetime + pd.DateOffset(months=12))
    lab = (elig & (r.max_days_past_due >= 90)).groupby(r.loan_id).any()
    ap["def"] = ap.loan_id.map(lab).fillna(False).astype(int)
    ap["hasrp"] = ap.loan_id.isin(lab.index)
    L = ap[["customer_id", "loan_id", "decision_datetime", "complete", "def", "hasrp"]].rename(
        columns={"decision_datetime": "dis"})
    pr = R[["row", "customer_id", "excl_loan", "t"]].merge(L, on="customer_id")
    pr = pr[pr.loan_id != pr.excl_loan]
    before = pr.dis <= pr.t - pd.DateOffset(months=12)
    after = pr.dis >= pr.t + pd.DateOffset(months=12)
    running = (pr.dis > pr.t - pd.DateOffset(months=12)) & (pr.dis < pr.t)
    use = (before | after) & pr.complete & pr.hasrp
    g = pr.groupby("row")
    out = pd.DataFrame({
        "n_other_loans": g.size(),
        "n_other_complete": use.groupby(pr.row).sum(),
        "k_other_default": (pr["def"] * use).groupby(pr.row).sum(),
        "n_other_running": (running & pr.hasrp).groupby(pr.row).sum(),
    })
    out["post_default"] = (out.k_other_default + PRIOR_A) / (out.n_other_complete + PRIOR_A + PRIOR_B)
    # tunggakan maksimum pinjaman yang berjalan saat t, hanya dari event sebelum t
    rr = pr.loc[running & pr.hasrp, ["row", "loan_id", "t"]]
    rr["t_cal"] = _cal(rr.t)
    e = rr.merge(ev[["loan_id", "cal", "A"]], on="loan_id")
    e = e[e.cal <= e.t_cal]
    out["max_arrears_running"] = e.groupby("row").A.max()
    out["arrears_running_now"] = e.sort_values("cal").groupby(["row", "loan_id"]).A.last().groupby(level=0).sum()
    return out


def _event_stats(R, ev):
    """Rantai bayar pinjaman lain, tanpa event di jendela 12 bulan baris P (dan tanpa pinjamannya sendiri)."""
    e = ev[["customer_id", "loan_id", "A_prev", "X", "miss", "cal", "A"]]
    pr = R[["row", "customer_id", "excl_loan", "t_cal", "kind"]].merge(e, on="customer_id")
    win = (pr.kind == 1) & (pr.cal > pr.t_cal) & (pr.cal <= pr.t_cal + 12)
    pr = pr[(pr.loan_id != pr.excl_loan) & ~win]
    at0 = pr.A_prev == 0
    behind_paid = (~at0) & (pr.X >= 1)
    g = pr.groupby("row")
    S = pd.DataFrame({"n_ev": g.size(), "n_miss": g.miss.sum(),
                      "n0": at0.groupby(pr.row).sum(), "m0": (pr.miss * at0).groupby(pr.row).sum(),
                      "n1": (~at0).groupby(pr.row).sum(), "m1": (pr.miss * ~at0).groupby(pr.row).sum(),
                      "p1": behind_paid.groupby(pr.row).sum(), "c1": (behind_paid & (pr.X >= 2)).groupby(pr.row).sum()})
    return S


def _bureau(R):
    bo = load("bureau_observations")
    pr = R[["row", "customer_id", "t"]].merge(bo, on="customer_id")
    pr = pr[pr.available_datetime <= pr.t]
    g = pr.groupby("row")
    return pd.DataFrame({"bu_n": g.size(), "bu_score_mean": g.score.mean(), "bu_score_min": g.score.min(),
                         "bu_delinq_max": g.delinquent_accounts_last_12m.max(),
                         "bu_delinq_mean": g.delinquent_accounts_last_12m.mean(),
                         "bu_inq_mean": g.inquiries_last_6m.mean(), "bu_util_mean": g.utilization.mean(),
                         "bu_out_mean": g.outstanding_balance.mean(), "bu_active_mean": g.active_accounts.mean()})


def _prev_apps(R, pa):
    pr = R[["row", "customer_id", "excl_app"]].merge(pa[["customer_id", "application_id", "decision"]], on="customer_id")
    pr = pr[pr.application_id != pr.excl_app]
    g = pr.groupby("row")
    return pd.DataFrame({"pa_n_other": g.size(), "pa_rej_other": (pr.decision == "rejected").groupby(pr.row).sum(),
                         "pa_canc_other": (pr.decision == "cancelled").groupby(pr.row).sum()})


def _stable_traits(apps):
    """Sifat e-commerce & rekening stabil per customer, di luar zona ramp keputusan sekarang."""
    cut = cutoffs(apps)[["customer_id", "cutoff"]]
    # hemat memori (RAM laptop 7,7 GB): ID sebagai kategori, cutoff dipetakan lewat kode kategori
    ec = pd.read_csv(DATA_DIR / "ecommerce_transactions.csv",
                     usecols=["order_id", "customer_id", "transaction_datetime", "order_total", "payment_method",
                              "order_status"], parse_dates=["transaction_datetime"],
                     dtype={"order_id": "category", "customer_id": "category", "payment_method": "category",
                            "order_status": "category"}).drop_duplicates("order_id")
    cats = ec.customer_id.cat.categories
    cutv = cut.set_index("customer_id").cutoff.reindex(cats).values
    age_d = (cutv[ec.customer_id.cat.codes.values] - ec.transaction_datetime.values) / np.timedelta64(1, "D")
    ec = ec[age_d > 90].copy()
    ec["customer_id"] = ec.customer_id.astype(str)
    done = ec.order_status == "completed"
    g = ec.groupby("customer_id")
    span = (g.transaction_datetime.max() - g.transaction_datetime.min()).dt.days.clip(lower=30)
    T = pd.DataFrame({"ec_cod_share": (ec.payment_method == "COD").groupby(ec.customer_id).mean(),
                      "ec_cc_share": (ec.payment_method == "credit_card").groupby(ec.customer_id).mean(),
                      "ec_cancel_share": (~done).groupby(ec.customer_id).mean(),
                      "ec_log_amount": np.log(ec.order_total[done]).groupby(ec.customer_id[done]).mean(),
                      "ec_orders_per_30d": g.size() / span * 30})
    ab = load("account_behavior").merge(cut, on="customer_id")
    mago = (ab.cutoff.dt.year * 12 + ab.cutoff.dt.month) - (ab.month.dt.year * 12 + ab.month.dt.month)
    ab = ab[mago > 3]
    prof = load("customer_profiles", usecols=["customer_id", "declared_monthly_income"]).set_index("customer_id")
    inc = ab.customer_id.map(prof.declared_monthly_income)
    T = T.join(pd.DataFrame({"ab_inc_ratio": (ab.income_like_credits / inc).groupby(ab.customer_id).mean(),
                             "ab_low_share": (ab.low_balance_days > 0).groupby(ab.customer_id).mean(),
                             "ab_out_ratio": (ab.outflows / inc).groupby(ab.customer_id).mean()}), how="outer")
    return T


def _features(apps):
    rp = load("repayment_history")
    pa = load("previous_applications")
    ev = _events(apps)
    ev["cal"] = _cal(ev.dis) + ev.j
    R = _rows(apps, rp, pa)
    R = R.set_index("row")
    F = R[["customer_id", "kind", "label", "term", "rate", "amount", "inst", "t"]].copy()
    F = F.join(_loan_outcomes(R.reset_index(), ev, rp, pa, apps))
    S = _event_stats(R.reset_index(), ev)
    a0, b0 = _beta_prior(S.m0.values, S.n0.values, 6)
    a1, b1 = _beta_prior(S.m1.values, S.n1.values, 4)
    am, bm = _beta_prior(S.n_miss.values, S.n_ev.values, 6)
    E = pd.DataFrame(index=S.index)
    E["ev_n"] = S.n_ev
    E["ev_miss_post"] = (S.n_miss + am) / (S.n_ev + am + bm)
    E["ev_q0_post"] = (S.m0 + a0) / (S.n0 + a0 + b0)
    E["ev_q1_post"] = (S.m1 + a1) / (S.n1 + a1 + b1)
    c_mu = S.c1.sum() / max(S.p1.sum(), 1)
    E["ev_catch_post"] = (S.c1 + 4 * c_mu) / (S.p1 + 4)
    F = F.join(E).join(_bureau(R.reset_index())).join(_prev_apps(R.reset_index(), pa))
    prof = load("customer_profiles")
    F = F.reset_index().merge(prof, on="customer_id", how="left").set_index("row")
    cutmap = cutoffs(apps).set_index("customer_id").cutoff
    yrs = (F.customer_id.map(cutmap) - F.t).dt.days / 365.25
    F["age_t"] = F.age - yrs
    F["tenure_t"] = F.employment_tenure_months - yrs * 12
    F["pti"] = F.inst / F.declared_monthly_income
    F["lti"] = F.amount / F.declared_monthly_income
    for c in ["region", "employment_type", "housing_status"]:
        F[c] = F[c].astype("category").cat.codes
    F = F.reset_index().merge(_stable_traits(apps), left_on="customer_id", right_index=True, how="left").set_index("row")
    drop = ["customer_id", "t", "age", "employment_tenure_months", "label", "kind"]
    feats = sorted(c for c in F.columns if c not in drop)
    return F, feats


def build_x_auxpd(apps, shuffle=False):
    """shuffle=True: audit label acak (label pinjaman lama diacak) -> fitur harus kehilangan nilai."""
    F, feats = _features(apps)
    P = F[F.kind == 1]
    C = F[F.kind == 0]
    y = P.label.values.astype(int)
    if shuffle:
        y = np.random.default_rng(SEED).permutation(y)
    groups = P.customer_id.values
    oof = np.zeros(len(P))
    its = []
    for tr, va in GroupKFold(5).split(P, y, groups):
        m = lgb.train({**PARAMS, "seed": SEED}, lgb.Dataset(P[feats].iloc[tr], y[tr]), 5000,
                      valid_sets=[lgb.Dataset(P[feats].iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(P[feats].iloc[va], num_iteration=m.best_iteration)
        its.append(m.best_iteration)
    print(f"  model bantu PD: {len(P)} pinjaman lama (default {y.mean():.3f}), AUC GroupKFold {roc_auc_score(y, oof):.4f}, "
          f"iterasi {its}", flush=True)
    n_it = int(np.mean(its) * 1.1)
    pred = np.zeros(len(C))
    for s in (SEED, SEED + 1, SEED + 2):
        m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(P[feats], y), n_it)
        pred += m.predict(C[feats]) / 3
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("  fitur terpenting:", ", ".join(f"{k} {v:.0f}" for k, v in imp.head(8).items()), flush=True)
    out = pd.DataFrame({"pd": pred, "logit": np.log(pred / (1 - pred))}, index=C.index.str[2:])
    out.index.name = "application_id"
    return finalize(out.reindex(apps.application_id), "x_auxpd_")
