"""Langkah 2h: apakah skor biro / tier bunga / keputusan lender adalah pengukuran berulang dari risiko laten yang stabil?

- struktur noise skor: per laporan independen, atau bias per provider yang menetap?
- pooled mean semua laporan (sekarang + lama) vs mean laporan sekarang saja
- tier bunga pinjaman lama vs sekarang
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 220)
A = apps()
BU = load("bureau_observations")
PA = load("previous_applications")
cur = set(A.application_id)
BU["is_cur"] = BU.application_id.isin(cur)
allapp = pd.concat([A[["application_id", "decision_datetime"]], PA[["application_id", "decision_datetime"]]])
BU = BU.merge(allapp, on="application_id")
cut = A.set_index("customer_id").decision_datetime
BU["age_d"] = (BU.customer_id.map(cut) - BU.report_datetime).dt.days
# struktur noise: residual skor terhadap rata-rata aplikasi; korelasi residual provider yang sama lintas aplikasi
BU["app_mean"] = BU.groupby("application_id").score.transform("mean")
BU["dev"] = BU.score - BU.app_mean
BU["n_app"] = BU.groupby("application_id").score.transform("count")
d = BU[BU.n_app == 3]
pv = d.pivot_table(index=["customer_id", "application_id"], columns="bureau_provider", values="dev")
# korelasi deviasi provider A antar dua aplikasi berbeda milik customer sama
x = d[d.bureau_provider == "Bureau_A"][["customer_id", "application_id", "dev", "decision_datetime"]]
x = x.sort_values(["customer_id", "decision_datetime"])
x["dev_next"] = x.groupby("customer_id").dev.shift(-1)
print("korelasi deviasi provider A antar aplikasi berurutan (bias provider menetap?):", x[["dev", "dev_next"]].corr().iloc[0, 1].round(4))
# perubahan skor laten menurut waktu: mean skor per aplikasi, korelasi antar aplikasi berurutan vs jarak waktu
am = BU.groupby("application_id").agg(customer_id=("customer_id", "first"), s=("score", "mean"), t=("decision_datetime", "first"),
                                      cur=("is_cur", "first")).dropna()
am = am.sort_values(["customer_id", "t"])
am["s_next"] = am.groupby("customer_id").s.shift(-1)
am["gap_d"] = (am.groupby("customer_id").t.shift(-1) - am.t).dt.days
z = am.dropna(subset=["s_next"])
z["gb"] = pd.cut(z.gap_d, [0, 60, 120, 240, 365, 1100])
print("korelasi mean skor aplikasi berurutan per jarak (hari):", z.groupby("gb", observed=True).apply(lambda g: g[["s", "s_next"]].corr().iloc[0, 1]).round(3).to_dict())
print("sd selisih skor berurutan per jarak:", z.groupby("gb", observed=True).apply(lambda g: (g.s_next - g.s).std()).round(2).to_dict())
# tren: rata-rata selisih (s_next - s) per target customer
ref = reference("E122x1")
tgt = ref.set_index("customer_id").target
z["target"] = z.customer_id.map(tgt)
print("rata-rata perubahan skor antar aplikasi berurutan per target:", z.groupby("target").apply(lambda g: (g.s_next - g.s).mean()).round(2).to_dict())
# fitur pooled
g_all = BU.groupby("customer_id")
g_cur = BU[BU.is_cur].groupby("customer_id")
g_prev = BU[~BU.is_cur].groupby("customer_id")
F = {
    "score_cur_mean": g_cur.score.mean(),
    "score_all_mean": g_all.score.mean(),
    "score_prev_mean": g_prev.score.mean(),
    "n_scores_all": g_all.score.count(),
    "delinq_all_mean": g_all.delinquent_accounts_last_12m.mean(),
    "inq_all_mean": g_all.inquiries_last_6m.mean(),
    "util_all_mean": g_all.utilization.mean(),
}
# precision-weighted: laten ~ N(mu, 48^2), noise per laporan sd 23 -> posterior mean
mu0, tau2, s2 = BU.score.mean(), 48.0 ** 2, 23.0 ** 2
n = g_all.score.count()
F["score_all_post"] = (mu0 / tau2 + g_all.score.sum() / s2) / (1 / tau2 + n / s2)
ap = PA[PA.decision == "approved"]
F["rate_prev_mean"] = ap.groupby("customer_id").annual_interest_rate.mean()
tier = A.set_index("customer_id").annual_interest_rate
allrate = pd.concat([ap[["customer_id", "annual_interest_rate"]], A[["customer_id", "annual_interest_rate"]]])
F["rate_all_mean"] = allrate.groupby("customer_id").annual_interest_rate.mean()
F["rate_cur"] = tier
F["n_rates"] = allrate.groupby("customer_id").size()
F["rej_share"] = PA.assign(r=(PA.decision == "rejected")).groupby("customer_id").r.mean()
print(table(ref, F).to_string(index=False))
