"""Langkah 2c: struktur baris biro dan rekening."""
import numpy as np
import pandas as pd

from common import apps, load

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
A = apps()
PA = load("previous_applications")
BU = load("bureau_observations")
print("== bureau", BU.shape)
print(BU.head(6).T)
cur_ids = set(A.application_id)
BU["is_cur"] = BU.application_id.isin(cur_ids)
print("laporan aplikasi sekarang:", BU.is_cur.sum(), "| aplikasi lama:", (~BU.is_cur).sum())
allapp = pd.concat([A[["application_id", "application_datetime", "decision_datetime"]],
                    PA[["application_id", "application_datetime", "decision_datetime"]]])
B = BU.merge(allapp, on="application_id")
B["rep_vs_app_d"] = (B.report_datetime - B.application_datetime).dt.total_seconds() / 86400
B["av_vs_app_d"] = (B.available_datetime - B.application_datetime).dt.total_seconds() / 86400
B["av_vs_dec_d"] = (B.available_datetime - B.decision_datetime).dt.total_seconds() / 86400
B["lag_d"] = (B.available_datetime - B.report_datetime).dt.total_seconds() / 86400
for nm, g in B.groupby("is_cur"):
    print(f"is_cur={nm}: report-app (hari) q:", g.rep_vs_app_d.quantile([0, .01, .5, .99, 1]).round(2).tolist(),
          "| avail-app q:", g.av_vs_app_d.quantile([0, .01, .5, .99, 1]).round(2).tolist(),
          "| avail-dec maks:", g.av_vs_dec_d.max().round(3), "| lag q:", g.lag_d.quantile([0, .5, .99, 1]).round(2).tolist())
print("jumlah provider per aplikasi:", BU.groupby("application_id").size().value_counts().to_dict())
print("aplikasi lama punya laporan:", PA.application_id.isin(BU.application_id).mean().round(4),
      "per keputusan:", PA.assign(h=PA.application_id.isin(BU.application_id)).groupby("decision").h.mean().round(3).to_dict())
# korelasi skor antar provider pada aplikasi sama
piv = BU[BU.is_cur].pivot(index="application_id", columns="bureau_provider", values="score")
print("korelasi skor antar provider:\n", piv.corr().round(3))
print("sd selisih A-B:", (piv["Bureau_A"] - piv["Bureau_B"]).std().round(2), "sd skor:", piv.stack().std().round(2))
print("skor bulat?", (BU.score.dropna() % 1 == 0).mean())
for c in ["active_accounts", "inquiries_last_6m", "delinquent_accounts_last_12m"]:
    print(c, BU[c].value_counts().sort_index().head(12).to_dict())
print("util = rev_bal/limit?", np.nanmax(np.abs(BU.utilization - BU.revolving_outstanding_balance / BU.revolving_credit_limit.replace(0, np.nan))))
print("limit==0 share:", (BU.revolving_credit_limit == 0).mean().round(3), "| util NaN share:", BU.utilization.isna().mean().round(3))
piv2 = BU[BU.is_cur].pivot(index="application_id", columns="bureau_provider", values="outstanding_balance")
print("korelasi outstanding antar provider:\n", piv2.corr().round(3))
piv3 = BU[BU.is_cur].pivot(index="application_id", columns="bureau_provider", values="active_accounts")
print("korelasi active_accounts antar provider:\n", piv3.corr().round(3))

AB = load("account_behavior")
print("\n== account_behavior", AB.shape)
print(AB.head(8).to_string())
AB = AB.sort_values(["customer_id", "month"]).reset_index(drop=True)
prev_close = AB.groupby("customer_id").closing_balance.shift()
print("opening == closing bulan sebelumnya:", (AB.opening_balance == prev_close)[prev_close.notna()].mean().round(4))
print("closing - opening == inflows - outflows:", ((AB.closing_balance - AB.opening_balance) == (AB.inflows - AB.outflows)).mean().round(4))
print("min <= avg:", (AB.minimum_balance <= AB.average_balance).mean().round(4), "| min <= closing:", (AB.minimum_balance <= AB.closing_balance).mean().round(4),
      "| min <= opening:", (AB.minimum_balance <= AB.opening_balance).mean().round(4))
print("income_like <= inflows:", (AB.income_like_credits <= AB.inflows).mean().round(4))
print("low_days>0 iff min<250k:", ((AB.low_balance_days > 0) == (AB.minimum_balance < 250000)).mean().round(4))
print("bulan berurutan tanpa lubang:", (AB.groupby("customer_id").month.diff().dt.days.dropna().between(28, 31)).mean().round(4))
n = AB.groupby("customer_id").size()
print("jumlah bulan per customer:", n.value_counts().sort_index().to_dict())
cut = A.set_index("customer_id").decision_datetime
last = AB.groupby("customer_id").month.max()
first = AB.groupby("customer_id").month.min()
mago_last = (cut.reindex(last.index).dt.year * 12 + cut.reindex(last.index).dt.month) - (last.dt.year * 12 + last.dt.month)
print("bulan terakhir = bulan sebelum keputusan:", (mago_last == 1).mean())
# apakah awal data rekening terkait aplikasi lama pertama?
pa_first = PA.groupby("customer_id").decision_datetime.min()
x = pd.DataFrame({"first": first, "pa_first": pa_first.reindex(first.index), "n": n})
x["diff_m"] = (x["first"].dt.year * 12 + x["first"].dt.month) - (x.pa_first.dt.year * 12 + x.pa_first.dt.month)
print("bulan pertama rekening - bulan aplikasi lama pertama:", x.diff_m.describe().round(2).to_dict())
print("negatif saldo:", (AB.closing_balance < 0).mean().round(4), "min saldo q:", AB.minimum_balance.quantile([0, .01, .1, .5]).tolist())
