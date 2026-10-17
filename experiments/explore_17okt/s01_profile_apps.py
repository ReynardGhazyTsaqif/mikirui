"""Langkah 2a: profil baris mentah aplikasi sekarang, profil customer, aplikasi lama."""
import numpy as np
import pandas as pd

from common import apps, load

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

A = apps()
tr = A[A.is_test == 0]
print("== aplikasi sekarang", A.shape)
print(A.head(3).T)
A["proc_h"] = (A.decision_datetime - A.application_datetime).dt.total_seconds() / 3600
print("lama proses (jam) quantiles:", A.proc_h.quantile([0, .01, .1, .5, .9, .99, 1]).round(2).tolist())
print("rentang decision:", A.decision_datetime.min(), A.decision_datetime.max())
for c in ["annual_interest_rate", "term_months", "loan_purpose"]:
    print(c, A[c].value_counts().sort_index().to_dict())
print("loan_amount quantiles:", A.loan_amount.quantile([0, .01, .1, .5, .9, .99, 1]).tolist())
print("loan_amount mod 1000 == 0:", (A.loan_amount % 1000 == 0).mean(), "mod 100000:", (A.loan_amount % 100000 == 0).mean())
r = A.annual_interest_rate / 12
ann = A.loan_amount * r / (1 - (1 + r) ** -12)
print("selisih anuitas maks:", (A.proposed_monthly_installment - ann).abs().max())
# jam/hari aplikasi & keputusan
for c in ["application_datetime", "decision_datetime"]:
    print(c, "jam:", A[c].dt.hour.value_counts().sort_index().to_dict())
    print(c, "hari:", A[c].dt.dayofweek.value_counts().sort_index().to_dict())
    print(c, "detik==0:", (A[c].dt.second == 0).mean(), "menit==0:", (A[c].dt.minute == 0).mean())
# target per tier bunga dan per bulan keputusan
print(tr.groupby("annual_interest_rate").target.agg(["mean", "size"]))
print(tr.groupby(tr.decision_datetime.dt.to_period("M")).target.agg(["mean", "size"]).T)
print("target per hari keputusan dalam bulan (5 kelompok):")
print(tr.groupby(pd.cut(tr.decision_datetime.dt.day, [0, 6, 12, 18, 24, 31])).target.agg(["mean", "size"]).T)
print("target per jam keputusan:", tr.groupby(tr.decision_datetime.dt.hour).target.mean().round(3).to_dict())

P = load("customer_profiles")
print("\n== customer_profiles", P.shape)
print(P.describe(include="all").T)
for c in ["region", "employment_type", "housing_status", "dependents"]:
    print(c, P[c].value_counts().to_dict())
print("income mod 1000==0:", (P.declared_monthly_income % 1000 == 0).mean(), "mod 50000:", (P.declared_monthly_income % 50000 == 0).mean())
X = tr.merge(P, on="customer_id")
X["lti"] = X.loan_amount / X.declared_monthly_income
X["pti"] = X.proposed_monthly_installment / X.declared_monthly_income
print("LTI quantiles:", X.lti.quantile([0, .01, .5, .99, 1]).round(3).tolist())
print("PTI quantiles:", X.pti.quantile([0, .01, .5, .99, 1]).round(3).tolist())
print("tenure > age*12-?:", ((X.employment_tenure_months / 12) > (X.age - 15)).mean())
print("korelasi spearman loan_amount~income:", X[["loan_amount", "declared_monthly_income"]].corr("spearman").iloc[0, 1].round(3))

PA = load("previous_applications")
print("\n== previous_applications", PA.shape)
print(PA.head(5).T)
print("per customer:", PA.groupby("customer_id").size().describe().round(2).to_dict())
print("decision:", PA.decision.value_counts().to_dict())
print("term_months:", PA.term_months.value_counts().sort_index().to_dict())
print("rate (approved):", PA.annual_interest_rate.value_counts().sort_index().to_dict())
ap = PA[PA.decision == "approved"]
print("approved/requested quantiles:", (ap.approved_amount / ap.requested_amount).quantile([0, .05, .25, .5, .75, 1]).round(4).tolist())
print("approved==requested:", (ap.approved_amount == ap.requested_amount).mean())
r = ap.annual_interest_rate / 12
ann = ap.approved_amount * r / (1 - (1 + r) ** -ap.term_months)
print("selisih anuitas pinjaman lama maks:", (ap.scheduled_monthly_installment - ann).abs().max())
PA["proc_h"] = (PA.decision_datetime - PA.application_datetime).dt.total_seconds() / 3600
print("lama proses lama per keputusan:", PA.groupby("decision").proc_h.describe().round(2))
print("rentang decision lama:", PA.decision_datetime.min(), PA.decision_datetime.max())
