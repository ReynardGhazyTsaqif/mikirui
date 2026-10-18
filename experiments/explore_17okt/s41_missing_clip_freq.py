"""H23/H25 sisa (mentor ide 15 dan 17): pola missing lintas tabel, nilai di batas min/maks (dipotong), nilai nol,
approved > requested, dan frekuensi nilai. Uji terhadap residual OOF E202 (resid.py) dan target rate per pola.
"""
import numpy as np
import pandas as pd

from common import apps, load
from resid import reference, table

pd.set_option("display.width", 250)
ref = reference("E202")
A = apps()
cid_all = A.customer_id
F = {}
# ---------- pola missing lintas tabel (bitmask)
pa = load("previous_applications")
rp_c = set(load("repayment_history", usecols=["customer_id"]).customer_id)
ab_c = set(load("account_behavior", usecols=["customer_id"]).customer_id)
bu = load("bureau_observations", usecols=["application_id", "customer_id", "score", "utilization", "revolving_credit_limit"])
bp_c = set(bu[~bu.application_id.isin(A.application_id)].customer_id)
appr_c = set(pa[pa.decision == "approved"].customer_id)
flags = pd.DataFrame({"pa": cid_all.isin(set(pa.customer_id)), "appr": cid_all.isin(appr_c), "rp": cid_all.isin(rp_c),
                      "ab": cid_all.isin(ab_c), "bp": cid_all.isin(bp_c)}).astype(int)
flags.index = cid_all.values
mask = flags.pa * 16 + flags.appr * 8 + flags.rp * 4 + flags.ab * 2 + flags.bp
r = ref.set_index("customer_id")
t = pd.DataFrame({"pola": mask.reindex(r.index).values, "y": r.target.values, "res": r.r.values})
g = t.groupby("pola").agg(n=("y", "size"), target=("y", "mean"), resid=("res", "mean"))
g["z_resid"] = g.resid / np.sqrt(t.y.var() / g.n)
print("pola missing (bit: pa appr rp ab bp) -> n, target, residual:\n", g.round(4).to_string())
# ---------- nilai di batas min/maks dan nol
p = load("customer_profiles").set_index("customer_id")
for c in ["age", "employment_tenure_months", "dependents", "declared_monthly_income"]:
    v = p[c]
    F[f"clip_min_{c}"] = (v == v.min()).astype(float)
    F[f"clip_max_{c}"] = (v == v.max()).astype(float)
cur = bu[bu.application_id.isin(A.application_id)]
for c in ["score", "utilization"]:
    v = cur[c]
    F[f"clip_min_bu_{c}"] = (v == v.min()).groupby(cur.customer_id).max().astype(float)
    F[f"clip_max_bu_{c}"] = (v == v.max()).groupby(cur.customer_id).max().astype(float)
F["zero_rev_limit"] = (cur.revolving_credit_limit == 0).groupby(cur.customer_id).mean()
ab = load("account_behavior")
for c in ["inflows", "outflows", "income_like_credits", "minimum_balance", "low_balance_days"]:
    F[f"zero_ab_{c}"] = (ab[c] == 0).groupby(ab.customer_id).mean()
F["max_ab_low_days"] = (ab.low_balance_days == ab.low_balance_days.max()).groupby(ab.customer_id).mean()
a = A.set_index("customer_id")
F["clip_min_loan"] = (a.loan_amount == a.loan_amount.min()).astype(float)
F["clip_max_loan"] = (a.loan_amount == a.loan_amount.max()).astype(float)
appr = pa[pa.approved_amount.notna()]
print("approved > requested:", int((appr.approved_amount > appr.requested_amount).sum()), "dari", len(appr))
# ---------- frekuensi nilai (train + test)
for c in ["loan_amount"]:
    F[f"freq_{c}"] = a[c].map(a[c].value_counts())
for c in ["age", "employment_tenure_months", "declared_monthly_income"]:
    F[f"freq_{c}"] = p[c].map(p[c].value_counts())
F["freq_purpose_x_tier"] = (a.loan_purpose.astype(str) + "_" + a.annual_interest_rate.astype(str)).map(
    (a.loan_purpose.astype(str) + "_" + a.annual_interest_rate.astype(str)).value_counts())
F["freq_amount_x_tier"] = (a.loan_amount.astype(str) + "_" + a.annual_interest_rate.astype(str)).map(
    (a.loan_amount.astype(str) + "_" + a.annual_interest_rate.astype(str)).value_counts())
T = table(ref, F)
print(T.to_string(index=False))
print("\nprevalensi flag batas/nol (porsi customer train > 0):")
y = ref.target.values
for k, v in F.items():
    if k.startswith(("clip", "zero", "max")):
        x = np.nan_to_num(ref.customer_id.map(v).values) > 0
        if x.any():
            print(f"  {k:<32} porsi {x.mean():.4f} target {y[x].mean():.3f} vs {y[~x].mean():.3f} resid {ref.r.values[x].mean():+.4f}")
