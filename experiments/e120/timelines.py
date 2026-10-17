"""E120 langkah 2: cetak timeline mentah nasabah yang paling salah diprediksi (FN = default tapi skor rendah, FP = tidak default tapi skor tinggi)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.io_utils import load
sys.stdout.reconfigure(encoding="utf-8")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
meta = pd.read_parquet("kaggle/dataset/apps_meta.parquet"); tr = meta[meta.is_test == 0].reset_index(drop=True)
p = np.load("experiments/e120/oof_rank.npy"); tr["p"] = p
rng = np.random.default_rng(0)
fn = tr[(tr.target == 1)].nsmallest(200, "p").sample(N, random_state=1)
fp = tr[(tr.target == 0)].nlargest(200, "p").sample(N, random_state=1)
tp = tr[(tr.target == 1)].nlargest(200, "p").sample(N // 2, random_state=1)
pick = pd.concat([fn.assign(kel="FN"), fp.assign(kel="FP"), tp.assign(kel="TP(benar)")])
cust = set(pick.customer_id)
apps = pd.concat([load("train_applications"), load("test_applications")]); apps = apps[apps.customer_id.isin(cust)].set_index("customer_id")
prof = load("customer_profiles"); prof = prof[prof.customer_id.isin(cust)].set_index("customer_id")
pa = load("previous_applications"); pa = pa[pa.customer_id.isin(cust)]
rp = load("repayment_history"); rp = rp[rp.customer_id.isin(cust)]
bu = load("bureau_observations"); bu = bu[bu.customer_id.isin(cust)]
ab = load("account_behavior"); ab = ab[ab.customer_id.isin(cust)]
ec = []
for ch in pd.read_csv("Dataset/ecommerce_transactions.csv", usecols=["customer_id", "order_id", "transaction_datetime", "product_category", "order_total", "payment_method", "order_status"], chunksize=1_000_000, parse_dates=["transaction_datetime"]):
    ec.append(ch[ch.customer_id.isin(cust)])
ec = pd.concat(ec).drop_duplicates("order_id")
M = 1e6
for _, r in pick.iterrows():
    c = r.customer_id; a = apps.loc[c]; cut = a.decision_datetime; pr = prof.loc[c]
    print("=" * 110)
    print(f"[{r.kel}] {c} skor persentil {r.p:.3f} target {int(r.target)} | keputusan {cut:%Y-%m-%d} pinjaman {a.loan_amount / M:.1f} jt, bunga {a.annual_interest_rate}, cicilan {a.proposed_monthly_installment / M:.2f} jt, tujuan {a.loan_purpose}")
    print(f"  profil: umur {pr.age}, {pr.employment_type} {pr.employment_tenure_months} bln, income {pr.declared_monthly_income / M:.1f} jt, tanggungan {pr.dependents}, {pr.housing_status}, {pr.region}")
    b = bu[bu.application_id == a.application_id]
    print("  biro sekarang: " + "; ".join(f"{x.bureau_provider[-1]} skor {x.score} aktif {x.active_accounts} utang {x.outstanding_balance / M:.1f}jt util {x.utilization if pd.notna(x.utilization) else '-'} inq {x.inquiries_last_6m} delinq {x.delinquent_accounts_last_12m}" for x in b.itertuples()))
    for x in pa[pa.customer_id == c].sort_values("decision_datetime").itertuples():
        r2 = rp[rp.loan_id == x.loan_id] if pd.notna(x.loan_id) else rp.iloc[:0]
        bp = bu[bu.application_id == x.application_id]
        s = f"  lama {x.decision_datetime:%Y-%m-%d} ({(cut - x.decision_datetime).days}h lalu) {x.decision:9s} minta {x.requested_amount / M:.1f}jt {x.term_months}bln"
        if x.decision == "approved":
            s += f" cair {x.approved_amount / M:.1f}jt bunga {x.annual_interest_rate}"
            if len(r2):
                s += f" | {len(r2)} cicilan, maxDPD per cicilan {r2.sort_values('due_date').max_days_past_due.tolist()}, DPD kini {r2.days_past_due.max()}, bayar {r2.amount_paid.sum() / r2.amount_due.sum():.2f}"
        if len(bp):
            s += f" | biro skor {bp.score.mean():.0f} delinq {bp.delinquent_accounts_last_12m.max()}"
        print(s)
    a2 = ab[ab.customer_id == c].sort_values("month").tail(6)
    if len(a2):
        print("  rekening 6 bln terakhir: " + " | ".join(f"{x.month:%y-%m} saldo {x.closing_balance / M:.1f} in {x.inflows / M:.1f} out {x.outflows / M:.1f} gaji {x.income_like_credits / M:.1f} low {x.low_balance_days}" for x in a2.itertuples()))
    e = ec[ec.customer_id == c].sort_values("transaction_datetime")
    e30 = e[e.transaction_datetime > cut - pd.Timedelta(days=30)]
    print(f"  e-commerce: {len(e)} order total, 30 hari terakhir {len(e30)}; 8 terakhir: " + " | ".join(f"-{(cut - x.transaction_datetime).days}h {x.product_category[:5]} {x.order_total / 1e3:.0f}rb {x.payment_method[:6]} {x.order_status[:4]}" for x in e.tail(8).itertuples()))
