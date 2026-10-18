"""PROMPT UTAMA Langkah 4.4 = PROMPT 3 Langkah 1 (sebagian): nilai ulang eksperimen lama dengan aturan keputusan 17 Okt.

Syarat: status lama kandidat/ditolak, delta > 0 terhadap acuan aslinya, OOF >= 3 seed di kedua sisi.
Keluaran: experiments/explore_17okt/reevaluasi_aturan_baru.csv
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CMP = ROOT / ".claude" / "skills" / "run-experiment" / "scripts" / "compare_oof.py"
OUT = Path(__file__).resolve().parent
# (kandidat, acuan asli, status lama, catatan)
PAIRS = [
    ("E130b", "E130a", "ditolak (+0,00036, 3/5)", "null importance ulang 333 kolom"),
    ("E141", "E130a", "kandidat (+0,00038, P 0,97)", "LGB DART 5000 ronde, set E130a"),
    ("E122r", "E122", "kandidat (+0,00019, P 0,97)", "seq_gru persentil per blok"),
    ("E122m", "E122", "kandidat", "rata-rata 3 varian urutan (1 kolom)"),
    ("E122c", "E122", "kandidat (+0,00044, P 1,00)", "3 kolom urutan gru/bigru/tf"),
    ("E122x1", "E122c", "kandidat (+0,00041, P 0,99)", "+ seqx_rp/ab/pa"),
    ("E122x2", "E122c", "kandidat (+0,00033)", "+ seqx_all"),
    ("E122x3", "E122c", "kandidat (+0,00041)", "+ seqx_rp/ab/pa + seqx_all"),
    ("E141x", "E122x1", "kandidat (+0,00020, P 0,85)", "DART pada set E122x1"),
    ("T_lgb_b3_4f3c7f_r15k", "T_lgb_b3_4f3c7f", "ditolak (+0,00009, 3/5) tapi dipakai default", "E140a: batas iterasi 15000 (B3)"),
    ("T_lgb_selb2_4f3c7f_r15k", "T_lgb_selb2_4f3c7f", "ditolak (+0,00041, 4/5)", "E140a: batas iterasi 15000 (sel_b2)"),
]
rows = []
for cand, base, old, note in PAIRS:
    js = OUT / f"_tmp_{cand}.json"
    cmd = [sys.executable, str(CMP), "--base", base, "--cand", cand, "--seeds", "42", "43", "44", "--json", str(js)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        rows.append(dict(kandidat=cand, acuan=base, status_lama=old, keputusan_baru="GAGAL: " + r.stderr[-200:]))
        continue
    d = json.loads(js.read_text(encoding="utf-8"))
    js.unlink()
    rows.append(dict(kandidat=cand, acuan=base, catatan=note, status_lama=old,
                     seed=f"{d['n_seed_base']}/{d['n_seed_cand']}",
                     auc_acuan=round(d["mean_fold_auc_base"], 5), auc_kandidat=round(d["mean_fold_auc_cand"], 5),
                     delta=round(d["delta"], 5), ci95=f"[{d['bootstrap_ci'][0]:+.5f}; {d['bootstrap_ci'][1]:+.5f}]",
                     p_delta_pos=round(d["p_delta_pos"], 3), fold_naik=f"{d['folds_up']}/5",
                     ks_acuan=round(sum(d["ks_base"]) / 5, 4), ks_kandidat=round(sum(d["ks_cand"]) / 5, 4),
                     keputusan_baru=d["decision"]))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows)
T.to_csv(OUT / "reevaluasi_aturan_baru.csv", index=False)
pd.set_option("display.width", 250)
print(T.drop(columns=["catatan"]).to_string(index=False))
