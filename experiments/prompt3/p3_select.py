"""PROMPT 3 Langkah 7: tabel semua submission dengan nilai_harapan = 0,45 x public_terlihat + 0,45 x perkiraan_private.

perkiraan_private = regresi public ~ mean-fold AUC (semua submission yang punya keduanya, CV >= 0,82), ditambah
koreksi jenis kandidat = rata-rata residual jenis itu yang disusutkan ke 0 (n / (n + 3)). Jenis: blend dengan NN,
blend dengan DART, tunggal GBDT. Repro: GPU bila memakai fitur urutan GRU (E122*, E202, E204, E210, ...) atau NN.

  python experiments/prompt3/p3_select.py [--extra "file.csv=0.84xxx"]   # kandidat belum di-submit (public kosong)
Keluaran: experiments/prompt3/nilai_harapan.md
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GPU_TAGS = ("E122", "E202", "E204", "E210", "E212", "E213", "E214", "NN", "E141x", "GBDT3", "E145")


def parse_log():
    rows = []
    for ln in (ROOT / "submissions" / "LOG.md").read_text(encoding="utf-8").splitlines():
        if not ln.startswith("|") or ".csv" not in ln:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        f = next((m.group(0) for c in cells for m in [re.search(r"[\w.\-]+\.csv", c)] if m), None)
        md5 = next((m.group(0) for c in cells for m in [re.search(r"\b[0-9a-f]{32}\b", c)] if m), "")
        pub = [c for c in cells if re.fullmatch(r"0\.5\d{2,6}", c)]
        cv = next((float(m.group(0)) for c in cells for m in [re.search(r"0\.8\d{3,5}", c)] if m), np.nan)
        cvf = re.search(r"auc(0\.8\d{3,5})", f or "")
        if cvf:
            cv = float(cvf.group(1))
        txt = ln.upper()
        kind = "nn" if ("NN" in txt and "TANPA NN" not in txt and "NNSEQ_ONLY" not in (f or "").upper()) else (
            "dart" if ("DART" in txt or "E141" in (f or "")) else "tunggal")
        if "NNSEQ_ONLY" in (f or "").upper():
            kind = "nn"
        repro = "GPU" if any(t in (f or "") or t in ln for t in GPU_TAGS) else "CPU"
        rows.append(dict(file=f, md5=md5[:8], cv=cv, public=float(pub[-1]) if pub else np.nan, jenis=kind, repro=repro))
    return pd.DataFrame(rows).drop_duplicates("file", keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="file.csv=AUC untuk kandidat yang belum di-submit")
    a = ap.parse_args()
    d = parse_log()
    for e in a.extra:
        f, cv = e.split("=")
        d = pd.concat([d, pd.DataFrame([dict(file=f, md5="", cv=float(cv), public=np.nan, jenis="tunggal",
                                              repro="GPU" if any(t in f for t in GPU_TAGS) else "CPU")])])
    fit = d[d.cv.notna() & d.public.notna() & (d.cv >= 0.82)]
    b, a0 = np.polyfit(fit.cv, fit.public, 1)
    res = fit.public - (a0 + b * fit.cv)
    corr = {k: g.mean() * len(g) / (len(g) + 3) for k, g in res.groupby(fit.jenis)}
    print(f"regresi public = {a0:.3f} + {b:.3f} x AUC (n {len(fit)}, sd residual {res.std():.4f}); "
          f"koreksi jenis (disusutkan): { {k: round(v, 4) for k, v in corr.items()} }")
    d["perkiraan_private"] = a0 + b * d.cv + d.jenis.map(corr).fillna(0)
    d["nilai_harapan"] = 0.45 * d.public + 0.45 * d.perkiraan_private
    d["nilai_harapan_jika_belum"] = 0.45 * d.perkiraan_private + 0.45 * d.perkiraan_private
    out = d[d.cv.notna()].sort_values("nilai_harapan", ascending=False)
    cols = ["file", "md5", "cv", "public", "perkiraan_private", "nilai_harapan", "jenis", "repro"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in out.iterrows():
        md.append("| " + " | ".join("" if (isinstance(r[c], float) and np.isnan(r[c])) else
                                    (f"{r[c]:.5f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    txt = (f"Regresi public = {a0:.3f} + {b:.3f} × AUC (n {len(fit)}, sd residual {res.std():.4f}); koreksi jenis: "
           f"{ {k: round(v, 4) for k, v in corr.items()} }\n\n" + "\n".join(md) + "\n")
    (ROOT / "experiments" / "prompt3" / "nilai_harapan.md").write_text(txt, encoding="utf-8")
    print(out[cols].head(15).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
