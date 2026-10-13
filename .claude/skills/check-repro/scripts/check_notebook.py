"""Cek statis notebook final DSC LOGIKA UI 2026 (tanpa menjalankannya).

Memeriksa nama file, self-contained (tanpa src/ atau cache), determinisme,
penulisan submission, sel error, bukti Run All berurutan, dan markdown.
Hanya butuh library standar.

Contoh:
    python .claude/skills/check-repro/scripts/check_notebook.py final/DSC0001_RandomSeed_Notebook.ipynb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

NAME_RE = re.compile(r"^DSC\d+_.+_Notebook\.ipynb$")

CODE_RULES = [
    # (level, regex, pesan)
    ("ERROR", r"^\s*(from\s+src(\.|\s)|import\s+src\b)", "impor dari src/ - notebook harus self-contained (inline kodenya)"),
    ("ERROR", r"""['"][^'"]*\b(features|oof|preds|experiments)[/\\][^'"]*['"]""",
     "membaca/menulis cache proyek (features/, oof/, preds/, experiments/) - hitung dari data mentah"),
    ("ERROR", r"\b(np\.load|joblib\.load|pickle\.load|pd\.read_pickle)\s*\(", "memuat artefak tersimpan - notebook harus melatih ulang dari data mentah"),
    ("ERROR", r"\btime_limit\s*=", "time_limit membuat hasil tidak deterministik"),
    ("ERROR", r"\bRUN_TUNING\s*=\s*True\b", "RUN_TUNING harus False di notebook final"),
    ("WARN", r"https?://", "ada URL - pastikan tidak mengunduh data/model dari luar"),
    ("WARN", r"(!|%)pip\s+install|python\s+-m\s+pip\s+install", "pip install di notebook - butuh internet; cek environment panitia/Kaggle"),
    ("WARN", r"\bfor\s+\w+\s+in\s+set\(", "iterasi set() - urutan bisa berubah antar proses; pakai sorted()"),
    ("WARN", r"\btorch\b", "torch dipakai - wajib torch.manual_seed + use_deterministic_algorithms, CPU"),
]

PATH_RE = re.compile(r"""['"]([A-Za-z]:\\\\|[A-Za-z]:/|/home/|/Users/|C:\\)""")


def src_of(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--min-markdown", type=int, default=12, help="minimal sel markdown (12 bagian di planning)")
    a = ap.parse_args()

    issues = []

    def add(level, where, msg):
        issues.append((level, where, msg))

    name = os.path.basename(a.notebook)
    if not NAME_RE.match(name):
        add("ERROR", "nama", f"'{name}' tidak sesuai format NomorPeserta_NamaTim_Notebook.ipynb (mis. DSC0001_RandomSeed_Notebook.ipynb)")
    try:
        with open(a.notebook, encoding="utf-8") as fh:
            nb = json.load(fh)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ERROR: notebook tidak bisa dibaca sebagai JSON: {e}")
    if nb.get("nbformat") != 4:
        add("WARN", "format", f"nbformat {nb.get('nbformat')} (diharapkan 4)")

    cells = nb.get("cells", [])
    code = [(i, c) for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    md = [c for c in cells if c.get("cell_type") == "markdown" and src_of(c).strip()]
    all_code = "\n".join(src_of(c) for _, c in code)

    for i, c in code:
        s = src_of(c)
        for lvl, rx, msg in CODE_RULES:
            if re.search(rx, s, flags=re.MULTILINE):
                add(lvl, f"sel {i}", msg)
        for line in s.splitlines():
            if PATH_RE.search(line) and "DATA_DIR" not in line:
                add("WARN", f"sel {i}", f"path absolut di luar DATA_DIR: {line.strip()[:80]}")
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                add("ERROR", f"sel {i}", f"output error: {o.get('ename')}: {str(o.get('evalue'))[:80]}")

    first = "\n".join(src_of(c) for _, c in code[:3])
    if "DATA_DIR" not in first:
        add("ERROR", "setup", "DATA_DIR tidak didefinisikan di 3 sel kode pertama")
    if not re.search(r"\bSEED\s*=\s*42\b", all_code):
        add("ERROR", "setup", "SEED = 42 tidak ditemukan")
    if "__version__" not in first and "version" not in first:
        add("WARN", "setup", "versi library tidak dicetak di awal")
    if re.search(r"import\s+lightgbm|from\s+lightgbm|\blgb\.(train|LGBM|Dataset)", all_code):
        if "deterministic" not in all_code:
            add("WARN", "lightgbm", "deterministic=True tidak ditemukan")
        if not re.search(r"num_threads|n_jobs", all_code):
            add("WARN", "lightgbm", "jumlah thread tidak dikunci (num_threads)")
    if "catboost" in all_code.lower() and "thread_count" not in all_code:
        add("WARN", "catboost", "thread_count tidak dikunci")
    if "xgb" in all_code.lower() and "nthread" not in all_code and "n_jobs" not in all_code:
        add("WARN", "xgboost", "nthread/n_jobs tidak dikunci")
    if "RUN_TUNING" in all_code and not re.search(r"\bRUN_TUNING\s*=\s*False\b", all_code):
        add("ERROR", "tuning", "RUN_TUNING tidak di-set False")

    writes = re.findall(r"\.to_csv\(([^)]*)\)", all_code)
    if not writes:
        add("ERROR", "submission", "tidak ada to_csv - notebook harus menulis submission.csv")
    else:
        if not any("lineterminator" in w for w in writes):
            add("ERROR", "submission", "to_csv tanpa lineterminator='\\n' - MD5 Windows (CRLF) != Kaggle (LF)")
        if not re.search(r"round\(\s*6\s*\)", all_code):
            add("WARN", "submission", "probabilitas tidak dibulatkan round(6)")
    if "md5" not in all_code.lower():
        add("ERROR", "submission", "MD5 submission tidak dicetak")
    if "sample_submission" not in all_code:
        add("WARN", "submission", "tidak merge ke sample_submission berdasarkan application_id")

    counts = [c.get("execution_count") for _, c in code if src_of(c).strip()]
    if not counts or any(x is None for x in counts):
        add("WARN", "run all", "ada sel kode belum dijalankan - lakukan Restart & Run All sebelum dikirim")
    elif counts != list(range(counts[0], counts[0] + len(counts))) or counts[0] != 1:
        add("WARN", "run all", f"execution_count tidak berurutan dari 1 ({counts[:6]}...) - bukan bukti Run All bersih")

    if len(md) < a.min_markdown:
        add("WARN", "markdown", f"hanya {len(md)} sel markdown (target >= {a.min_markdown}; penjelasan & analisis dinilai)")

    errs = [x for x in issues if x[0] == "ERROR"]
    warns = [x for x in issues if x[0] == "WARN"]
    print(f"Notebook: {a.notebook}")
    print(f"Sel     : {len(code)} kode, {len(md)} markdown")
    print(f"ERROR {len(errs)}, WARN {len(warns)}")
    for lvl, where, msg in errs + warns:
        print(f"  [{lvl}] {where}: {msg}")
    print("STATUS: " + ("PERLU PERBAIKAN" if errs else "cek statis lolos - lanjut fresh run 2x dan cocokkan MD5"))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
