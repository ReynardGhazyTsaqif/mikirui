"""Ubah skrip ber-sel (`# %%` dan `# %% [markdown]`) menjadi .ipynb tanpa dependensi tambahan.

  python kaggle/py2ipynb.py kaggle/e122_gru.py      # -> kaggle/e122_gru.ipynb
  python kaggle/py2ipynb.py kaggle/e122_gru.py --out kaggle/e122_bigru.ipynb --set VARIANT='"bigru"'
--set NAMA=NILAI mengganti baris pertama `NAMA = ...` (varian notebook tanpa menyalin kode).
"""
import json
import sys
from pathlib import Path


def convert(src, out=None, sets=()):
    cells, kind, buf = [], None, []
    text = Path(src).read_text(encoding="utf-8").splitlines()
    for kv in sets:
        key, val = kv.split("=", 1)
        i = next(i for i, ln in enumerate(text) if ln.startswith(f"{key} = "))
        text[i] = f"{key} = {val}"

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        src_lines = [ln + "\n" for ln in lines]
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src_lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in text:
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(out) if out else Path(src).with_suffix(".ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"{out} ({len(cells)} sel)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    convert(a.src, a.out, a.set)
