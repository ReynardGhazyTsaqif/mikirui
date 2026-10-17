"""Bangun cache grup fitur eksperimen 17 Okt+ (EXTRA_GROUPS) ke features/<nama>.parquet.

Grup ini tidak masuk GROUPS (set acuan "all" tetap 528 kolom); pakai lewat e121.py --extra <nama>.parquet.

  python build_extra.py rp_events_v1            # bangun (lewati jika cache sudah ada)
  python build_extra.py rp_events_v1 --rebuild  # bangun ulang
"""
import argparse
import sys

from src.features.common import cached
from src.features.registry import EXTRA_GROUPS
from src.io_utils import load_apps
from src.train import Timer


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(EXTRA_GROUPS))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    apps = load_apps()
    for n in a.names:
        with Timer() as t:
            df = cached(n, EXTRA_GROUPS[n], apps, rebuild=a.rebuild)
        print(f"{n}: {df.shape[1]} kolom, {df.notna().any(axis=1).mean():.1%} baris terisi ({t.minutes:.1f} mnt)")
        print(df.describe().T.round(4).to_string())


if __name__ == "__main__":
    main()
