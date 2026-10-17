"""Utilitas bersama fungsi fitur.

Kontrak setiap build_<grup>(apps): input `apps` (train+test dari io_utils.load_apps),
output DataFrame ber-index application_id, kolom ber-prefix dan terurut (sorted).
"""
import pandas as pd

from ..config import FEATURES_DIR


def finalize(df, prefix):
    """Tambah prefix (jika belum), urutkan kolom, cast float64 -> float32 untuk hemat memori."""
    df = df.rename(columns={c: c if c.startswith(prefix) else f"{prefix}{c}" for c in df.columns})
    df = df[sorted(df.columns)]
    f64 = df.select_dtypes("float64").columns
    df[f64] = df[f64].astype("float32")
    return df


def by_customer(agg, apps, prefix):
    """Peta agregat per customer_id -> per application_id (customer tanpa data -> NaN)."""
    out = apps[["application_id", "customer_id"]].merge(
        agg, left_on="customer_id", right_index=True, how="left"
    ).set_index("application_id").drop(columns="customer_id")
    return finalize(out, prefix)


def cached(name, builder, apps, rebuild=False):
    """Cache hasil fitur ke features/<name>.parquet."""
    path = FEATURES_DIR / f"{name}.parquet"
    if path.exists() and not rebuild:
        df = pd.read_parquet(path)
        assert df.index.equals(pd.Index(apps["application_id"], name="application_id")), \
            f"Index cache {name} tidak cocok dengan apps; jalankan ulang dengan rebuild=True"
        return df
    df = builder(apps)
    df = df.reindex(apps["application_id"])
    df.index.name = "application_id"
    FEATURES_DIR.mkdir(exist_ok=True)
    df.to_parquet(path)
    return df


def days_between(later, earlier):
    return (later - earlier).dt.total_seconds() / 86400
