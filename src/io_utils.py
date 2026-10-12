"""Loader hemat memori untuk semua tabel di Dataset/."""
import pandas as pd

from .config import DATA_DIR

_DATES = {
    "train_applications": ["application_datetime", "decision_datetime"],
    "test_applications": ["application_datetime", "decision_datetime"],
    "previous_applications": ["application_datetime", "decision_datetime"],
    "repayment_history": ["due_date", "last_payment_datetime"],
    "bureau_observations": ["report_datetime", "available_datetime"],
    "account_behavior": ["month"],
    "ecommerce_transactions": ["transaction_datetime"],
}

_CATS = {
    "previous_applications": ["decision"],
    "bureau_observations": ["bureau_provider"],
    "ecommerce_transactions": ["product_category", "payment_method", "order_status"],
}


def load(name, usecols=None, data_dir=DATA_DIR):
    """Baca satu tabel dengan parse tanggal dan dtype kategori.

    Kolom ID string (customer_id, order_id, ...) dibiarkan object karena dipakai untuk join.
    """
    dates = [c for c in _DATES.get(name, []) if usecols is None or c in usecols]
    dtype = {c: "category" for c in _CATS.get(name, []) if usecols is None or c in usecols}
    return pd.read_csv(data_dir / f"{name}.csv", usecols=usecols, parse_dates=dates, dtype=dtype)


def load_apps(data_dir=DATA_DIR):
    """Train + test digabung (test punya target NaN), diurutkan train dulu lalu test."""
    tr = load("train_applications", data_dir=data_dir)
    te = load("test_applications", data_dir=data_dir)
    tr["is_test"] = 0
    te["is_test"] = 1
    return pd.concat([tr, te], ignore_index=True)


def cutoffs(apps):
    """Peta customer_id -> application_id dan cutoff (= decision_datetime aplikasi saat ini)."""
    return apps[["customer_id", "application_id", "decision_datetime"]].rename(
        columns={"decision_datetime": "cutoff"}
    )
